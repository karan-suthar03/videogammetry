"""Compares a videogammetry camera path with a COLMAP reconstruction.

Usage:
  python tools/compare_trajectory.py <colmap images.txt> <colmap cameras.txt> <frames.csv> [frame_step]

images.txt names must be <n>.jpg where video frame = n * frame_step (default 5),
e.g. from: ffmpeg -i video -vf "select=not(mod(n\\,5))" -fps_mode passthrough -start_number 0 %05d.jpg
frames.csv is written by `videogammetry <video> --dump <dir>`.

Reports, after a similarity alignment (rotation, translation, scale):
  - ATE: RMS camera position error, in % of the reference path's extent
  - scale over time: per-window similarity fits, relative to the whole-path fit
  - RPE over 1 s: direction error of the motion and its length ratio
  - focal length: COLMAP's estimate vs the field of view we assume
"""
import csv
import math
import sys

import numpy as np


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_colmap(images_txt, step):
    centers = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        frame = int(p[9].split(".")[0]) * step
        R = quat_to_rot(qw, qx, qy, qz)
        centers[frame] = -R.T @ np.array([tx, ty, tz])
    return centers


def read_ours(frames_csv):
    out, times = {}, {}
    for r in csv.DictReader(open(frames_csv)):
        if r["state"] != "tracking":
            continue
        f = int(r["frame"])
        out[f] = np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])])
        times[f] = float(r["time"])
    return out, times


def umeyama(src, dst):
    """s, R, t minimizing |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    images_txt, cameras_txt, frames_csv = sys.argv[1:4]
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    ref = read_colmap(images_txt, step)
    ours, times = read_ours(frames_csv)
    # Sparse paths (e.g. keyframes.csv): interpolate positions at the
    # reference frames between the nearest frames we have (at most 2 s apart).
    have = sorted(ours)
    if have:
        fps = (have[-1] - have[0]) / max(times[have[-1]] - times[have[0]], 1e-9) if len(have) > 1 else 30.0
        for f in sorted(ref):
            if f in ours or f < have[0] or f > have[-1]:
                continue
            i = np.searchsorted(have, f)
            a, b = have[i - 1], have[i]
            if (b - a) / fps > 2.0:
                continue
            w = (f - a) / (b - a)
            ours[f] = (1 - w) * ours[a] + w * ours[b]
            times[f] = (1 - w) * times[a] + w * times[b]
    common = sorted(set(ref) & set(ours))
    print(f"frames: {len(ref)} in COLMAP, {len(ours)} tracked by us, {len(common)} in both")
    if len(common) < 10:
        return
    A = np.array([ours[f] for f in common])
    B = np.array([ref[f] for f in common])
    extent = np.linalg.norm(B.max(0) - B.min(0))

    s, R, t = umeyama(A, B)
    err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    print(f"ATE (whole path, similarity-aligned): RMS {100 * math.sqrt((err ** 2).mean()) / extent:.1f}% "
          f"of path extent, median {100 * np.median(err) / extent:.1f}%, max {100 * err.max() / extent:.1f}%")

    # Local scale: similarity fit per window, relative to the global one.
    print("scale over time (window fit / whole-path fit; 1.00 = no drift):")
    window = 20.0
    t0, t1 = times[common[0]], times[common[-1]]
    w = t0
    while w + window <= t1 + 1e-9:
        sel = [f for f in common if w <= times[f] < w + window]
        if len(sel) >= 10:
            sw, _, _ = umeyama(np.array([ours[f] for f in sel]), np.array([ref[f] for f in sel]))
            # Our scale relative to truth is 1/sw (sw maps ours -> COLMAP).
            print(f"  {w:6.1f}-{w + window:6.1f} s: {s / sw:5.2f}")
        w += window

    # Relative motion over 1 s steps (COLMAP frames are every `step` frames).
    dir_err, len_ratio = [], []
    by_time = sorted(common, key=lambda f: times[f])
    for i, f in enumerate(by_time):
        target = times[f] + 1.0
        g = next((h for h in by_time[i + 1:] if times[h] >= target), None)
        if g is None:
            break
        da = R @ (ours[g] - ours[f]) * s
        db = ref[g] - ref[f]
        na, nb = np.linalg.norm(da), np.linalg.norm(db)
        if na < 1e-9 or nb < 1e-3 * extent:
            continue
        dir_err.append(math.degrees(math.acos(np.clip(da @ db / (na * nb), -1, 1))))
        len_ratio.append(na / nb)
    if dir_err:
        print(f"RPE over 1 s: motion direction error median {np.median(dir_err):.1f} deg; "
              f"length ratio median {np.median(len_ratio):.2f} (10-90%: {np.percentile(len_ratio, 10):.2f}-"
              f"{np.percentile(len_ratio, 90):.2f})")

    # Focal length.
    for line in open(cameras_txt):
        if line.startswith("#"):
            continue
        p = line.split()
        model, w_px, h_px, f_px = p[1], int(p[2]), int(p[3]), float(p[4])
        fov = math.degrees(2 * math.atan(0.5 * max(w_px, h_px) / f_px))
        extra = " ".join(p[7:])
        print(f"COLMAP camera: {model} {w_px}x{h_px}, f = {f_px:.1f} px -> {fov:.1f} deg across the long side"
              f" (we assume 65.0 deg = {0.5 * max(w_px, h_px) / math.tan(math.radians(32.5)):.1f} px); "
              f"distortion: {extra}")
        break


if __name__ == "__main__":
    main()
