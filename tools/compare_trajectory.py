"""Compares a videogammetry camera path with a COLMAP reconstruction.

Usage:
  python tools/compare_trajectory.py <colmap images.txt> <colmap cameras.txt> <frames.csv> [frame_step]

images.txt names must be <n>.jpg where video frame = n * frame_step (default 5),
e.g. from: ffmpeg -i video -vf "select=not(mod(n\\,5))" -fps_mode passthrough -start_number 0 %05d.jpg
frames.csv (per-frame poses while tracking) or keyframes.csv (final keyframe
poses) are written by `videogammetry <video> --dump <dir>`.

Reports, after a similarity alignment (rotation, translation, scale):
  - ATE: RMS camera position error, in % of the reference path's extent
  - scale over time: per-window similarity fits, relative to the whole-path fit
  - RPE over 1 s: direction error of the motion and its length ratio
  - the reference camera (focal length, distortion)
Needs numpy. tools/benchmark.py uses evaluate() and read_camera().
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
    """Camera centers by video frame. Image lines only (points lines, if
    present, have fewer or non-numeric fields and are skipped)."""
    centers = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 10 or not p[9].lower().endswith((".jpg", ".png")):
                continue
            try:
                qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
            except ValueError:
                continue
            frame = int(p[9].split(".")[0]) * step
            R = quat_to_rot(qw, qx, qy, qz)
            centers[frame] = -R.T @ np.array([tx, ty, tz])
    return centers


def read_camera(cameras_txt):
    """First camera: model, size, focal (px), field of view across the long side (deg), extra params."""
    for line in open(cameras_txt):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        w, h, f = int(p[2]), int(p[3]), float(p[4])
        extra = [float(x) for x in p[7:]]
        return {"model": p[1], "width": w, "height": h, "focal": f,
                "fov": math.degrees(2 * math.atan(0.5 * max(w, h) / f)),
                "k1": extra[0] if p[1] in ("SIMPLE_RADIAL", "RADIAL") and extra else 0.0}
    return None


def read_ours(path):
    out, times = {}, {}
    for r in csv.DictReader(open(path)):
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


def evaluate(images_txt, poses_csv, step=5, window=20.0):
    """Error metrics of a path (frames.csv / keyframes.csv) against a COLMAP model."""
    ref = read_colmap(images_txt, step)
    ours, times = read_ours(poses_csv)
    result = {"reference_frames": len(ref), "our_frames": len(ours)}
    # Sparse paths (keyframes): interpolate positions at the reference frames
    # between the nearest frames we have (at most 2 s apart).
    have = sorted(ours)
    if len(have) > 1:
        fps = (have[-1] - have[0]) / max(times[have[-1]] - times[have[0]], 1e-9)
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
    result["common_frames"] = len(common)
    if len(common) < 10:
        return result
    A = np.array([ours[f] for f in common])
    B = np.array([ref[f] for f in common])
    extent = np.linalg.norm(B.max(0) - B.min(0))

    s, R, t = umeyama(A, B)
    err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    result["ate_rms_pct"] = 100 * math.sqrt((err ** 2).mean()) / extent
    result["ate_median_pct"] = 100 * float(np.median(err)) / extent
    result["ate_max_pct"] = 100 * float(err.max()) / extent

    # Local scale: similarity fit per window, relative to the whole-path fit.
    scales = []
    t0, t1 = times[common[0]], times[common[-1]]
    w = t0
    while w + window <= t1 + 1e-9:
        sel = [f for f in common if w <= times[f] < w + window]
        if len(sel) >= 10:
            sw, _, _ = umeyama(np.array([ours[f] for f in sel]), np.array([ref[f] for f in sel]))
            scales.append((w, w + window, s / sw))  # our scale relative to the whole-path fit
        w += window
    result["scale_windows"] = scales
    if scales:
        result["scale_min"] = min(x[2] for x in scales)
        result["scale_max"] = max(x[2] for x in scales)

    # Relative motion over 1 s.
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
        result["rpe_direction_deg"] = float(np.median(dir_err))
        result["rpe_length_ratio"] = float(np.median(len_ratio))
        result["rpe_length_ratio_p10"] = float(np.percentile(len_ratio, 10))
        result["rpe_length_ratio_p90"] = float(np.percentile(len_ratio, 90))
    return result


def main():
    images_txt, cameras_txt, poses_csv = sys.argv[1:4]
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    r = evaluate(images_txt, poses_csv, step)
    print(f"frames: {r['reference_frames']} in COLMAP, {r['our_frames']} tracked by us, "
          f"{r['common_frames']} in both")
    if "ate_rms_pct" not in r:
        return
    print(f"ATE (whole path, similarity-aligned): RMS {r['ate_rms_pct']:.1f}% of path extent, "
          f"median {r['ate_median_pct']:.1f}%, max {r['ate_max_pct']:.1f}%")
    print("scale over time (window fit / whole-path fit; 1.00 = no drift):")
    for a, b, sc in r.get("scale_windows", []):
        print(f"  {a:6.1f}-{b:6.1f} s: {sc:5.2f}")
    if "rpe_direction_deg" in r:
        print(f"RPE over 1 s: motion direction error median {r['rpe_direction_deg']:.1f} deg; "
              f"length ratio median {r['rpe_length_ratio']:.2f} "
              f"(10-90%: {r['rpe_length_ratio_p10']:.2f}-{r['rpe_length_ratio_p90']:.2f})")
    cam = read_camera(cameras_txt)
    if cam:
        print(f"COLMAP camera: {cam['model']} {cam['width']}x{cam['height']}, f = {cam['focal']:.1f} px -> "
              f"{cam['fov']:.1f} deg across the long side; k1 {cam['k1']:.4f}")


if __name__ == "__main__":
    main()
