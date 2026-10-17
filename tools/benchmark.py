"""Runs the videogammetry CLI on the benchmark sequences and reports accuracy,
speed and map quality.

Usage:
  python tools/benchmark.py [--runs N] [--only name,name] [--label text]
                            [--cli path] [--no-save] [-- extra CLI args]

Sequences: benchmarks/benchmark.json. For sequences with a COLMAP reference,
the final map (keyframes after all refinement) and the live path (each frame's
pose when it was tracked) are compared with it (tools/compare_trajectory.py).
With background BA results vary a little from run to run: --runs N reports the
median (and the range of the final-map error). Results are printed as a table
and saved to benchmarks/results/<date>_<commit>[_label].json.
Needs numpy.
"""
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_trajectory  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLI = os.path.join(ROOT, "build", "windows", "apps", "desktop", "Release",
                           "videogammetry.exe" if os.name == "nt" else "videogammetry")


def grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def parse_log(log):
    r = {
        "frames": grab(r"frames:\s+(\d+)", log, int),
        "lost_frames": grab(r"lost (\d+) frames", log, int),
        "maps": grab(r"(\d+) map\(s\)", log, int),
        "tracked_per_frame": grab(r"\((\d+) map points/frame avg\)", log, int),
        "keyframes": grab(r"(\d+) keyframes, \d+ map points", log, int),
        "points": grab(r"quality: (\d+) points", log, int),
        "surface_variation": grab(r"surface variation median ([\d.]+)", log),
        "scattered_pct": grab(r"(\d+)% scattered", log, int),
        "noise_median_pct": grab(r"uncertainty \(1 px\) median ([\d.]+)%", log),
        "noise_p90_pct": grab(r"90th percentile ([\d.]+)%", log),
        "two_view_pct": grab(r"(\d+)% seen by only 2", log, int),
        "fov": grab(r"field of view ([\d.]+) deg", log),
        "k1": grab(r"radial distortion k1 (-?[\d.]+)", log),
        "background_passes": grab(r"bg ba:\s+(\d+) passes merged", log, int),
        "fps": grab(r"throughput: ([\d.]+) fps overall", log),
        "core_fps": grab(r"([\d.]+) fps core only", log),
        "video_fps": grab(r"video is ([\d.]+) fps", log),
        "stages_ms": {},
    }
    for name, total, avg in re.findall(r"^\s{2}(\S[\w*. ]*?)\s{2,}([\d.]+)\s+([\d.]+)", log, re.M):
        r["stages_ms"][name.strip()] = {"total": float(total), "avg": float(avg)}
    return r


def realtime(frames_csv, video_fps, queue=3):
    """Live-capture view of the per-frame times on the tracking thread
    (tracking + mapping; feature extraction runs ahead on other threads):
    frames arrive every 1/fps; a camera keeps at most `queue` waiting frames
    and drops new ones when full."""
    import csv as _csv
    costs = [float(r["track_ms"]) + float(r["mapping_ms"]) for r in _csv.DictReader(open(frames_csv))]
    if not costs or not video_fps:
        return {}
    budget = 1000.0 / video_fps
    s = sorted(costs)
    pct = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    # Unbounded queue: latency; bounded queue: drops.
    t_free, worst_latency = 0.0, 0.0
    for i, c in enumerate(costs):
        arrive = i * budget
        start = max(arrive, t_free)
        t_free = start + c
        worst_latency = max(worst_latency, t_free - arrive)
    waiting, t_free, dropped = [], 0.0, 0
    for i, c in enumerate(costs):
        arrive = i * budget
        waiting = [t for t in waiting if t > arrive]  # finish times of frames still queued or running
        if len(waiting) > queue:
            dropped += 1
            continue
        start = max(arrive, t_free)
        t_free = start + c
        waiting.append(t_free)
    return {"frame_ms_p50": pct(0.5), "frame_ms_p99": pct(0.99), "frame_ms_max": s[-1],
            "over_budget_pct": 100.0 * sum(c > budget for c in costs) / len(costs),
            "worst_latency_ms": worst_latency, "dropped": dropped, "budget_ms": budget}


def run_once(cli, seq, extra, workdir):
    dump = os.path.join(workdir, seq["name"])
    cmd = [cli, os.path.join(ROOT, seq["video"]), "-o", os.path.join(workdir, seq["name"] + ".ply"),
           "--dump", dump, "--dump-every", "1000000"] + extra
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    wall = time.time() - start
    r = parse_log(proc.stdout)
    r["wall_s"] = wall
    r["exit_code"] = proc.returncode
    frames_csv = os.path.join(dump, "frames.csv")
    if os.path.exists(frames_csv):
        r.update(realtime(frames_csv, r["video_fps"]))
    if "reference" in seq:
        ref = os.path.join(ROOT, seq["reference"])
        images = os.path.join(ref, "images.txt")
        cam = compare_trajectory.read_camera(os.path.join(ref, "cameras.txt"))
        r["reference_fov"], r["reference_k1"] = cam["fov"], cam["k1"]
        for kind, csv_name in (("final", "keyframes.csv"), ("live", "frames.csv")):
            path = os.path.join(dump, csv_name)
            if os.path.exists(path):
                r[kind] = compare_trajectory.evaluate(images, path, seq.get("step", 5))
    return r


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(runs):
    """Median over runs of every number; ranges for the path errors."""
    s = {}
    for key in ("frames", "lost_frames", "maps", "tracked_per_frame", "keyframes", "points", "surface_variation",
                "scattered_pct", "noise_median_pct", "noise_p90_pct", "two_view_pct", "fov", "k1", "background_passes", "fps", "core_fps", "video_fps",
                "wall_s", "reference_fov", "reference_k1", "frame_ms_p50", "frame_ms_p99", "frame_ms_max",
                "over_budget_pct", "worst_latency_ms", "dropped", "budget_ms"):
        s[key] = median([r.get(key) for r in runs])
    for kind in ("final", "live"):
        for key in ("ate_rms_pct", "ate_max_pct", "scale_min", "scale_max", "rpe_direction_deg", "rpe_length_ratio"):
            vals = [r.get(kind, {}).get(key) for r in runs]
            s[f"{kind}_{key}"] = median(vals)
            if key == "ate_rms_pct":
                vals = [v for v in vals if v is not None]
                if vals:
                    s[f"{kind}_ate_rms_pct_range"] = [min(vals), max(vals)]
    stages = {}
    for r in runs:
        for name, v in r["stages_ms"].items():
            stages.setdefault(name, []).append(v["avg"])
    s["stage_avg_ms"] = {name: statistics.median(v) for name, v in stages.items()}
    return s


def fmt(v, spec, none="-"):
    return none if v is None else format(v, spec)


def print_table(results):
    print()
    print(f"{'sequence':26s} {'final ATE':>12s} {'live ATE':>9s} {'scale':>11s} {'dir':>5s} {'fov/ref':>11s} "
          f"{'k1/ref':>13s} {'lost':>4s} {'pts':>7s} {'sv':>6s} {'fps':>6s}")
    for name, s in results.items():
        rng = s.get("final_ate_rms_pct_range")
        final = fmt(s.get("final_ate_rms_pct"), ".1f") + "%" if s.get("final_ate_rms_pct") is not None else "-"
        if rng and rng[1] - rng[0] >= 0.05:
            final += f" ({rng[0]:.1f}-{rng[1]:.1f})"
        live = fmt(s.get("live_ate_rms_pct"), ".1f") + ("%" if s.get("live_ate_rms_pct") is not None else "")
        scale = (f"{s['final_scale_min']:.2f}-{s['final_scale_max']:.2f}"
                 if s.get("final_scale_min") is not None else "-")
        fov = fmt(s.get("fov"), ".1f") + ("/" + fmt(s.get("reference_fov"), ".1f") if s.get("reference_fov") else "")
        k1 = fmt(s.get("k1"), ".3f") + ("/" + fmt(s.get("reference_k1"), ".3f") if s.get("reference_fov") else "")
        print(f"{name:26s} {final:>12s} {live:>9s} {scale:>11s} {fmt(s.get('final_rpe_direction_deg'), '.1f'):>5s} "
              f"{fov:>11s} {k1:>13s} {fmt(s.get('lost_frames'), 'd'):>4s} {fmt(s.get('points'), 'd'):>7s} "
              f"{fmt(s.get('surface_variation'), '.3f'):>6s} {fmt(s.get('fps'), '.0f'):>6s}")
    print("\nfinal/live ATE: RMS camera position error vs COLMAP in % of the path extent (final keyframes / pose when "
          "tracked); scale: min-max of 20 s window scale (1.00 = no drift); dir: motion direction error over 1 s (deg); "
          "sv: surface variation (noise); fps: overall throughput\n")
    print(f"{'real time':26s} {'frame ms p50/p99/max':>22s} {'budget':>7s} {'over':>6s} {'latency':>9s} {'dropped':>8s}")
    for name, s in results.items():
        if s.get("frame_ms_p50") is None:
            continue
        print(f"{name:26s} {s['frame_ms_p50']:6.1f}/{s['frame_ms_p99']:6.1f}/{s['frame_ms_max']:7.1f} "
              f"{s['budget_ms']:6.1f}ms {s['over_budget_pct']:5.1f}% {s['worst_latency_ms']:7.0f}ms {s['dropped']:8.0f}")
    print("(tracking thread per frame; over = frames slower than 1/fps; latency = worst wait with an unbounded "
          "queue; dropped = live camera keeping at most 3 waiting frames)\n")
    names = sorted({n for s in results.values() for n in s["stage_avg_ms"]})
    print("stage avg ms/frame: " + "  ".join(
        f"{n}: " + "/".join(fmt(s["stage_avg_ms"].get(n), ".1f") for s in results.values()) for n in names))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--cli", default=DEFAULT_CLI)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra CLI arguments (after --)")
    args = ap.parse_args()

    config = json.load(open(os.path.join(ROOT, "benchmarks", "benchmark.json")))
    only = set(filter(None, args.only.split(",")))
    sequences = [s for s in config["sequences"] if not only or s["name"] in only]
    missing = [s["video"] for s in sequences if not os.path.exists(os.path.join(ROOT, s["video"]))]
    if missing:
        print("missing videos (see benchmarks/README.md):\n  " + "\n  ".join(missing))
        sequences = [s for s in sequences if s["video"] not in missing]

    results, raw = {}, {}
    with tempfile.TemporaryDirectory() as work:
        for seq in sequences:
            runs = []
            for i in range(args.runs):
                print(f"{seq['name']}: run {i + 1}/{args.runs} ...", flush=True)
                runs.append(run_once(args.cli, seq, args.extra, work))
            raw[seq["name"]] = runs
            results[seq["name"]] = summarize(runs)
    print_table(results)

    if not args.no_save:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "core", "apps"], capture_output=True, text=True,
                               cwd=ROOT).stdout.strip()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        name = f"{stamp}_{commit}{'-dirty' if dirty else ''}{'_' + args.label if args.label else ''}.json"
        out = os.path.join(ROOT, "benchmarks", "results", name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"commit": commit, "dirty": bool(dirty), "label": args.label, "extra_args": args.extra,
                   "runs": args.runs, "summary": results, "runs_detail": raw}, open(out, "w"), indent=1)
        print(f"saved {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
