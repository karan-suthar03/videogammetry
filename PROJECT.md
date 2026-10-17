# Videogammetry: Project Definition

> Status: **Draft**. Decisions recorded below; open questions at the bottom.

---

## 1. One-line goal
Take a video as input and produce a **3D point cloud** of the filmed scene. The pipeline ends at the point cloud: no meshing, texturing, splats or NeRF.

## 2. Guiding principles
- **Progressive:** fastest/simplest working version first (sparse), then advance (denser, faster, real-time).
- **Mostly our own code.** Use existing libraries only for low-level building blocks; COLMAP and ORB-SLAM3 serve as **reference code and benchmarks**, not as the pipeline.
- **Very cross-platform:** Windows, Linux, macOS. **The CPU path is complete and the reference; the GPU is optional** (Vulkan compute, decided 2026-10-17: runs on NVIDIA/AMD/Intel and Android, no CUDA). Every GPU stage falls back to the CPU and is checked against it.
- **Offline first, real-time later:** build and test offline on recorded video, then turn the same core into a real-time system.

## 3. Input
- **Source:** recorded video file (offline phases); live camera later (real-time phase)
- **Formats:** <!-- mp4, mov, ... -->
- **Typical length / resolution:**
- **Subject:** <!-- small objects, rooms, outdoor scenes -->
- **Camera intrinsics:** <!-- known (calibrated) or must be estimated/guessed from video? -->

## 4. Output
- Point cloud (`.ply`, colored) + camera trajectory/poses
- **Real-world scale required?** <!-- yes/no; monocular video has no scale by default -->

## 5. Roadmap

### Phase 0: Setup and benchmark baseline
- Project skeleton, build system, test videos
- Run **COLMAP (CPU-only, sparse)** on the test videos → reference poses + sparse cloud
- Evaluation tooling: compare our poses/points against the reference (trajectory error, point count, runtime)

### Phase 1: Offline sparse reconstruction (first goal)
1. Video decode + frame selection (skip blurry/redundant frames)
2. Feature detection + description (ORB to start, since it's fastest on CPU)
3. Feature matching between nearby frames (sequential, video order)
4. Two-view initialization (essential matrix, triangulation)
5. Incremental pose estimation (PnP) + triangulation of new points
6. Bundle adjustment (local, then global)
7. Export sparse `.ply` + poses
8. Benchmark vs. COLMAP

### Phase 2: Real-time sparse (SLAM-style)
- Restructure Phase 1 into tracking / local mapping threads (ORB-SLAM3 as reference)
- Keyframe selection, motion model, local BA
- Loop closure / relocalization (later)
- Live camera input

### Phase 3: Denser point cloud (CPU)
- Semi-dense first (points on edges/high-gradient pixels), then
- Dense depth per keyframe (plane sweep / stereo matching), fused into one cloud

## 6. Project structure
One platform-independent C++ core, plus thin per-platform apps that include it.

```
core/                  C++20 static library (namespace vg): the whole pipeline
  include/videogammetry/   public API: FrameView, Reconstructor, PointCloud, writePly
  src/
apps/desktop/          CLI for Windows + Linux (+ macOS), same source; live viewer (Win32 window, CPU renderer)
apps/android/          Gradle project; JNI bridge (app/src/main/cpp) links core
tools/                 offline dev tools (train_orb_pattern)
tests/                 unit tests (CTest)
CMakeLists.txt         root: builds core + desktop app
CMakePresets.json      presets: windows (MSVC), linux (Ninja)
```

- The core never decodes video or touches OS APIs; each app decodes frames its own way and passes `vg::FrameView`s in.
- Desktop decodes video with FFmpeg (`apps/desktop/src/video_reader.*`, LGPL shared build, dynamically linked).
- Build Windows: `powershell -ExecutionPolicy Bypass -File scripts/fetch_ffmpeg_windows.ps1` (once, downloads FFmpeg 8.1 into `third_party/`), then `cmake --preset windows && cmake --build --preset windows-release`
- Run: `build/windows/apps/desktop/Release/videogammetry.exe sample_video/street_walk_10s.mp4 -o cloud.ply [--save-frame <n> frame.ppm]`
- Live viewer (Windows): `build/windows/apps/desktop/Release/videogammetry_viewer.exe [video] [--fov <deg>] [--size <px>] [--real-time]`. Without a video it opens on a start menu: open a video, recent videos, and every reconstruction/view setting (any value can be typed, e.g. `1e6` or `max`; bounds only where a value stops making sense), saved to `%APPDATA%/videogammetry/viewer.ini`; Esc returns to it from the view. It plays the video through the reconstruction and draws the point cloud, keyframe and current-camera frustums and the camera path as they are built, plus the video with tracked keypoints. Runs as fast as possible by default (M or --real-time paces it to the video's frame rate, Space pauses); S saves the final `.ply`. Rendering is software-only (`apps/desktop/src/viewer/renderer.*`, portable); only `main_win32.cpp` is Windows-specific
- Self-contained Windows build (one exe per app, no DLLs, runs on a clean machine): `powershell -ExecutionPolicy Bypass -File scripts/build_ffmpeg_static_windows.ps1` (once; compiles FFmpeg statically with vcpkg into `third_party/ffmpeg-static`, ~15-30 min), then `cmake --preset windows-static && cmake --build --preset windows-static-release`; exes in `build/windows-static/apps/desktop/Release/`. `VG_STATIC=ON` also links the MSVC runtime statically (/MT). FFmpeg is LGPL: shipping a statically linked exe means offering what is needed to relink it
- Build Linux: `cmake --preset linux && cmake --build --preset linux-release` (needs `g++`/`clang++`, `cmake`, `ninja`, `pkg-config`, FFmpeg dev packages; untested)
- Build Android: open `apps/android` in Android Studio (needs NDK + CMake from SDK Manager)
- Tests: `ctest --test-dir build/windows -C Release`
- Run + visual output: `videogammetry.exe sample_video/street_walk_10s.mp4 -o test_output/cloud.ply --trajectory test_output/trajectory.ply --dump test_output` writes the colored point cloud (`.ply`), keyframe camera positions, `test_output/tracking/*.bmp` (keypoints tracked to map points green) and `test_output/map.bmp` (top + side view); `test_output/` is git-ignored
- Motion variants for testing (git-ignored, generated with ffmpeg): `sample_video/variants/{backward,rotated90,fast4x,mirrored_backward}.mp4`
- ORB descriptor pattern: learned offline by `tools/train_orb_pattern` into `core/src/orb_pattern.inc` (committed); retrain when more varied footage is available

## 7. Interface
- CLI first (`videogammetry input.mp4 -o cloud.ply`)
- Viewer: built-in live viewer on Windows (`videogammetry_viewer`); external tools (CloudCompare / MeshLab) for the saved `.ply`

## 8. Tech constraints
- **OS:** cross-platform (developed on Windows 11)
- **GPU:** optional Vulkan compute backend (`gpu/`, `VG_GPU=ON` by default). Vulkan is loaded at run time (volk), so the apps still start without a driver; headers, volk and the glslang shader compiler are fetched by CMake (`cmake/Vulkan.cmake`, pinned, ~17 MB once), no Vulkan SDK needed. Done: feature extraction (`--gpu` in the CLI, a menu setting in the viewer; identical keypoints, ~1 descriptor bit in 3 million differs; `tools/gpu_check` verifies). Mapping thread done (local/focal BA solved off the tracking thread, merged before the next keyframe; `--no-mapping-thread` for the old deterministic path). Next: local BA and the triangulation/fusion searches on the GPU
- **Language:** C++20 core; Java (JNI) shim on Android
- **References:** `references/colmap`, `references/ORB_SLAM3`
- **Allowed libraries (candidates):** <!-- e.g. OpenCV (video I/O, features, geometry), Eigen (math), Ceres (bundle adjustment) -->

## 9. Scope
**Must have (v1 = Phase 1):**
- Offline: video in → sparse colored point cloud + poses out, CPU only, cross-platform
- Benchmark against COLMAP

**Later:**
- Real-time (Phase 2), denser clouds (Phase 3)

**Out of scope:**
- Meshing, texturing, Gaussian splatting, NeRF
- CUDA (the GPU backend is Vulkan, so it also runs on AMD/Intel and Android)

## 10. Success criteria
- **Phase 1:** a 30s phone video gives a recognizable sparse cloud; poses close to COLMAP's; runtime well under COLMAP's
- **Phase 2:** ≥ 20–30 fps tracking on a laptop CPU at 640×480–720p
- **Phase 3:** <!-- TBD -->

## 11. Open questions
- Which libraries count as "existing parts" we're allowed to use (OpenCV? Eigen? Ceres?) vs. what we write ourselves (feature detector? BA solver?)
- Camera calibration: do we calibrate the phone once, or estimate intrinsics from the video? Finding (2026-10-14): on the street clip (forward walking, little rotation) two-view geometry fits equally well for any FOV from 40 to 100 deg, so focal length is **not** observable from this footage; default is 65 deg (`--fov`). Needs calibration, metadata, or bundle adjustment over rotating motion.
- Test videos: what scenes, what device?
