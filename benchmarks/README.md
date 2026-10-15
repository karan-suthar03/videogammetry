# Benchmarks

`python tools/benchmark.py` runs the CLI (`build/windows/apps/desktop/Release/videogammetry.exe`
by default, `--cli` to change) on the sequences in `benchmark.json`, and prints accuracy against
COLMAP, intrinsics, map quality and speed per sequence. Results go to `results/` (one JSON per
run, named by date and commit), so changes can be compared over time. Needs Python 3 with numpy.

```
python tools/benchmark.py                      # everything, once
python tools/benchmark.py --runs 3             # medians over 3 runs (background BA varies a little)
python tools/benchmark.py --only family,m60    # some sequences
python tools/benchmark.py --label nofuse -- --fuse 0   # extra CLI arguments after --
```

## Videos (not in git)

Make them with the FFmpeg in `third_party/ffmpeg/bin` (NVENC shown; `-c:v mpeg4 -q:v 2` works anywhere):

| file | source | command |
|---|---|---|
| `sample_video/street_walk_full_28s.mp4` | see `sample_video/SOURCES.md` | |
| `sample_video/variants/*.mp4` | from the street video | `backward`: `-vf reverse`; `rotated90`: `-vf transpose=1`; `fast4x`: `-vf setpts=0.25*PTS`; `mirrored_backward`: `-vf hflip,reverse` |
| `sample_video/family_1080p.mp4` | Tanks and Temples "Family" video (3840x2160) | `ffmpeg -i Family.mp4 -vf scale=1920:1080:flags=lanczos -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -an family_1080p.mp4` |
| `sample_video/m60_920p_120s.mp4` | Tanks and Temples "M60" video (4096x2160) | `ffmpeg -hwaccel cuda -hwaccel_output_format cuda -t 120 -i M60.mp4 -vf scale_cuda=1744:920 -c:v h264_nvenc -preset p1 -tune hq -rc vbr -cq 20 -b:v 0 -an m60_920p_120s.mp4` |

## References (`reference/<name>/`)

COLMAP 4.2.1 models (`cameras.txt`, `images.txt` with poses only) of every `step`-th frame
(image `n` = video frame `n * step`), single SIMPLE_RADIAL camera, SIFT 4096 features,
sequential matching (overlap 15, quadratic), `global_mapper` (`mapper` for family):

```
ffmpeg -i <video> -vf "select=not(mod(n\,<step>))" -fps_mode passthrough -q:v 2 -start_number 0 images/%05d.jpg
colmap feature_extractor --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --SiftExtraction.max_num_features 4096
colmap sequential_matcher --database_path db.db --SequentialMatching.overlap 15 --SequentialMatching.quadratic_overlap 1
colmap global_mapper --database_path db.db --image_path images --output_path sparse
colmap model_converter --input_path sparse/0 --output_path sparse/0_txt --output_type TXT
```
