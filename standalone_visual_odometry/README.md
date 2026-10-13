# Standalone Visual Odometry

This is the isolated OpenCV visual odometry application launched by the ANVESHAK web dashboard. It is not imported into the segmentation or A* pipeline.

## Direct use

```bash
python main.py --video /path/to/video.mp4
python main.py --video http://camera-address/stream
python main.py --video 0
python main.py --sequence 99
```

It can also be called from Python:

```python
from main import run

run("/path/to/video.mp4")
```

## Window controls

- `Space`: pause or resume
- `D` or Right Arrow: advance one frame while paused
- `R`: reset
- `+` or `-`: change playback rate
- `Q` or `Esc`: close

## Inputs

`--video` accepts local video files, webcam indices, RTSP sources, and HTTP MJPEG streams. For KITTI data, place sequences and poses below `data/kitti/dataset` or pass another directory through `--data_dir`.

Monocular output has arbitrary relative scale unless a valid scale or ground-truth motion source is supplied.
