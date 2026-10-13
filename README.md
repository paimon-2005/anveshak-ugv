# ANVESHAK UGV

Vision-based autonomous navigation for an Unmanned Ground Vehicle (UGV) in GPS-denied environments. It watches the world through a single camera, figures out where it can drive, plans a path, and steers an ESP32-controlled rover, all while you watch live on a web dashboard.

Built by **Team LuminaForge** (SIH 2026, PS 26126).

## What it does

- **Terrain segmentation**: BiSeNetV2 labels each pixel (traversable ground, sky, obstacles) in real time.
- **Path planning**: A* on the traversability mask, with configurable obstacle clearance.
- **Human detection**: YOLOv8n spots people and publishes `human_detected`, `human_count`, `human_max_confidence`.
- **GPS-denied localization**: visual odometry pose, local map, waypoint route, coverage and survivor tags.
- **Rover control**: sends commands (`forward`, `backward`, `left`, `right`, `stop`) to the ESP32 over HTTP, with emergency stop and `--dry-run` safety mode.
- **Web dashboard**: live camera / segmentation / A* views, local map (OpenStreetMap or satellite), video upload, e-stop.
- **Standalone visual odometry**: separate OpenCV app launched from the dashboard.

## Pipeline

```text
ESP32-CAM / video  ->  BiSeNetV2 segmentation  ->  traversable mask
                   ->  YOLOv8 human detection  ->  A* path planner  ->  motor commands -> ESP32 rover
                   ->  visual odometry          ->  local map / dashboard
```

## Project structure

```text
run.py                         # entry point
requirements.txt
semantic segmentation/
  combined_run.py              # main pipeline + CLI
  segmentation.py, bisenetv2.py, model_final.pth
  human_detection.py           # YOLO person detection
  navigation.py                # A* planner
  motor_control.py             # rover HTTP controller
  camera.py                    # stream/video input
  web_dashboard.py, dashboard/ # web UI
localization.py               # GPS-denied localization + local map
standalone_visual_odometry/    # separate OpenCV VO app
tools/                         # camera calibration, model validation
tests/                         # pytest suite
validation/                    # model validation results
data/kitti/                    # sample KITTI sequence for VO
```

## Install

Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

> Uses `opencv-python` (not `-headless`) because the standalone VO window needs OpenCV GUI support.

## Run

```bash
python run.py --dry-run
```

Open `http://localhost:8080`. `--dry-run` blocks real motor commands, so use it first.

Pick a camera source:

```bash
python run.py --source http://192.168.4.1:81/stream   # ESP32-CAM (default)
python run.py --source 0                              # webcam
python run.py --source example.mp4                    # video file
```

Drive a local-frame route (metres): `python run.py --route "10,0;10,6;2,6"`

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--source` | ESP32 stream | Camera, video file, or URL |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--width` / `--height` | 640 / 480 | Processing resolution |
| `--seg-every` | 4 | Run segmentation every N frames |
| `--seg-confidence` | 0.65 | Min confidence for traversable |
| `--floor-fraction` | 0.45 | Bottom frame fraction where sky is treated as ground (0 disables) |
| `--clearance` | 16 | Obstacle clearance for A* |
| `--human-confidence` | 0.50 | YOLO person threshold |
| `--human-every` | 1 | Run detection every N frames |
| `--fp16`, `--seg-tta`, `--threads N` | off / auto | Speed / accuracy tuning |
| `--host` / `--port` | `0.0.0.0` / 8080 | Dashboard address |
| `--no-dashboard` | off | Run headless |
| `--max-frames N`, `--metrics-json PATH` | off | Benchmarking |

See `python run.py --help` for the full list.

### Environment variables

| Variable | Purpose |
|---|---|
| `STREAM_URL` | Default camera source |
| `DASHBOARD_HOST`, `DASHBOARD_PORT` | Dashboard bind address |
| `ROVER_COMMAND_URL_TEMPLATE` | Rover command URL (default `http://192.168.4.1/action?go={command}&speed={speed}`) |
| `ROVER_<COMMAND>_VALUE` | Override command strings, e.g. `ROVER_FORWARD_VALUE` |

## Dashboard

- **All views / Only Segmentation / Only A\*** - switch what's displayed
- **Local Map** - draw an area on OSM/satellite, set heading offset, apply overlay. Cyan = VO trajectory, green dashed = LocalNav, magenta dashed = A* route, green marker = rover
- **Open Visual Odometry** - launches the standalone VO OpenCV window on the host machine
- **Upload test video** - runs the pipeline on a video (motors disabled)
- **Return to camera** - back to live source (motors stay disabled for safety)
- **Emergency stop**

API: `GET /api/status`, `GET /stream.mjpg?view=all|segmentation|astar`, `POST /api/upload`, `POST /api/source/live`, `POST /api/vo/launch`, `POST /api/emergency`.

## Standalone visual odometry

```bash
python standalone_visual_odometry/main.py --video example.mp4    # file / webcam / RTSP / MJPEG
python standalone_visual_odometry/main.py --sequence 99          # KITTI
```

Controls: `Space` pause, `D`/`→` step, `R` reset, `+`/`-` speed, `Q`/`Esc` quit. Monocular VO has arbitrary scale unless a scale source is provided.

## Tools

```bash
python tools/calibrate_camera.py <image_dir> --columns 9 --rows 6 --square-mm 25
python tools/validate_model.py <reference_checkpoint.pth>
python convert.py video.mp4 --output my_images --every 5     # video -> PNG frames
```

## Model

BiSeNetV2, 4 classes, auxiliary heads stripped for inference: 9.9M -> 3.6M parameters, checkpoint 39.9 MB -> 14.6 MB, identical class output on validation frames (100% agreement), about 1.6x faster on CPU (286 ms -> 175 ms median).

## Tests

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

Native OpenCV windows are mocked, so tests run without a display.

## Safety

- The dashboard has **no authentication**. Use only on a trusted local network.
- Always test with `--dry-run` before enabling motors.
- The floor override is a manual rule and can mark obstacles as traversable. Check the mask before driving.
- VO has no absolute lat/lon or compass heading; the map anchor and heading offset are set manually.
- Map tiles need browser internet access.

