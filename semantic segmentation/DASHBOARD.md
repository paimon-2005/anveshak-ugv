# Dashboard

The web dashboard serves camera, segmentation, A* planning, and autonomous rover control and emergency stop from `web_dashboard.py`.

## Start

```bash
python run.py --dry-run
```

Then open `http://localhost:8080`.

## Visual odometry button

The dashboard does not calculate or display visual odometry. `POST /api/vo/launch` starts `standalone_visual_odometry/main.py` in a separate Python process. The separate program creates its own OpenCV window on the computer running the dashboard.

Local and uploaded videos are passed directly to that program. Live camera sources are passed through the dashboard's raw MJPEG mirror at `/stream.mjpg?view=raw`, preventing a second direct ESP32 camera connection.

Only one standalone process runs at a time. The dashboard stops it during shutdown so temporary uploaded videos are not removed while still in use.

## Endpoints

- `GET /api/status`
- `GET /stream.mjpg?view=all`
- `GET /stream.mjpg?view=segmentation`
- `GET /stream.mjpg?view=astar`
- `POST /api/upload`
- `POST /api/source/live`
- `POST /api/vo/launch`
- `POST /api/emergency`

The `raw` MJPEG view is reserved for the standalone process and is not shown as a web-dashboard button.
