# ANVESHAK UGV

Optimized BiSeNetV2 terrain segmentation, visual odometry, local A* planning, ESP32 camera input, and web motor controls. All Python code is comment-free.

## Dashboard update

The website now has **All views**, **Only VO**, **Only Segmentation**, and **Only A*** buttons. These change the displayed video; the underlying perception pipeline continues running. All views uses a four-panel layout. A* has its own full-resolution mask with the route, start marker, goal marker, planner outcome, and separate control-readiness status. A blocked start or disconnected terrain displays a no-path explanation; it does not invent a route through obstacles.

**Upload test video** accepts MP4, AVI, MOV, MKV, WebM, or M4V files up to 512 MB, subject to the installed OpenCV codec support. The file is validated before being selected. Uploading switches the process to the test video, resets VO, and disables physical motor requests. At the end, the website stays available. **Reset VO** restarts that test video, and **Return to camera** restores the source configured by `--source`. Motor requests remain disabled after a website source switch; restart the program when returning to physical driving. Uploaded files are temporary and are deleted when the server closes.

VO starts at the first processed frame after the page's session-start request, not when Python starts. Opening/reloading the page or pressing **Reset VO** clears the shared rover trajectory. A browser heartbeat keeps the session active; after six seconds without heartbeats, tracking pauses. This is one shared rover session: opening a second tab also resets it. In `--no-dashboard` mode, tracking starts immediately. A configured video waits for the website to open before playback.

Stationary rejection now uses a 1.25-pixel median-flow threshold, with a retained reference frame to accumulate genuine slow motion. Initial translation additionally requires three direction-consistent estimates with broad feature coverage and at least 60% pose support. After relative landmark scale is lost, translation freezes rather than adding a fabricated unit step; use **Reset VO** to reinitialize. These conservative gates can defer small or ambiguous motions. They do not guarantee drift-free monocular VO on every camera: camera calibration, blur, moving objects, scene depth and scale observability still matter.

The 90-frame regression sequence with JPEG recompression, brightness variation, sensor noise and subpixel jitter produced exactly zero displacement. A separate sustained-translation test verified that the motion gate still allows real projected motion. These are synthetic validation results, not a measurement of your camera's drift.

## Start

Use Python 3.10 or newer. From this extracted project directory:

```bash
python -m pip install -r requirements.txt
python run.py
```

Open `http://localhost:8080`. The default stream is `http://192.168.4.1:81/stream`. The existing entry point still works:

```bash
python "semantic segmentation/combined_run.py"
```

Settings for a low-end CPU laptop:

```bash
python run.py --device cpu --threads 2 --max-features 800 --seg-every 4
```

Processing resolution remains 640 x 480 by default. A smaller segmentation input can reduce latency, with a possible accuracy trade-off:

```bash
python run.py --device cpu --threads 2 --seg-scale 0.75
```

For a different camera address or a USB webcam:

```bash
python run.py --source http://192.168.4.1:81/stream
python run.py --source 0
```

HTTP sources use the MJPEG reader. RTSP sources use OpenCV/FFmpeg. Webcam support depends on the local camera backend. Live input keeps the newest decoded frame rather than an unbounded frame queue. Segmentation runs in a separate worker, while VO processes incoming frames. Prerecorded video is processed synchronously without dropping source frames.

## Your requested floor override

The bottom 45% of the image retains sky-to-traversable relabeling. Only class 0 pixels in that region become class 1. This updates both the colored mask and the traversable mask and bypasses the confidence threshold for those overridden sky pixels. Other classes are not overwritten.

At 480 pixels high, the override covers zero-based rows 264 through 479. This is the bottom 216 rows, not the region beginning 45% down the frame. The original file started its override at 45% of image height, which covered the bottom 55%; this version follows your explicit bottom-45% request.

The override is a manually imposed floor assumption, not an improvement to the learned classifier. It can incorrectly mark real obstacles as free when the model calls them sky. Inspect the mask before driving. Disable it when that assumption does not hold:

```bash
python run.py --floor-fraction 0
```

`--floor-fraction 0.55` reproduces the original cutoff at 45% of frame height. Outside the override, class 1 must exceed the default confidence of 0.65 to become traversable. `--seg-confidence` adjusts that threshold.

## VO integration and corrections

`FrameVisualOdometry` lives in `semantic segmentation/combined_run.py`. It integrates the supplied VO archive's spatial feature bucketing, FAST detection, pyramidal Lucas–Kanade tracking, forward/backward consistency filtering, robust essential-matrix estimation, and pose accumulation.

Additional changes include full fx/fy intrinsics, distortion correction, a Shi–Tomasi fallback, explicit zero-distance handling, bounded histories, stationary/low-motion handling, rotation-only estimation, depth/reprojection/parallax checks, and triangulated landmarks. PnP with RANSAC and nonlinear refinement tracks these landmarks across subsequent frames, allowing variable inter-frame translation instead of assigning every moving frame a unit step.

The point order is consistent throughout essential-matrix estimation and pose recovery. For the previous-to-current camera transform `(R, t)`, camera-to-world integration is:

```text
C_new = C_old + R_world_old @ (-R.T @ t)
R_world_new = R_world_old @ R.T
```

There is no forced forward sign and no manual sign flip of the output coordinates. Coordinates use the first camera as the world reference: X right, Y down, Z forward. The large map auto-scales the latest 5,000 accepted poses and shows the current heading. Path length remains cumulative.

Without an external scale measurement, coordinates are relative units, not metres. After landmark scale is lost, uncertain translation is rejected until the VO is reset; no automatic unit-step scale restart is applied. Drift and ambiguous motion are still possible on smooth tiles, blurred video, repetitive textures, or moving objects. This is local monocular VO, not a loop-closing SLAM system.

## Camera calibration

The default intrinsics are approximate, not a measured OV3660 calibration. Capture at least 12 varied checkerboard views using the same lens, crop, focus, orientation and streaming resolution used during operation. Include tilted boards and positions near the image edges.

```bash
python tools/calibrate_camera.py calibration_images --columns 9 --rows 6 --square-mm 25 --output camera_calibration.json
python run.py --calibration camera_calibration.json
```

Columns and rows count inner corners, not squares. Inspect the reported per-view reprojection errors and retake blurry or poorly detected views. The JSON contains `camera_matrix`, `distortion_coefficients`, and `image_size` in width-height order. Intrinsics are scaled with the processing resolution, and invalid rectification borders are excluded from planning. Fisheye calibration models are not supported by this helper.

For a prerecorded video with measured cumulative travel distance:

```bash
python run.py --source recording.mp4 --calibration camera_calibration.json --scale-csv distance.csv
```

The CSV must have `timestamp_s,distance_m` columns, strictly increasing video-relative timestamps, and nondecreasing cumulative distances. It must cover the entire processed interval. Interpolation gives each frame interval's measured travel distance. For live wheel-encoder or other sensor integration, pass nonnegative inter-frame metres to `FrameVisualOdometry.update(frame, scale=distance, timestamp=timestamp)` and initialize it with `metric_scale=True`. Calibration alone does not recover monocular metric scale.

## Segmentation model optimization

The checkpoint has been pruned by removing only auxiliary training heads. Its retained inference tensors are unchanged. The loader strictly checks the deployment state dictionary, disables training gradients, fuses Conv–BatchNorm pairs in evaluation mode, and uses channels-last memory layout. Model parameters drop from approximately 9.93 million to 3.59 million after fusion. The checkpoint drops from 39.9 MB to 14.6 MB.

Normalization and four-class ordering are preserved from your original code. Input dimensions not divisible by 32 are padded and cropped correctly. `--seg-tta` optionally averages normal/flipped logits at additional compute cost. `--fp16` is an optional CUDA-only mode; full precision remains the default. Neither option has a demonstrated accuracy improvement on your camera footage.

The pruned weights are inference weights. To load them directly, use `BiSeNetV2(4, output_aux=False)`; pass `return_logits=True` when logits are needed. For training with auxiliary heads, use the original full checkpoint or explicitly initialize the missing heads. No training data or labeled validation masks were supplied, so there is no claimed mIoU increase and no active-channel pruning or retraining.

YOLO code, weights, flags and dependencies have been removed completely. There is no object-detection stage.

## Path planning and motor controls

A* uses orthogonal cost 1 and diagonal cost sqrt(2), with an admissible octile heuristic and no diagonal corner cutting. Every pixel in a planning cell must be traversable. Obstacles are expanded before downsampling so thin obstacles are not erased. A blocked bottom-centre start returns no path rather than teleporting the start to a distant free cell.

This is shortest-path planning on the current image grid, not shortest travel distance in metres. The automatic goal is a reachable forward image location; no world-space destination or metrically calibrated ground map is provided. Pixel clearance is not a measured rover footprint.

The dashboard starts in manual mode. Releasing manual input stops the rover, manual commands expire after 650 ms, and autonomous commands require a perception result less than 800 ms old. Stale camera input stops movement. Speed changes propagate even when the direction stays unchanged. An emergency stop locks movement until released. The host watchdog cannot stop a rover after complete Wi-Fi or laptop failure; the ESP32 firmware must independently stop its motors if command heartbeats expire.

Default motor request:

```text
http://192.168.4.1/action?go={command}&speed={speed}
```

Use `ROVER_COMMAND_URL_TEMPLATE` to change the endpoint and `ROVER_FORWARD_VALUE`, `ROVER_BACKWARD_VALUE`, `ROVER_LEFT_VALUE`, `ROVER_RIGHT_VALUE`, `ROVER_STOP_VALUE` to map firmware command names. Keep the dashboard on a trusted local network; it is not an authenticated public control service.

For testing with motor requests disabled:

```bash
python run.py --dry-run
python run.py --source example.mp4 --max-frames 300
```

Video-file mode always disables physical motor requests. `--no-dashboard` runs without the HTTP server; `--host`, `--port`, `DASHBOARD_HOST`, and `DASHBOARD_PORT` configure it. `--show-local-window` or `SHOW_LOCAL_WINDOW=1` enables an OpenCV window. The default dependency is headless OpenCV; to use native windows, install the corresponding non-headless `opencv-python` package instead, not alongside it.

## Validation and utilities

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
python run.py --source example.mp4 --no-dashboard --max-frames 60 --metrics-json metrics.json --trajectory-csv trajectory.csv
python tools/validate_model.py /path/to/original/model_final.pth --output model_validation.json
```

`validation/` contains measured test-run results, including the model comparison and a 60-frame video smoke test. These are development-environment measurements, not your laptop's performance or a labeled accuracy benchmark. CUDA, an actual ESP32 camera, physical motor firmware, and calibration quality require local verification.

`convert.py` extracts numbered video frames and refuses to mix them into a nonempty output sequence. `test.py --img_path images --pose_path poses.txt --calibration camera_calibration.json` evaluates an image sequence. That benchmark explicitly uses ground-truth step distances when poses are provided and reports ATE in first-camera coordinates; it must not be presented as scale-free metric VO.

The bundled example video is preserved. Repository history, generated caches, the obsolete trajectory image, and detector files are omitted from the distribution; originals remain recoverable from your original upload. Runtime model downloads are not required.

## Technical references

- [OpenCV camera calibration and pose recovery](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [PyTorch evaluation-time Conv–BatchNorm fusion](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fusion.fuse_conv_bn_eval.html)
