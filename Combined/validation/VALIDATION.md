# Validation results

All 25 automated tests passed in the dashboard-update run. Python compilation and JavaScript syntax checks passed.

Coverage includes forward, reverse and lateral motion with noisy/outlier correspondences; PnP variable translation; rotation-only motion; stationary and blank frames; explicit zero distance; camera-to-world integration; scaled intrinsics; measured-scale coverage; shortest detours and corner blocking; thin obstacles and blocked starts; bottom-45% sky relabeling; preservation of other classes; confidence gating; motor speed updates and watchdogs; dashboard HTTP responses; and fragmented MJPEG reconnection.

The earlier base-version 60-frame bundled-video smoke test ran the segmentation, VO, planner, dashboard rendering and shutdown paths on CPU with physical motor requests disabled. It completed at approximately 13.3 processing FPS in this environment, including dashboard overhead. This is not a benchmark of the user's laptop or a test of physical ESP32 hardware.

The optimized segmentation model was compared with the original full checkpoint on three 640 x 480 video frames (indices 0, 120 and 600). All compared class predictions matched. Maximum absolute logit difference was approximately 0.0000072. The comparison is before the manually requested sky-to-floor override.

With two CPU threads, five timed forwards after warm-up gave median times of 285.8 ms for the original model and 175.4 ms for optimized inference, approximately 1.63x faster in this environment. The original model has 9,927,792 parameters; the fused deployment model has 3,593,016. Removing unused auxiliary heads reduced the checkpoint from 39,936,486 to 14,576,229 bytes. All retained checkpoint tensors are unchanged.

These checks establish implementation behavior and prediction preservation, not improved mIoU or real-world trajectory accuracy. No labeled terrain dataset, measured camera calibration, or live rover test was supplied. The bottom-45% floor rule intentionally overrides sky predictions and is not learned model accuracy. Metric scale still requires measured distance input.

Environment: Python 3.12, PyTorch 2.14.0+cpu, OpenCV 5.0.0. CUDA and native OpenCV windows were not exercised. See the JSON and CSV files here for measured outputs and the README for reproduction commands.

## Dashboard and stationary-drift update

The final regression run passed 25 tests. New coverage checks noisy stationary camera frames, rejection of unit-step recovery after scale loss, sustained motion acceptance, browser-session start/reset, invalid-upload rejection, valid video upload with safe temporary filenames, source switching, and a full-size A* route view.

The 90-frame stationary-noise test reported zero translation and zero accumulated distance. This includes JPEG recompression, brightness changes, Gaussian noise, and subpixel image jitter. It is a synthetic test and does not establish a drift bound for the user's physical camera.

An additional full-process HTTP integration run confirmed that no video frames or VO trajectory were processed before opening the browser session, that tracking started after session opening, that all four MJPEG views returned images, that upload reset the pose and disabled motors, that EOF kept the dashboard online, that reset replayed the test video, and that the configured source could be restored. See `dashboard_flow.json`.

The model weights and inference optimization are unchanged. Earlier model-equivalence results therefore remain relevant; the older combined-pipeline FPS measurement is historical and is not a performance claim for the new multi-view dashboard. Actual ESP32 camera motion and motor firmware remain untested here.
