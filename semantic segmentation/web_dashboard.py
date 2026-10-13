import json
import mimetypes
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from motor_control import RoverController


DASHBOARD_DIRECTORY = Path(__file__).with_name("dashboard")
DEFAULT_VO_APP_DIRECTORY = Path(__file__).resolve().parents[1] / "standalone_visual_odometry"


class DashboardState:
    def __init__(self, controller):
        self.controller = controller
        self.condition = threading.Condition()
        self.jpeg_frame = None
        self.frame_sequence = 0
        self.view_frames = {}
        self.vo_frame = None
        self.vo_frame_sequence = 0
        self.vo_metrics = {
            "vo_tracking": False,
            "vo_fps": 0.0,
            "vo_features": 0,
            "vo_inliers": 0,
            "vo_x": 0.0,
            "vo_y": 0.0,
            "vo_z": 0.0,
            "vo_travel": 0.0,
            "vo_frames": 0,
            "vo_trail": [],
            "vo_error": "",
            "vo_last_update": 0.0,
        }
        self.telemetry = {
            "camera_connected": False,
            "planner_status": "WAITING",
            "fps": 0.0,
            "path_points": 0,
            "frame_number": 0,
            "source_kind": "camera",
            "perception_age_ms": None,
            "inference_ms": None,
        }
        # Geographic anchor for the browser local-map view. VO is metric but has
        # no absolute world reference, so the operator selects an OSM area and
        # optionally supplies the local-frame heading relative to east.
        self.map_selection = None

    def publish(self, frame, telemetry):
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.condition:
            self.telemetry.update(telemetry)
            if success:
                self.jpeg_frame = encoded.tobytes()
                self.frame_sequence += 1
            self.condition.notify_all()

    def update_telemetry(self, telemetry):
        with self.condition:
            self.telemetry.update(telemetry)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            result = dict(self.telemetry)
        result.update(self.controller.status())
        result.update(self.vo_metrics)
        result["map_selection"] = self.map_selection
        result["timestamp"] = time.time()
        return result

    def publish_views(self, views, telemetry):
        encoded = {}
        for name, frame in views.items():
            ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                encoded[name] = data.tobytes()
        with self.condition:
            self.view_frames = encoded
            self.jpeg_frame = encoded.get("all")
            self.telemetry.update(telemetry)
            self.frame_sequence += 1
            self.condition.notify_all()

    def publish_vo(self, frame, metrics):
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with self.condition:
            if ok:
                self.vo_frame = data.tobytes()
                self.vo_frame_sequence += 1
            self.vo_metrics.update(metrics)
            self.condition.notify_all()

    def reset_vo(self):
        with self.condition:
            self.vo_frame = None
            self.vo_metrics.update({
                "vo_tracking": False,
                "vo_fps": 0.0,
                "vo_features": 0,
                "vo_inliers": 0,
                "vo_x": 0.0,
                "vo_y": 0.0,
                "vo_z": 0.0,
                "vo_travel": 0.0,
                "vo_frames": 0,
                "vo_trail": [],
                "vo_error": "",
                "vo_last_update": 0.0,
            })
            self.condition.notify_all()

    def wait_for_raw_frame(self, after_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_sequence > after_sequence or not self.controller.running,
                timeout=2.0
            )
            return self.frame_sequence, self.view_frames.get("raw")

    def wait_for_vo_frame(self, after_sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: self.vo_frame_sequence > after_sequence or not self.controller.running,
                timeout=2.0
            )
            return self.vo_frame_sequence, self.vo_frame

    def wait_for_frame(self, after_sequence):
        with self.condition:
            self.condition.wait_for(lambda: self.frame_sequence > after_sequence or not self.controller.running, timeout=2.0)
            return self.frame_sequence, self.jpeg_frame


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "AnveshakDashboard/1.0"

    def log_message(self, format_string, *args):
        return

    @property
    def dashboard(self):
        return self.server.dashboard

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length < 0 or content_length > 4096:
            raise ValueError("Request is too large")
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def _serve_static(self, request_path):
        relative_path = "index.html" if request_path == "/" else request_path.lstrip("/")
        target = (DASHBOARD_DIRECTORY / relative_path).resolve()
        if DASHBOARD_DIRECTORY.resolve() not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream_video(self):
        view = parse_qs(urlparse(self.path).query).get("view", ["all"])[0]
        if view not in {"all", "raw", "segmentation", "astar", "vo", "map"}:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        sequence = 0
        try:
            while self.dashboard.controller.running:
                if view == "vo":
                    sequence, jpeg_frame = self.dashboard.state.wait_for_vo_frame(sequence)
                else:
                    sequence, jpeg_frame = self.dashboard.state.wait_for_frame(sequence)
                    with self.dashboard.state.condition:
                        jpeg_frame = self.dashboard.state.view_frames.get(view, jpeg_frame)
                if jpeg_frame is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg_frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg_frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/status":
            self._send_json(self.dashboard.state.snapshot())
        elif request_path == "/stream.mjpg":
            self._stream_video()
        else:
            self._serve_static(request_path)

    def do_POST(self):
        request_path = urlparse(self.path).path
        try:
            if request_path == "/api/upload":
                self._upload_video()
                return
            payload = self._read_json()
            if request_path == "/api/source/live":
                self.dashboard.queue_source(None)
            elif request_path == "/api/vo/launch":
                self._send_json(self.dashboard.launch_visual_odometry())
                return
            elif request_path == "/api/map/select":
                selection = self._validate_map_selection(payload)
                self.dashboard.state.map_selection = selection
            elif request_path == "/api/emergency":
                self.dashboard.controller.set_emergency(payload.get("active", True))
            else:
                self._send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(self.dashboard.state.snapshot())
        except (TypeError, ValueError, json.JSONDecodeError, OSError, RuntimeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    @staticmethod
    def _validate_map_selection(payload):
        bbox = payload.get("bbox")
        if not isinstance(bbox, dict):
            raise ValueError("bbox is required")
        try:
            south = float(bbox["south"])
            west = float(bbox["west"])
            north = float(bbox["north"])
            east = float(bbox["east"])
            heading_deg = float(payload.get("heading_deg", 0.0))
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid map selection")
        if not all(np.isfinite(v) for v in (south, west, north, east, heading_deg)):
            raise ValueError("Map selection contains non-finite values")
        if south >= north or west >= east:
            raise ValueError("Map selection bounds are invalid")
        if not (-85.0 <= south < north <= 85.0 and -180.0 <= west < east <= 180.0):
            raise ValueError("Map selection is outside supported geographic bounds")
        if (north - south) > 1.0 or (east - west) > 1.0:
            raise ValueError("Select a local area no larger than 1 degree")
        return {
            "south": south, "west": west, "north": north, "east": east,
            "heading_deg": ((heading_deg + 180.0) % 360.0) - 180.0,
        }

    def _upload_video(self):
        length = int(self.headers.get("Content-Length", "0"))
        suffix = Path(self.headers.get("X-Filename", "test.mp4")).suffix.lower()
        if not 0 < length <= 512 * 1024 * 1024 or suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            raise ValueError("Upload a supported video of at most 512 MB")
        target = Path(self.dashboard.upload_directory.name) / (uuid.uuid4().hex + suffix)
        self.connection.settimeout(30)
        try:
            with target.open("wb") as output:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1048576, remaining))
                    if not chunk:
                        raise ValueError("Upload interrupted")
                    output.write(chunk)
                    remaining -= len(chunk)
            capture = cv2.VideoCapture(str(target))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                raise ValueError("The uploaded file cannot be decoded as video")
            self.dashboard.queue_source(str(target))
            self._send_json({"uploaded": True, "message": "Test video queued; motors disabled"})
        except Exception:
            target.unlink(missing_ok=True)
            raise


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DashboardServer:
    def __init__(self, host="0.0.0.0", port=8080, dry_run=False, initial_source="0", vo_app_directory=None, camera_matrix=None):
        self.source_lock = threading.Lock()
        self.pending_source = None
        self.original_source = self._normalize_source(initial_source)
        self.current_source = self.original_source
        self.vo_lock = threading.Lock()
        self.vo_process = None
        self.vo_app_directory = Path(vo_app_directory or DEFAULT_VO_APP_DIRECTORY).resolve()
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64) if camera_matrix is not None else None
        self.vo_stop = threading.Event()
        self.vo_thread = None
        self.upload_directory = tempfile.TemporaryDirectory(prefix="anveshak-video-")
        self.http_server = DashboardHTTPServer((host, int(port)), DashboardRequestHandler)
        try:
            self.controller = RoverController(dry_run=dry_run)
        except Exception:
            self.http_server.server_close()
            self.upload_directory.cleanup()
            raise
        self.state = DashboardState(self.controller)
        self.http_server.dashboard = self
        self.thread = threading.Thread(target=self.http_server.serve_forever, name="dashboard-http-server", daemon=True)
        self.host = host
        self.port = self.http_server.server_address[1]
        self.vo_thread = threading.Thread(target=self._vo_worker, name="browser-visual-odometry", daemon=True)

    @staticmethod
    def _normalize_source(source):
        value = str(source)
        if value.isdecimal() or value.startswith(("http://", "https://", "rtsp://")):
            return value
        return str(Path(value).expanduser().resolve())

    def start(self):
        self.thread.start()
        self.vo_thread.start()

    def queue_source(self, path):
        self.controller.stop_motion()
        self.state.reset_vo() if hasattr(self, "state") else None
        with self.controller.lock:
            self.controller.dry_run = True
        with self.source_lock:
            self.pending_source = (path,)
            self.current_source = self.original_source if path is None else self._normalize_source(path)

    def set_active_source(self, path):
        with self.source_lock:
            self.current_source = self._normalize_source(path)

    def take_source(self):
        with self.source_lock:
            pending, self.pending_source = self.pending_source, None
            return pending

    @staticmethod
    def _render_vo_path(trajectory, width=960, height=540):
        """Render only the live estimated trajectory; no feature dots/flow lines."""
        canvas = np.full((height, width, 3), (13, 19, 37), dtype=np.uint8)
        margin = 55
        x0, x1 = margin, width - margin
        y0, y1 = margin, height - margin

        # Subtle engineering grid.
        for x in np.linspace(x0, x1, 9).astype(int):
            cv2.line(canvas, (x, y0), (x, y1), (31, 42, 66), 1)
        for y in np.linspace(y0, y1, 7).astype(int):
            cv2.line(canvas, (x0, y), (x1, y), (31, 42, 66), 1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (55, 68, 94), 1)

        points = np.asarray(trajectory, dtype=np.float64) if trajectory else np.empty((0, 3))
        if len(points) == 0:
            cv2.putText(canvas, "WAITING FOR VO TRAJECTORY", (margin + 18, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (145, 157, 180), 1, cv2.LINE_AA)
            return canvas

        # Use local X/Z motion from the actual VO trajectory. Keep the origin visible.
        x = points[:, 0]
        z = -points[:, 2]  # forward motion is -Z in the VO frame; draw it upward
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_z, max_z = float(np.min(z)), float(np.max(z))
        span_x = max(max_x - min_x, 1e-3)
        span_z = max(max_z - min_z, 1e-3)
        pad_x = max(span_x * 0.18, 0.25)
        pad_z = max(span_z * 0.18, 0.25)
        lo_x, hi_x = min_x - pad_x, max_x + pad_x
        lo_z, hi_z = min_z - pad_z, max_z + pad_z

        def pixel(px, pz):
            u = int(x0 + (px - lo_x) / max(hi_x - lo_x, 1e-9) * (x1 - x0))
            v = int(y1 - (pz - lo_z) / max(hi_z - lo_z, 1e-9) * (y1 - y0))
            return max(x0, min(x1, u)), max(y0, min(y1, v))

        pts = np.array([pixel(px, pz) for px, pz in zip(x, z)], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, (70, 220, 120), 3, cv2.LINE_AA)

        current = tuple(pts[-1, 0])
        cv2.rectangle(canvas, (current[0] - 5, current[1] - 5), (current[0] + 5, current[1] + 5), (255, 255, 255), -1)
        cv2.rectangle(canvas, (current[0] - 5, current[1] - 5), (current[0] + 5, current[1] + 5), (70, 220, 120), 2)

        cv2.putText(canvas, "VISUAL ODOMETRY · LIVE PATH", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 239, 246), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(points)} POSES", (18, 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 220, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, "START", (x0 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 162, 184), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ROVER", (current[0] + 9, current[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 220, 150), 1, cv2.LINE_AA)
        return canvas

    def _vo_worker(self):
        """Run the actual standalone VO continuously and publish diagnostics + trajectory."""
        try:
            vo_root = str(self.vo_app_directory)
            if vo_root not in sys.path:
                sys.path.insert(0, vo_root)
            from config import default_config as vo_config
            from src.visual_odometry import VisualOdometry
        except Exception as exc:
            self.state.update_telemetry({
                "vo_error": f"import failed: {exc}",
                "vo_tracking": False,
            })
            return

        vo = None
        prev_source_kind = None
        sequence = 0
        processed = 0
        started = time.monotonic()

        while not self.vo_stop.is_set() and self.controller.running:
            try:
                sequence, jpeg = self.state.wait_for_raw_frame(sequence)
                if jpeg is None:
                    continue

                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                with self.state.condition:
                    source_kind = self.state.telemetry.get("source_kind", "camera")

                if vo is None or source_kind != prev_source_kind:
                    K = self.camera_matrix.copy() if self.camera_matrix is not None else None
                    h, w = frame.shape[:2]
                    if K is None or K.shape != (3, 3):
                        # Approximate intrinsics for an uncalibrated live camera.
                        f = 0.9 * max(w, h)
                        K = np.array(
                            [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                            dtype=np.float64,
                        )
                    vo_config.detector.detector_type = "FAST"
                    vo_config.detector.mask_sky = True
                    vo_config.detector.mask_vehicle_hood = True
                    vo_config.gui.show_feature_points = False
                    vo_config.gui.show_flow_lines = False
                    vo = VisualOdometry(intrinsic_matrix=K, config=vo_config)
                    prev_source_kind = source_kind
                    processed = 0
                    started = time.monotonic()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = vo.step(gray, scale=vo_config.pose.default_scale)
                processed += 1

                elapsed = max(time.monotonic() - started, 1e-6)
                fps = processed / elapsed
                pos = np.asarray(result.get("position", np.zeros(3)), dtype=float).reshape(-1)
                trajectory = vo.trajectory

                travel = 0.0
                if len(trajectory) > 1:
                    arr = np.asarray(trajectory, dtype=float)
                    travel = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())

                path_frame = self._render_vo_path(
                    trajectory,
                    width=max(frame.shape[1], 640),
                    height=max(frame.shape[0], 360),
                )

                vo_trail = []
                for point in trajectory[-1500:]:
                    arrp = np.asarray(point, dtype=float).reshape(-1)
                    if len(arrp) >= 3 and np.all(np.isfinite(arrp[:3])):
                        vo_trail.append([
                            float(arrp[0]), float(arrp[1]), float(arrp[2])
                        ])

                self.state.publish_vo(path_frame, {
                    "vo_tracking": bool(
                        result.get("success", False)
                        and int(result.get("num_inliers", 0)) >= 8
                    ),
                    "vo_fps": fps,
                    "vo_features": int(result.get("num_features", 0)),
                    "vo_inliers": int(result.get("num_inliers", 0)),
                    "vo_x": float(pos[0]) if len(pos) > 0 else 0.0,
                    "vo_y": float(pos[1]) if len(pos) > 1 else 0.0,
                    "vo_z": float(pos[2]) if len(pos) > 2 else 0.0,
                    "vo_travel": travel,
                    "vo_frames": processed,
                    "vo_trail": vo_trail,
                    "vo_error": "",
                    "vo_last_update": time.time(),
                })
            except Exception as exc:
                # A single bad frame must not permanently kill the VO thread.
                self.state.update_telemetry({
                    "vo_error": f"{type(exc).__name__}: {exc}",
                    "vo_tracking": False,
                    "vo_last_update": time.time(),
                })
                vo = None
                prev_source_kind = None
                time.sleep(0.05)

        self.state.reset_vo()

    def _vo_input_source(self):
        with self.source_lock:
            source = self.current_source
        if Path(source).is_file():
            return source
        return f"http://127.0.0.1:{self.port}/stream.mjpg?view=raw"

    def launch_visual_odometry(self):
        entrypoint = self.vo_app_directory / "main.py"
        if not entrypoint.is_file():
            raise RuntimeError(f"Standalone visual odometry entry point is missing: {entrypoint}")
        with self.vo_lock:
            if self.vo_process is not None and self.vo_process.poll() is None:
                return {"launched": False, "already_running": True, "message": "Visual odometry window is already running"}
            source = self._vo_input_source()
            self.vo_process = subprocess.Popen(
                [sys.executable, str(entrypoint), "--video", source],
                cwd=str(self.vo_app_directory),
            )
        return {"launched": True, "already_running": False, "message": "Visual odometry opened in a separate OpenCV window"}

    def _stop_visual_odometry(self):
        with self.vo_lock:
            process, self.vo_process = self.vo_process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def publish(self, frame, telemetry):
        self.state.publish(frame, telemetry)

    def camera_disconnected(self):
        self.controller.stop_motion()
        self.state.update_telemetry({"camera_connected": False, "planner_status": "CAMERA OFFLINE"})

    def stop(self):
        self.vo_stop.set()
        if self.vo_thread is not None and self.vo_thread.is_alive():
            self.vo_thread.join(timeout=1.5)
        self._stop_visual_odometry()
        self.controller.close()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.thread.join(timeout=1.0)
        self.upload_directory.cleanup()
