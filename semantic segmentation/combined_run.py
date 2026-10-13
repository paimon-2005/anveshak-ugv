import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import CameraCalibration, FrameSource
from navigation import draw_planned_path, plan_shortest_path
from segmentation import SegmentationEngine, load_segmentation_model
from human_detection import HumanDetector
from web_dashboard import DashboardServer

# GPS-denied local navigation (optional; disabled by --no-localization)
try:
    _LOC_ROOT = str(Path(__file__).resolve().parents[1])
    if _LOC_ROOT not in sys.path:
        sys.path.insert(0, _LOC_ROOT)
    from localization import LocalNav
except Exception as _loc_import_err:
    LocalNav = None
    print(f"[localization] import failed – feature disabled: {_loc_import_err}")


BASE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = BASE_DIRECTORY.parent
MODEL_FILE = BASE_DIRECTORY / "bisenetv2.py"
WEIGHTS_FILE = BASE_DIRECTORY / "model_final.pth"
VO_APP_DIRECTORY = PROJECT_DIRECTORY / "standalone_visual_odometry"
STREAM_URL = "http://192.168.4.1:81/stream"
HUMAN_MODEL_FILE = BASE_DIRECTORY / "yolov8n.pt"


class PerceptionWorker:
    def __init__(self, engine, valid_mask, asynchronous=True, clearance=16):
        self.engine, self.valid_mask = engine, valid_mask
        self.asynchronous, self.clearance = asynchronous, clearance
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.running = True
        self.thread = None
        if asynchronous:
            self.thread = threading.Thread(target=self._run, name="perception-worker", daemon=True)
            self.thread.start()

    def _process(self, frame, packet):
        started = time.perf_counter()
        color, traversable = self.engine.predict(frame)
        traversable[self.valid_mask == 0] = 0
        path, status = plan_shortest_path(traversable, clearance=self.clearance)
        return {
            "frame": frame,
            "color": color,
            "traversable": traversable,
            "path": path,
            "status": status,
            "captured_at": packet.captured_at,
            "sequence": packet.sequence,
            "epoch": packet.epoch,
            "inference_ms": (time.perf_counter() - started) * 1000,
        }

    def submit(self, frame, packet):
        if not self.asynchronous:
            self.result = self._process(frame, packet)
        else:
            with self.condition:
                self.pending = frame, packet
                self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or not self.running)
                if not self.running:
                    return
                frame, packet = self.pending
                self.pending = None
            try:
                result = self._process(frame, packet)
                with self.condition:
                    self.result = result
            except Exception as error:
                with self.condition:
                    self.error, self.running = error, False
                return

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Perception worker failed") from self.error
            return self.result

    def close(self):
        with self.condition:
            self.running, self.pending = False, None
            self.condition.notify()
        if self.thread:
            self.thread.join(timeout=10)


def create_views(frame, result, fps, status, age_ms, human_detections=None):
    h, w = frame.shape[:2]
    human_detections = human_detections or []
    raw = frame.copy()
    segmentation = frame.copy() if result is None else cv2.addWeighted(result["frame"], 0.55, result["color"], 0.45, 0)
    astar = np.zeros_like(frame) if result is None else cv2.cvtColor(result["traversable"], cv2.COLOR_GRAY2BGR)
    start = (w // 2, h - 5)
    target = (w // 2, int(h * 0.15))
    cv2.circle(astar, start, 8, (0, 220, 255), -1)
    cv2.drawMarker(astar, target, (0, 80, 255), cv2.MARKER_CROSS, 20, 2)
    if result is not None:
        draw_planned_path(astar, result["path"])
    planner_status = result["status"] if result is not None else "WAITING"
    labels = [
        (raw, f"CAMERA | {fps:.1f} FPS"),
        (segmentation, "SEGMENTATION"),
        (astar, f"A*: {planner_status} | age {age_ms:.0f} ms"),
    ]
    for panel, label in labels:
        cv2.rectangle(panel, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    if result is not None and not result["path"]:
        cv2.putText(
            astar,
            "No connected free route from the rover start",
            (12, max(70, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 160, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(astar, f"Control: {status}", (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, cv2.LINE_AA)
    raw = HumanDetector.annotate(raw, human_detections)
    segmentation = HumanDetector.annotate(segmentation, human_detections)
    astar = HumanDetector.annotate(astar, human_detections)
    combined = np.hstack((raw, segmentation, astar))
    return {"all": combined, "raw": frame, "segmentation": segmentation, "astar": astar}


def _parse_route(route_str):
    """Parse route string '10,0;10,6;2,6' into list of (x, y) float tuples."""
    if not route_str or not route_str.strip():
        return []
    waypoints = []
    for segment in route_str.strip().split(";"):
        parts = segment.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Waypoint must be 'x,y', got: {segment!r}")
        waypoints.append((float(parts[0]), float(parts[1])))
    return waypoints


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ANVESHAK terrain segmentation, A* navigation and dashboard")
    parser.add_argument("--source", default=os.environ.get("STREAM_URL", STREAM_URL))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_FILE)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seg-every", type=int, default=4)
    parser.add_argument("--seg-scale", type=float, default=1.0)
    parser.add_argument("--seg-confidence", type=float, default=0.65)
    parser.add_argument("--floor-fraction", type=float, default=0.45, help="Bottom fraction where sky becomes traversable; 0 disables the override")
    parser.add_argument("--seg-tta", action="store_true")
    parser.add_argument("--human-model", type=Path, default=HUMAN_MODEL_FILE)
    parser.add_argument("--human-confidence", type=float, default=0.50)
    parser.add_argument("--human-imgsz", type=int, default=640)
    parser.add_argument("--human-every", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--clearance", type=int, default=16)
    parser.add_argument("--perception-timeout", type=float, default=2.5, help="Maximum age of the latest segmentation result before control pauses")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--show-local-window", action="store_true", default=os.environ.get("SHOW_LOCAL_WINDOW", "0") == "1")
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8080")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--metrics-json", type=Path)
    # ---------- localization flags ----------
    parser.add_argument("--route", default="", help="Semicolon-separated local waypoints in metres, e.g. '10,0;10,6;2,6'")
    parser.add_argument("--speed-mps", type=float, default=0.25, help="Commanded rover speed in m/s (used for dead-reckoning distance)")
    parser.add_argument("--no-localization", action="store_true", help="Disable GPS-denied localization entirely")
    args = parser.parse_args(argv)
    if (
        args.seg_every < 1
        or args.human_every < 1
        or not 0 < args.human_confidence < 1
        or args.human_imgsz < 32
        or args.threads < 1
        or args.max_frames < 0
        or args.clearance < 0
        or not np.isfinite(args.perception_timeout)
        or args.perception_timeout <= 0
    ):
        parser.error("Invalid frame interval, thread count, frame limit, clearance, or perception timeout")
    if args.speed_mps <= 0:
        parser.error("--speed-mps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    calibration = CameraCalibration(args.width, args.height, args.calibration)
    source = FrameSource(args.source)
    model = load_segmentation_model(MODEL_FILE, args.weights, device)
    engine = SegmentationEngine(model, device, args.seg_confidence, args.seg_scale, args.seg_tta, args.fp16, args.floor_fraction)
    human_model = str(args.human_model) if args.human_model.exists() else "yolov8n.pt"
    human_detector = HumanDetector(human_model, args.human_confidence, args.human_imgsz, device)
    dashboard = None
    worker = None
    frame_number, sequence, epoch = 0, -1, None
    human_detections = []
    intervals = deque(maxlen=60)
    last_frame_at, last_render_at = time.monotonic(), 0.0
    playback_started = time.monotonic()
    started = time.perf_counter()
    print(f"Device: {device} | Camera: {'calibrated' if calibration.calibrated else 'approximate intrinsics'}")
    print(f"Sky-to-traversable override: bottom {args.floor_fraction:.0%} of the frame")

    # ---------- GPS-denied localization ----------
    nav = None
    _nav_last_t = time.time()
    if not args.no_localization and LocalNav is not None:
        try:
            route = _parse_route(args.route)
            nav = LocalNav(
                K=calibration.K,
                route=route,
                speed_mps=args.speed_mps,
            )
            # Give the controller a reference so it can blend mission steering.
            # We pass it after the controller is created (done in DashboardServer).
            print(f"Localization enabled | route waypoints: {len(route)} | speed: {args.speed_mps} m/s")
        except Exception as _nav_err:
            print(f"[localization] init failed – disabled: {_nav_err}")
            nav = None
    try:
        if not args.no_dashboard:
            dashboard = DashboardServer(
                args.host,
                args.port,
                dry_run=args.dry_run or not source.live,
                initial_source=args.source,
                vo_app_directory=VO_APP_DIRECTORY,
                camera_matrix=calibration.K,
            )
            dashboard.start()
            print(f"Dashboard: http://localhost:{dashboard.port}")
        # Give controller a nav reference for mission-steering blending (safe if nav is None).
        if nav is not None and dashboard is not None:
            dashboard.controller.nav = nav
        elif dashboard is not None:
            dashboard.controller.nav = None
        worker = PerceptionWorker(engine, calibration.valid_mask, True, args.clearance)
        source.start()
        if args.show_local_window:
            cv2.namedWindow("ANVESHAK", cv2.WINDOW_NORMAL)
        print("Running. Press Ctrl+C to stop.")
        while not args.max_frames or frame_number < args.max_frames:
            if dashboard:
                pending = dashboard.take_source()
                if pending is not None:
                    source.close()
                    worker.close()
                    selected = args.source if pending[0] is None else pending[0]
                    source = FrameSource(selected)
                    source.start()
                    worker = PerceptionWorker(engine, calibration.valid_mask, True, args.clearance)
                    sequence, epoch = -1, None
                    frame_number = 0
                    human_detections = []
                    intervals.clear()
                    last_frame_at = time.monotonic()
                    playback_started = last_frame_at
                    dashboard.set_active_source(selected)
                    with dashboard.controller.lock:
                        dashboard.controller.dry_run = True
            packet = source.read(sequence)
            now = time.monotonic()
            if packet is None:
                worker.latest()
                if dashboard and (not source.connected or now - last_frame_at > 0.5):
                    dashboard.camera_disconnected()
                if source.ended:
                    if dashboard and not args.max_frames:
                        dashboard.state.update_telemetry({"planner_status": "VIDEO FINISHED", "source_status": "Upload another video or return to camera"})
                        time.sleep(0.03)
                        continue
                    break
                if args.show_local_window and cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                    break
                continue
            if not source.live:
                wait = playback_started + packet.timestamp - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, 0.1))
            sequence = packet.sequence
            if epoch != packet.epoch:
                epoch = packet.epoch
                if dashboard:
                    dashboard.controller.stop_motion()
            frame = calibration.prepare(packet.frame)
            if frame_number % args.human_every == 0:
                human_detections = human_detector.detect(frame)
            if frame_number % args.seg_every == 0:
                worker.submit(frame, packet)
            result = worker.latest()
            if result is not None and result["epoch"] != epoch:
                result = None
            now = time.monotonic()
            age = now - result["captured_at"] if result is not None else float("inf")
            latest_capture = source.last_captured_at
            fresh_camera = not source.live or source.connected and latest_capture is not None and now - latest_capture <= 1.5
            usable = result is not None and age <= args.perception_timeout and fresh_camera
            path = result["path"] if usable else []
            status = result["status"] if usable else "WAITING" if result is None else "STALE PERCEPTION"
            if dashboard:
                if not fresh_camera:
                    dashboard.camera_disconnected()
                else:
                    dashboard.controller.update_autonomous(path, args.width, result["captured_at"] if usable else None)
            intervals.append(max(now - last_frame_at, 1e-6))
            last_frame_at = now
            fps = len(intervals) / sum(intervals)
            frame_number += 1

            # ---------- GPS-denied localization (non-fatal) ----------
            nav_telemetry = {}
            nav_map_frame = None
            if nav is not None:
                try:
                    now_t = time.time()
                    dt = max(now_t - _nav_last_t, 1e-6)
                    _nav_last_t = now_t
                    cmd = dashboard.controller.command if dashboard else "stop"
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    nav.process(gray, cmd, dt)
                    # Pass survivor detections to localizer.
                    if human_detections:
                        boxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3], d["confidence"])
                                 for d in human_detections]
                        nav.add_survivors(boxes, args.width)
                    nav_telemetry = nav.status()
                    nav_map_frame = nav.render()
                    # Mission complete: stop the rover via the existing safe stop path.
                    if nav.done and dashboard is not None:
                        dashboard.controller.stop_motion()
                except Exception as _nav_loop_err:
                    print(f"[localization] frame error (ignored): {_nav_loop_err}")

            telemetry = {
                "camera_connected": bool(fresh_camera),
                "planner_status": status,
                "fps": fps,
                "source_kind": "camera" if source.live else "test video",
                "path_points": len(path),
                "frame_number": frame_number,
                "floor_fraction": args.floor_fraction,
                "perception_age_ms": age * 1000 if math.isfinite(age) else None,
                "inference_ms": result["inference_ms"] if result else None,
                "calibrated": calibration.calibrated,
                "human_detected": bool(human_detections),
                "human_count": len(human_detections),
                "human_max_confidence": max((d["confidence"] for d in human_detections), default=0.0),
            }
            # Merge localization keys into telemetry (safe: nav_telemetry may be empty).
            telemetry.update(nav_telemetry)

            if dashboard:
                dashboard.state.update_telemetry(telemetry)
            if (dashboard or args.show_local_window) and now - last_render_at >= 0.1:
                views = create_views(frame, result, fps, status, age * 1000 if math.isfinite(age) else 0, human_detections)
                # Inject local map view so the dashboard can stream it.
                if nav_map_frame is not None:
                    views["map"] = nav_map_frame
                if dashboard:
                    dashboard.state.publish_views(views, telemetry)
                if args.show_local_window:
                    cv2.imshow("ANVESHAK", views["all"])
                last_render_at = now
            if args.show_local_window and cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        source.close()
        if worker:
            worker.close()
        if dashboard:
            dashboard.stop()
        if args.show_local_window:
            cv2.destroyAllWindows()
        # Save mission report regardless of exit reason.
        if nav is not None:
            try:
                nav.save_report("mission_report.json")
                print("mission_report.json written.")
            except Exception as _rep_err:
                print(f"[localization] failed to write mission_report.json: {_rep_err}")
    elapsed = time.perf_counter() - started
    metrics = {
        "frames": frame_number,
        "elapsed_s": elapsed,
        "processing_fps": frame_number / max(elapsed, 1e-9),
        "calibrated": calibration.calibrated,
        "floor_fraction": args.floor_fraction,
        "device": str(device),
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Processed {frame_number} frames | {metrics['processing_fps']:.1f} FPS")
    return metrics


if __name__ == "__main__":
    main()
