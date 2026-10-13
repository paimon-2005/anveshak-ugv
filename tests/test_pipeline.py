import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest
import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
PIPELINE_DIRECTORY = PROJECT_DIRECTORY / "semantic segmentation"
VO_DIRECTORY = PROJECT_DIRECTORY / "standalone_visual_odometry"
sys.path.insert(0, str(PIPELINE_DIRECTORY))

from camera import CameraCalibration, FrameSource
from combined_run import create_views
from navigation import astar_search, plan_shortest_path
from segmentation import SegmentationEngine, load_segmentation_model
from web_dashboard import DashboardServer, RoverController


def test_camera_calibration_defaults():
    np.testing.assert_allclose(CameraCalibration(320, 240).K, [[277, 0, 160], [0, 277, 120], [0, 0, 1]])


def test_astar_chooses_shorter_detour():
    free = np.ones((15, 15), bool)
    free[3:12, 5:11] = False
    route = astar_search(free, (13, 7), (1, 7))
    cost = sum(np.linalg.norm(np.subtract(b, a)) for a, b in zip(route, route[1:]))
    assert cost == pytest.approx(14 + 2 * np.sqrt(2))
    assert min(cell[1] for cell in route) == 4
    assert all(free[cell] for cell in route)
    assert astar_search(np.array([[1, 0], [0, 1]], bool), (0, 0), (1, 1)) == []


def test_planner_preserves_obstacles():
    mask = np.full((240, 320), 255, np.uint8)
    mask[-20:, 140:180] = 0
    assert plan_shortest_path(mask)[0] == []
    mask[:] = 255
    mask[150, :] = 0
    path, _ = plan_shortest_path(mask, clearance=0)
    assert all(y > 150 for _, y in path)


class FixedClass(torch.nn.Module):
    def __init__(self, class_id, logit=10):
        super().__init__()
        self.class_id, self.logit = class_id, logit

    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], 4, *inputs.shape[-2:]), device=inputs.device)
        logits[:, self.class_id] = self.logit
        return logits


@pytest.mark.parametrize("height", [100, 480])
def test_sky_override_is_exactly_bottom_45_percent(height):
    colors, mask = SegmentationEngine(FixedClass(0), "cpu").predict(np.zeros((height, 128, 3), np.uint8))
    start = height - height * 45 // 100
    assert not mask[:start].any()
    assert np.all(mask[start:] == 255)
    assert np.all(colors[start:] == [0, 255, 0])
    assert np.all(colors[:start] == [255, 0, 0])


def test_floor_override_does_not_change_obstacles():
    _, mask = SegmentationEngine(FixedClass(2), "cpu").predict(np.zeros((100, 128, 3), np.uint8))
    assert not mask.any()


def wait_until(condition, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    assert condition()


def test_controller_watchdog_and_emergency(monkeypatch):
    sent = []
    monkeypatch.setattr(RoverController, "_send_command", lambda self, command, speed: sent.append((command, speed)))
    controller = RoverController(dry_run=True)
    try:
        controller.update_autonomous(
            [(320, 450), (320, 430), (320, 410)],
            frame_width=640,
            perception_at=time.monotonic(),
        )
        wait_until(lambda: ("forward", 55) in sent)
        wait_until(lambda: controller.status()["command"] == "stop")
        controller.set_emergency(True)
        controller.update_autonomous(
            [(320, 450), (320, 430), (320, 410)],
            frame_width=640,
            perception_at=time.monotonic(),
        )
        assert controller.status()["command"] == "stop"
    finally:
        controller.close()


class FakeProcess:
    def __init__(self):
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.running = False
        self.terminated = True

    def wait(self, timeout=None):
        self.running = False
        return 0

    def kill(self):
        self.running = False


def test_dashboard_launches_standalone_vo_for_selected_video(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_popen(command, cwd):
        process = FakeProcess()
        calls.append((command, cwd, process))
        return process

    monkeypatch.setattr("web_dashboard.subprocess.Popen", fake_popen)
    server = DashboardServer("127.0.0.1", 0, dry_run=True, initial_source=video, vo_app_directory=VO_DIRECTORY)
    server.start()
    try:
        result = server.launch_visual_odometry()
        assert result["launched"]
        assert calls[0][0] == [sys.executable, str(VO_DIRECTORY / "main.py"), "--video", str(video.resolve())]
        assert calls[0][1] == str(VO_DIRECTORY)
        assert server.launch_visual_odometry()["already_running"]
    finally:
        server.stop()
    assert calls[0][2].terminated


def test_live_source_uses_dashboard_raw_stream(monkeypatch):
    calls = []
    monkeypatch.setattr("web_dashboard.subprocess.Popen", lambda command, cwd: calls.append((command, cwd)) or FakeProcess())
    server = DashboardServer("127.0.0.1", 0, dry_run=True, initial_source="http://camera/stream", vo_app_directory=VO_DIRECTORY)
    server.start()
    try:
        server.launch_visual_odometry()
        assert calls[0][0][-1] == f"http://127.0.0.1:{server.port}/stream.mjpg?view=raw"
    finally:
        server.stop()


def test_dashboard_api_upload_and_views(tmp_path):
    source_video = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(source_video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((64, 96, 3), 110, np.uint8))
    writer.release()
    server = DashboardServer("127.0.0.1", 0, dry_run=True, initial_source=source_video, vo_app_directory=VO_DIRECTORY)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        frame = np.zeros((64, 96, 3), np.uint8)
        server.state.publish_views({"all": frame, "raw": frame, "segmentation": frame, "astar": frame}, {})
        with urlopen(base + "/api/status", timeout=2) as response:
            assert json.load(response)["dry_run"]
        request = Request(base + "/api/upload", data=source_video.read_bytes(), headers={"X-Filename": "../../test.avi", "Content-Type": "application/octet-stream"})
        with urlopen(request, timeout=3) as response:
            assert json.load(response)["uploaded"]
        selected = server.take_source()[0]
        assert Path(selected).parent == Path(server.upload_directory.name)
        with urlopen(Request(base + "/api/source/live", data=b"{}", headers={"Content-Type": "application/json"}), timeout=2):
            pass
        assert server.take_source() == (None,)
    finally:
        server.stop()


def test_fragmented_mjpeg_reader():
    image = np.full((64, 96, 3), 100, np.uint8)
    payload = cv2.imencode(".jpg", image)[1].tobytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for _ in range(3):
                    data = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                    for start in range(0, len(data), 127):
                        self.wfile.write(data[start:start + 127])
                        self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    source = FrameSource(f"http://127.0.0.1:{http.server_port}/stream", read_timeout=0.4)
    source.start()
    try:
        packet = source.read(timeout=1)
        assert packet is not None and packet.frame.shape == image.shape
    finally:
        source.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=1)


def test_display_views_have_no_embedded_vo():
    frame = np.zeros((480, 640, 3), np.uint8)
    mask = np.full((480, 640), 255, np.uint8)
    path, status = plan_shortest_path(mask)
    result = {"frame": frame, "color": frame.copy(), "traversable": mask, "path": path, "status": status}
    views = create_views(frame, result, 10, status, 100)
    assert set(views) == {"all", "raw", "segmentation", "astar"}
    assert views["all"].shape == (480, 1920, 3)
    assert views["astar"].shape == frame.shape


def test_fused_model_preserves_predictions():
    torch.set_num_threads(2)
    reference = load_segmentation_model(PIPELINE_DIRECTORY / "bisenetv2.py", PIPELINE_DIRECTORY / "model_final.pth", "cpu", fuse=False)
    optimized = load_segmentation_model(PIPELINE_DIRECTORY / "bisenetv2.py", PIPELINE_DIRECTORY / "model_final.pth", "cpu", fuse=True)
    torch.manual_seed(5)
    inputs = torch.randn(1, 3, 128, 192).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        expected, actual = reference(inputs), optimized(inputs)
    torch.testing.assert_close(expected, actual, atol=0.0002, rtol=0.0002)
    assert (expected.argmax(1) == actual.argmax(1)).float().mean() > 0.999


def test_dashboard_frontend_structure():
    html = (PIPELINE_DIRECTORY / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "openVO" in html
    assert "openVOSecondary" in html
    assert "page-vo" not in html
    assert "page-perception" not in html
    assert "page-logs" not in html
    assert "page-map" in html
    assert "page-dashboard" in html

    js = (PIPELINE_DIRECTORY / "dashboard" / "app.js").read_text(encoding="utf-8")
    assert "/api/vo/launch" in js
    assert "openVisualOdometry" in js
    assert "renderMap" in js


def test_standalone_vo_main_default():
    sys.path.insert(0, str(VO_DIRECTORY))
    from main import run
    ret = run(argv=["--no_gui"])
    assert ret == 0
