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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "semantic segmentation"))

from camera import CameraCalibration, FrameSource, ScaleSamples
from combined_run import FrameVisualOdometry
from navigation import astar_search, plan_shortest_path
from segmentation import SegmentationEngine, load_segmentation_model
from web_dashboard import DashboardServer, RoverController


def scene():
    rng = np.random.default_rng(41)
    points = rng.uniform([-3, -1.4, 5], [3, 1.4, 14], (400, 3))
    k = np.array([[500., 0, 320], [0, 530, 240], [0, 0, 1.]])
    return points, k


@pytest.mark.parametrize("centre", [[0.05, 0.01, 0.3], [0.02, -0.01, -0.3], [0.3, 0.0, 0.02]])
def test_essential_forward_reverse_and_lateral_motion(centre):
    points, k = scene()
    rng = np.random.default_rng(3)
    rotation = cv2.Rodrigues(np.array([0.002, 0.035, -0.003]))[0]
    centre = np.asarray(centre)
    translation = -(rotation @ centre).reshape(3, 1)
    current_xyz = points @ rotation.T + translation.T
    previous = points[:, :2] / points[:, 2:3]
    current = current_xyz[:, :2] / current_xyz[:, 2:3]
    current += rng.normal(0, 0.08 / 515, current.shape)
    current[:40] = rng.uniform(-0.25, 0.25, (40, 2))
    vo = FrameVisualOdometry(intrinsic_matrix=k)
    pose = vo._essential(previous, current, np.full_like(points, np.nan), float(np.linalg.norm(centre)))
    assert pose is not None
    r, t, inliers, _ = pose
    assert np.linalg.norm(-(r.T @ t).ravel() - centre) < 0.025
    assert np.linalg.norm(cv2.Rodrigues(r @ rotation.T)[0]) < 0.01
    assert inliers[:40].sum() < 8


def test_pnp_variable_step():
    points, k = scene()
    vo = FrameVisualOdometry(intrinsic_matrix=k)
    for centre in ([0, 0, 0.04], [0.1, 0, 0.4], [-0.12, 0.02, -0.2]):
        rotation = cv2.Rodrigues(np.array([0.01, -0.04, 0.005]))[0]
        translation = -(rotation @ np.asarray(centre)).reshape(3, 1)
        xyz = points @ rotation.T + translation.T
        pose = vo._pnp(xyz[:, :2] / xyz[:, 2:3], points)
        assert pose is not None
        np.testing.assert_allclose(-(pose[0].T @ pose[1]).ravel(), centre, atol=1e-5)


def test_rotation_only():
    points, k = scene()
    rotation = cv2.Rodrigues(np.array([0.01, 0.08, -0.02]))[0]
    current = points @ rotation.T
    result = FrameVisualOdometry(intrinsic_matrix=k)._rotation_only(
        points[:, :2] / points[:, 2:3], current[:, :2] / current[:, 2:3])
    assert result is not None
    np.testing.assert_allclose(result[0], rotation, atol=1e-8)
    np.testing.assert_array_equal(result[1], 0)


def test_stationary_and_blank_frames():
    frame = np.random.default_rng(12).integers(0, 256, (480, 640), dtype=np.uint8)
    vo = FrameVisualOdometry(max_features=500)
    for index in range(20):
        vo.update(frame, timestamp=index / 30)
    np.testing.assert_array_equal(vo.get_coordinates(), 0)
    vo.update(np.zeros_like(frame), timestamp=0.7)
    np.testing.assert_array_equal(vo.get_coordinates(), 0)


def test_zero_scale_and_pose_accumulation(monkeypatch):
    points, k = scene()
    pixels = (k @ points.T).T
    pixels = (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)
    vo = FrameVisualOdometry(intrinsic_matrix=k, metric_scale=True)
    image = np.zeros((480, 640), np.uint8)
    monkeypatch.setattr(vo, "detect_features", lambda *args: pixels.reshape(-1, 1, 2))
    vo.update(image, scale=0, timestamp=0)
    monkeypatch.setattr(vo, "track_features", lambda *args: (pixels, pixels + 3, np.arange(len(pixels))))
    rotation = cv2.Rodrigues(np.array([0., 0.1, 0.]))[0]
    t = np.array([[0.], [0.], [-1.]])
    monkeypatch.setattr(vo, "_pnp", lambda *args: (rotation, t.copy(), np.ones(len(pixels), bool), "TRACKING PNP"))
    vo.update(image, scale=0, timestamp=0.03)
    np.testing.assert_array_equal(vo.get_coordinates(), 0)
    np.testing.assert_allclose(vo.rotation, rotation.T, atol=1e-8)
    vo.previous_points = pixels.reshape(-1, 1, 2)
    vo.landmarks = np.full_like(points, np.nan)
    vo.update(image, scale=0.2, timestamp=0.06)
    np.testing.assert_allclose(vo.get_coordinates(), (rotation.T @ (-rotation.T @ (t * 0.2))).ravel(), atol=1e-8)


def test_calibration_and_scale_coverage(tmp_path):
    np.testing.assert_allclose(CameraCalibration(320, 240).K, [[277, 0, 160], [0, 277, 120], [0, 0, 1]])
    path = tmp_path / "scale.csv"
    path.write_text("timestamp_s,distance_m\n0,0\n1,0.2\n2,0.6\n")
    samples = ScaleSamples(path)
    assert samples.step(0) == 0
    assert samples.step(0.5) == pytest.approx(0.1)
    assert samples.step(2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        samples.step(3)


def test_astar_chooses_shorter_detour():
    free = np.ones((15, 15), bool)
    free[3:12, 5:11] = False
    route = astar_search(free, (13, 7), (1, 7))
    cost = sum(np.linalg.norm(np.subtract(b, a)) for a, b in zip(route, route[1:]))
    assert cost == pytest.approx(14 + 2 * np.sqrt(2))
    assert min(cell[1] for cell in route) == 4
    assert all(free[cell] for cell in route)
    blocked = np.array([[1, 0], [0, 1]], bool)
    assert astar_search(blocked, (0, 0), (1, 1)) == []
    assert astar_search(blocked, (0, 1), (0, 1)) == []


def test_planner_does_not_teleport_or_erase_thin_obstacles():
    mask = np.full((240, 320), 255, np.uint8)
    mask[-20:, 140:180] = 0
    assert plan_shortest_path(mask)[0] == []
    mask[:] = 255
    mask[150, :] = 0
    path, _ = plan_shortest_path(mask, clearance=0)
    assert all(y > 150 for _, y in path)
    mask[:] = 255
    path, status = plan_shortest_path(mask)
    assert status == "PATH FOUND" and len(path) > 10


class FixedClass(torch.nn.Module):
    def __init__(self, class_id, logit=10):
        super().__init__()
        self.class_id, self.logit = class_id, logit

    def forward(self, x):
        logits = torch.zeros((x.shape[0], 4, *x.shape[-2:]), device=x.device)
        logits[:, self.class_id] = self.logit
        return logits


@pytest.mark.parametrize("height", [100, 480])
def test_sky_override_is_exactly_bottom_45_percent(height):
    engine = SegmentationEngine(FixedClass(0), "cpu")
    colors, mask = engine.predict(np.zeros((height, 128, 3), np.uint8))
    start = height - height * 45 // 100
    assert not mask[:start].any()
    assert np.all(mask[start:] == 255)
    assert np.all(colors[start:] == [0, 255, 0])
    assert np.all(colors[:start] == [255, 0, 0])


@pytest.mark.parametrize("class_id", [2, 3])
def test_floor_override_does_not_change_other_classes(class_id):
    _, mask = SegmentationEngine(FixedClass(class_id), "cpu").predict(np.zeros((100, 128, 3), np.uint8))
    assert not mask.any()


def test_floor_override_can_be_disabled_and_terrain_confidence_stays_active():
    frame = np.zeros((99, 131, 3), np.uint8)
    assert not SegmentationEngine(FixedClass(0), "cpu", floor_fraction=0).predict(frame)[1].any()
    assert not SegmentationEngine(FixedClass(1, 0.1), "cpu").predict(frame)[1].any()


def wait_until(condition, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    assert condition()


def test_control_speed_watchdogs_and_emergency(monkeypatch):
    sent = []
    monkeypatch.setattr(RoverController, "_send_command", lambda self, command, speed: sent.append((command, speed)))
    controller = RoverController(dry_run=True)
    try:
        controller.manual_drive("forward", 25)
        wait_until(lambda: ("forward", 25) in sent)
        controller.manual_drive("forward", 70)
        wait_until(lambda: ("forward", 70) in sent)
        wait_until(lambda: controller.status()["command"] == "stop")
        controller.set_mode("auto")
        path = [(320, 480), (320, 420), (320, 400)]
        controller.update_autonomous(path, 640, time.monotonic() - 2)
        assert controller.status()["command"] == "stop"
        controller.update_autonomous(path, 640, time.monotonic())
        assert controller.status()["command"] == "forward"
        wait_until(lambda: controller.status()["command"] == "stop")
        controller.set_emergency(True)
        controller.manual_drive("forward", 50)
        assert controller.status()["command"] == "stop"
    finally:
        controller.close()
    assert sent[-1][0] == "stop"


def test_dashboard_api_and_camera_disconnect():
    server = DashboardServer("127.0.0.1", 0, dry_run=True)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urlopen(base + "/api/status", timeout=2) as response:
            assert json.load(response)["dry_run"]
        request = Request(base + "/api/drive", data=b'{"command":"forward","speed":40}',
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            assert json.load(response)["command"] == "forward"
        server.camera_disconnected()
        assert server.controller.status()["command"] == "stop"
        server.publish(np.zeros((64, 64, 3), np.uint8), {})
        assert server.state.jpeg_frame.startswith(b"\xff\xd8")
    finally:
        server.stop()


def test_fused_model_preserves_predictions():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1] / "semantic segmentation"
    reference = load_segmentation_model(root / "bisenetv2.py", root / "model_final.pth", "cpu", fuse=False)
    optimized = load_segmentation_model(root / "bisenetv2.py", root / "model_final.pth", "cpu", fuse=True)
    torch.manual_seed(5)
    x = torch.randn(1, 3, 128, 192).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        expected, actual = reference(x), optimized(x)
    torch.testing.assert_close(expected, actual, atol=0.0002, rtol=0.0002)
    assert (expected.argmax(1) == actual.argmax(1)).float().mean() > 0.999


def test_fragmented_mjpeg_and_reconnection():
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
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    source = FrameSource(f"http://127.0.0.1:{http.server_port}/stream", read_timeout=0.4)
    source.start()
    try:
        first = source.read(timeout=1)
        assert first is not None and first.frame.shape == image.shape
        newest = first
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and newest.epoch == first.epoch:
            packet = source.read(newest.sequence, timeout=0.1)
            if packet is not None:
                newest = packet
        assert newest.epoch > first.epoch and newest.sequence > first.sequence
    finally:
        source.close()
        http.shutdown()
        http.server_close()
        thread.join(timeout=1)


def test_stationary_camera_with_noise_jpeg_and_subpixel_jitter():
    rng = np.random.default_rng(121)
    image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)
    vo = FrameVisualOdometry(max_features=800)
    for i in range(90):
        shift = rng.uniform(-0.2, 0.2, 2)
        frame = cv2.warpAffine(image, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (640, 480), borderMode=cv2.BORDER_REFLECT)
        frame = np.clip(frame.astype(float) + rng.normal(0, 2, frame.shape) + rng.uniform(-5, 5), 0, 255).astype(np.uint8)
        encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65 + i % 20])[1]
        vo.update(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), timestamp=i / 30)
    np.testing.assert_array_equal(vo.get_coordinates(), 0)
    assert vo.distance == 0
    assert len(vo.trajectory) == 1


def test_no_unit_step_after_landmark_scale_is_lost():
    points, k = scene()
    shifted = points - np.array([0, 0, 0.3])
    vo = FrameVisualOdometry(intrinsic_matrix=k)
    vo.distance = 10
    pose = vo._essential(points[:, :2] / points[:, 2:3], shifted[:, :2] / shifted[:, 2:3], np.full_like(points, np.nan), None)
    assert pose is None


def test_browser_session_and_invalid_upload():
    server = DashboardServer('127.0.0.1', 0, dry_run=True)
    server.start()
    base = f'http://127.0.0.1:{server.port}'
    try:
        assert server.session() == (0, False)
        for endpoint in ['/api/session/start', '/api/session/heartbeat', '/api/vo/reset']:
            with urlopen(Request(base + endpoint, data=b'{}', headers={'Content-Type': 'application/json'}), timeout=2):
                pass
        revision, active = server.session()
        assert revision == 2 and active
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/upload', data=b'not a video', headers={'X-Filename': 'test.mp4'}), timeout=2)
        assert error.value.code == 400
        assert server.take_source() is None
    finally:
        server.stop()


def test_video_upload_and_source_switch(tmp_path):
    path = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((64, 96, 3), 110, np.uint8))
    writer.release()
    server = DashboardServer('127.0.0.1', 0, dry_run=True)
    server.start()
    base = f'http://127.0.0.1:{server.port}'
    try:
        request = Request(base + '/api/upload', data=path.read_bytes(), headers={'X-Filename': '../../test.avi', 'Content-Type': 'application/octet-stream'})
        with urlopen(request, timeout=3) as response:
            assert json.load(response)['uploaded']
        selected = server.take_source()[0]
        assert Path(selected).parent == Path(server.upload_directory.name)
        assert server.controller.dry_run
        with urlopen(Request(base + '/api/source/live', data=b'{}'), timeout=2):
            pass
        assert server.take_source() == (None,)
    finally:
        server.stop()


def test_all_display_views_include_full_size_astar():
    from combined_run import create_views
    frame = np.zeros((480, 640, 3), np.uint8)
    mask = np.full((480, 640), 255, np.uint8)
    path, status = plan_shortest_path(mask)
    result = {'frame': frame, 'color': frame.copy(), 'traversable': mask, 'path': path, 'status': status}
    views = create_views(frame, result, FrameVisualOdometry(), 10, status, 100)
    assert set(views) == {'all', 'vo', 'segmentation', 'astar'}
    assert views['astar'].shape == frame.shape
    assert np.any(np.all(views['astar'] == [255, 0, 255], axis=2))


def test_motion_gate_accepts_sustained_real_translation(monkeypatch):
    world, k = scene()
    def project(points):
        pixels = (k @ points.T).T
        return (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)
    previous = project(world)
    current = previous.copy()
    vo = FrameVisualOdometry(intrinsic_matrix=k)
    monkeypatch.setattr(vo, 'detect_features', lambda *args: previous.reshape(-1, 1, 2))
    monkeypatch.setattr(vo, 'track_features', lambda *args: (previous, current, np.arange(len(world))))
    image = np.zeros((480, 640), np.uint8)
    vo.update(image, timestamp=0)
    for index in range(1, 12):
        current = project(world - np.array([0, 0, index * 0.1]))
        vo.update(image, timestamp=index / 30)
        if vo.distance > 0:
            break
    assert vo.distance > 0
    assert vo.get_coordinates()[2] > 0.9
