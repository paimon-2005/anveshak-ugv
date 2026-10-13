import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


class CameraCalibration:
    def __init__(self, width=640, height=480, path=None):
        self.size = int(width), int(height)
        if min(self.size) < 32:
            raise ValueError("Frame dimensions must be at least 32 pixels")
        self.calibrated = path is not None
        data = json.loads(Path(path).read_text()) if path else {
            "camera_matrix": [[554, 0, 320], [0, 554, 240], [0, 0, 1]],
            "image_size": [640, 480], "distortion_coefficients": [0] * 5}
        original = np.asarray(data["image_size"], np.float64)
        self.K = np.asarray(data["camera_matrix"], np.float64)
        self.distortion = np.asarray(data.get("distortion_coefficients", [0] * 5), np.float64).ravel()
        if original.shape != (2,) or not np.isfinite(original).all() or np.any(original <= 0):
            raise ValueError("Invalid calibration image_size")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Invalid camera_matrix")
        if not np.allclose(self.K[2], [0, 0, 1]) or abs(self.K[0, 1]) > 1e-8 or abs(self.K[1, 0]) > 1e-8:
            raise ValueError("A standard zero-skew OpenCV pinhole calibration is required")
        if len(self.distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("Invalid distortion coefficients")
        self.K = np.diag([width / original[0], height / original[1], 1]) @ self.K
        self.maps = None
        self.valid_mask = np.full((height, width), 255, np.uint8)
        if np.any(self.distortion):
            self.maps = cv2.initUndistortRectifyMap(self.K, self.distortion, None, self.K, self.size, cv2.CV_32FC1)
            x, y = self.maps
            self.valid_mask = ((x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)).astype(np.uint8) * 255

    def prepare(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA if frame.shape[1] > self.size[0] else cv2.INTER_LINEAR)
        return cv2.remap(frame, *self.maps, interpolation=cv2.INTER_LINEAR) if self.maps is not None else frame


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    sequence: int
    timestamp: float
    captured_at: float
    epoch: int


class FrameSource:
    def __init__(self, source, read_timeout=2.0):
        self.source = str(source)
        self.web = self.source.startswith(("http://", "https://"))
        self.live = self.web or self.source.isdecimal() or self.source.startswith("rtsp://")
        if not self.live and not Path(self.source).is_file():
            raise FileNotFoundError(self.source)
        self.read_timeout = float(read_timeout)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.packet = None
        self.frame_queue = deque(maxlen=8)
        self.last_captured_at = None
        self.sequence = -1
        self.epoch = 0
        self.connected = self.ended = False
        self.capture = self.thread = None
        self.error = None
        self.fps = 30.0
        self.file_index = 0
        self.started_at = time.monotonic()

    def start(self):
        self.started_at = time.monotonic()
        if self.live:
            self.thread = threading.Thread(target=self._worker, name="camera-reader", daemon=True)
            self.thread.start()
        else:
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"Cannot decode video: {self.source}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if np.isfinite(fps) and fps > 0 else 30.0
            self.connected = True

    def _publish(self, frame):
        now = time.monotonic()
        with self.condition:
            self.sequence += 1
            packet = FramePacket(frame, self.sequence, now - self.started_at, now, self.epoch)
            self.packet = packet
            self.frame_queue.append(packet)
            self.last_captured_at = now
            self.connected, self.error = True, None
            self.condition.notify_all()

    def _mjpeg(self):
        with urlopen(self.source, timeout=self.read_timeout) as response:
            buffer = bytearray()
            last_frame = time.monotonic()
            while not self.stop_event.is_set():
                chunk = response.read1(65536)
                if not chunk:
                    raise ConnectionError("Camera stream ended")
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        buffer[:] = buffer[-1:]
                        break
                    if start:
                        del buffer[:start]
                    end = buffer.find(b"\xff\xd9", 2)
                    if end < 0:
                        break
                    jpeg = bytes(buffer[:end + 2])
                    del buffer[:end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._publish(frame)
                        last_frame = time.monotonic()
                if len(buffer) > 4 * 1024 * 1024 or time.monotonic() - last_frame > self.read_timeout:
                    raise ConnectionError("No complete JPEG received")

    def _opencv_live(self):
        source = int(self.source) if self.source.isdecimal() else self.source
        capture = cv2.VideoCapture(source) if isinstance(source, int) else cv2.VideoCapture(source, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.read_timeout * 1000), cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout * 1000)])
        try:
            if not capture.isOpened():
                raise ConnectionError("Cannot open camera")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ConnectionError("Camera frame unavailable")
                self._publish(frame)
        finally:
            capture.release()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                with self.condition:
                    self.epoch += 1
                    self.frame_queue.clear()
                    self.packet = None
                    self.last_captured_at = None
                self._mjpeg() if self.web else self._opencv_live()
            except Exception as error:
                with self.condition:
                    self.connected, self.packet, self.error = False, None, str(error)
                    self.frame_queue.clear()
                    self.last_captured_at = None
                    self.condition.notify_all()
            self.stop_event.wait(0.5)

    def read(self, after_sequence=-1, timeout=0.1):
        if not self.live:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.connected, self.ended = False, True
                return None
            packet = FramePacket(frame, self.file_index, self.file_index / self.fps, time.monotonic(), 0)
            self.file_index += 1
            return packet

        def next_packet():
            for candidate in self.frame_queue:
                if candidate.sequence > after_sequence:
                    return candidate
            return None

        with self.condition:
            packet = next_packet()
            if packet is None:
                self.condition.wait_for(
                    lambda: self.stop_event.is_set() or next_packet() is not None,
                    timeout=timeout,
                )
                packet = next_packet()
            return packet

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.frame_queue.clear()
            self.packet = None
            self.last_captured_at = None
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=self.read_timeout + 1)
        if self.capture is not None:
            self.capture.release()
        self.connected = False
