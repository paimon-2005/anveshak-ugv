from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import urlopen

import cv2
import numpy as np


class KittiDataset:
    def __init__(self, base_dir: Union[str, Path], sequence: str = "00", camera_dir: str = "image_0"):
        self.base_dir = Path(base_dir)
        self.sequence = str(sequence).zfill(2)
        self.camera_dir = camera_dir
        self.seq_dir = self.base_dir / "sequences" / self.sequence
        self.img_dir = self.seq_dir / self.camera_dir
        self.calib_file = self.seq_dir / "calib.txt"
        self.pose_file = self.base_dir / "poses" / f"{self.sequence}.txt"
        self.live = False
        if not self.seq_dir.exists():
            raise FileNotFoundError(f"KITTI sequence directory not found: {self.seq_dir}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Camera directory not found: {self.img_dir}")
        if not self.calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calib_file}")
        self.K = self._load_calibration()
        self.image_paths = sorted((*self.img_dir.glob("*.png"), *self.img_dir.glob("*.jpg")))
        if not self.image_paths:
            raise ValueError(f"No images found in camera folder: {self.img_dir}")
        self.ground_truth_poses = self._load_poses()

    def _load_calibration(self) -> np.ndarray:
        tag = next((f"P{i}:" for i in range(4) if f"image_{i}" in self.camera_dir), "P0:")
        calibration = {}
        with open(self.calib_file) as handle:
            for line in handle:
                if ":" in line:
                    key, value = line.split(":", 1)
                    calibration[key.strip() + ":"] = np.fromstring(value, sep=" ")
        if tag not in calibration:
            tag = next((key for key in ("P0:", "P2:", "P1:", "P3:") if key in calibration), None)
        if tag is None:
            raise ValueError(f"Could not find projection matrix in {self.calib_file}")
        return calibration[tag].reshape(3, 4)[:3, :3].copy()

    def _load_poses(self) -> Optional[List[np.ndarray]]:
        if not self.pose_file.exists():
            return None
        poses = []
        with open(self.pose_file) as handle:
            for line in handle:
                values = np.fromstring(line, sep=" ")
                if len(values) == 12:
                    transform = np.eye(4, dtype=np.float64)
                    transform[:3, :] = values.reshape(3, 4)
                    poses.append(transform)
        return poses or None

    def get_ground_truth_scale(self, index: int) -> Optional[float]:
        if self.ground_truth_poses is None or not 0 < index < len(self.ground_truth_poses):
            return None
        poses = self.ground_truth_poses
        return float(np.linalg.norm(poses[index][:3, 3] - poses[index - 1][:3, 3]))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to read image at: {self.image_paths[index]}")
        poses = self.ground_truth_poses
        return image, poses[index] if poses is not None and index < len(poses) else None

    def release(self):
        return


class VideoDataset:
    def __init__(self, video_path: Union[str, Path], focal_length: Optional[float] = None):
        self.source = str(video_path)
        self.is_http = self.source.startswith(("http://", "https://"))
        self.live = self.is_http or self.source.startswith("rtsp://") or self.source.isdecimal()
        self.cap = None
        self.response = None
        self.buffer = bytearray()
        self._buffered_frame = None
        self._next_frame = 0
        self.ground_truth_poses = None
        if self.is_http:
            self.response = urlopen(self.source, timeout=15)
            first_frame = self._read_http_frame()
            self.total_frames = 0
        else:
            capture_source = int(self.source) if self.source.isdecimal() else self.source
            if not self.live and not Path(self.source).is_file():
                raise FileNotFoundError(f"Video file not found at: {self.source}")
            self.cap = cv2.VideoCapture(capture_source)
            if not self.cap.isOpened():
                raise IOError(f"Could not open video source: {self.source}")
            ok, first_frame = self.cap.read()
            if not ok or first_frame is None:
                raise IOError(f"Could not read video source: {self.source}")
            reported_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total_frames = int(reported_frames) if np.isfinite(reported_frames) and 0 < reported_frames < 1_000_000_000 else 0
        self._buffered_frame = first_frame
        height, width = first_frame.shape[:2]
        focal = focal_length if focal_length is not None else width * 0.85
        self.K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)

    def _read_http_frame(self):
        while True:
            start = self.buffer.find(bytes((255, 216)))
            if start >= 0:
                end = self.buffer.find(bytes((255, 217)), start + 2)
                if end >= 0:
                    jpeg = bytes(self.buffer[start:end + 2])
                    del self.buffer[:end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
            chunk = self.response.read1(65536) if hasattr(self.response, "read1") else self.response.read(65536)
            if not chunk:
                raise IOError(f"Video stream ended: {self.source}")
            self.buffer.extend(chunk)
            if len(self.buffer) > 8 * 1024 * 1024:
                marker = self.buffer.rfind(bytes((255, 216)))
                self.buffer[:] = self.buffer[marker:] if marker >= 0 else self.buffer[-1:]

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._buffered_frame is not None:
            frame, self._buffered_frame = self._buffered_frame, None
        elif self.is_http:
            frame = self._read_http_frame()
        else:
            if not self.live and index != self._next_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise IOError(f"Failed to read frame {index} from: {self.source}")
        self._next_frame = index + 1
        if frame.ndim == 2:
            return frame, None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None

    def get_ground_truth_scale(self, index: int) -> Optional[float]:
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.response is not None:
            self.response.close()
