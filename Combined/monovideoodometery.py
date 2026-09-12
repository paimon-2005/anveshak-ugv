import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "semantic segmentation"))

from combined_run import FrameVisualOdometry


class MonoVideoOdometery:
    def __init__(self, img_file_path, pose_file_path=None, focal_length=554.0, pp=(320.0, 240.0),
                 lk_params=None, detector=None, intrinsic_matrix=None, distortion=None, fps=30.0):
        self.files = sorted(p for p in Path(img_file_path).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not self.files:
            raise ValueError("No supported images found")
        self.poses = None
        if pose_file_path:
            rows = np.loadtxt(pose_file_path, dtype=np.float64, ndmin=2)
            if rows.shape[1] != 12 or len(rows) < len(self.files) or not np.isfinite(rows).all():
                raise ValueError("One finite KITTI 3 x 4 pose per image is required")
            poses = np.tile(np.eye(4), (len(rows), 1, 1))
            poses[:, :3, :] = rows.reshape(-1, 3, 4)
            self.poses = np.linalg.inv(poses[0]) @ poses
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        self.vo = FrameVisualOdometry(focal_length=focal_length, principal_point_x=pp[0], principal_point_y=pp[1],
            intrinsic_matrix=intrinsic_matrix, distortion=distortion, metric_scale=self.poses is not None)
        if lk_params:
            self.vo.lk_parameters.update(lk_params)
        if detector is not None:
            self.vo.detector = detector
        self.id = 0
        self.current_frame = None
        self.true_coord = np.full(3, np.nan)
        self.last_scale = None
        self.process_frame()

    def hasNextFrame(self):
        return self.id < len(self.files)

    def process_frame(self):
        if not self.hasNextFrame():
            return False
        self.current_frame = cv2.imread(str(self.files[self.id]), cv2.IMREAD_GRAYSCALE)
        if self.current_frame is None:
            raise ValueError(f"Cannot decode {self.files[self.id]}")
        if self.poses is not None:
            self.true_coord = self.poses[self.id, :3, 3].copy()
            self.last_scale = float(np.linalg.norm(self.true_coord - self.poses[max(0, self.id - 1), :3, 3]))
        self.vo.update(self.current_frame, scale=self.last_scale, timestamp=self.id / self.fps)
        self.id += 1
        return True

    def get_mono_coordinates(self):
        return self.vo.get_coordinates()

    def get_true_coordinates(self):
        return self.true_coord.copy()

    def get_absolute_scale(self):
        return self.last_scale

    @property
    def R(self):
        return self.vo.rotation

    @property
    def t(self):
        return self.vo.translation


MonoVideoOdometry = MonoVideoOdometery
