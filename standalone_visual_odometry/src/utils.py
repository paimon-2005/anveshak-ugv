import time
from collections import deque
from typing import Deque, List

import numpy as np

def make_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def compute_trajectory_length(trajectory: List[np.ndarray]) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(trajectory), axis=0), axis=1).sum())

def compute_ate_rmse(est_trajectory: List[np.ndarray], gt_trajectory: List[np.ndarray]) -> float:
    n = min(len(est_trajectory), len(gt_trajectory))
    if not n:
        return 0.0
    err = np.asarray(est_trajectory[:n]) - np.asarray(gt_trajectory[:n])
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

class FPSCounter:
    def __init__(self, window_size: int = 30):
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.timestamps.append(dt)
        if not self.timestamps:
            return 0.0
        avg = sum(self.timestamps) / len(self.timestamps)
        return 1.0 / avg if avg > 0 else 0.0
