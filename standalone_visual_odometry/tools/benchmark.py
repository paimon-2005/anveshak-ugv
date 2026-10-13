import time
import numpy as np
import cv2
from config import default_config
from src.visual_odometry import VisualOdometry

def main():
    cfg = default_config
    h, w = 376, 1241
    rng = np.random.default_rng(0)
    base = (rng.random((h, w)) * 255).astype(np.uint8)
    frames = [base]
    for i in range(1, 31):
        M = np.float32([[1, 0, i * 0.8], [0, 1, i * 0.15]])
        frames.append(cv2.warpAffine(base, M, (w, h)))

    vo = VisualOdometry(np.array([[718.856,0,607.1928],[0,718.856,185.2157],[0,0,1.0]]), cfg)
    t0 = time.perf_counter()
    for f in frames:
        vo.step(f, scale=1.0)
    elapsed = time.perf_counter() - t0
    print(f"frames={len(frames)} time={elapsed:.3f}s fps={len(frames)/elapsed:.2f}")

if __name__ == "__main__":
    main()
