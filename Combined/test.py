import argparse
from pathlib import Path

import cv2
import numpy as np

from monovideoodometery import MonoVideoOdometery
from camera import CameraCalibration
from combined_run import draw_trajectory


def main():
    parser = argparse.ArgumentParser(description="Evaluate VO on an ordered image sequence")
    parser.add_argument("--img_path", required=True, type=Path)
    parser.add_argument("--pose_path", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    images = sorted(p for p in args.img_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise ValueError("No images found")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError("First image cannot be decoded")
    calibration = CameraCalibration(first.shape[1], first.shape[0], args.calibration)
    vo = MonoVideoOdometery(args.img_path, args.pose_path, intrinsic_matrix=calibration.K,
                           distortion=calibration.distortion, fps=args.fps)
    errors = []
    try:
        while True:
            position, truth = vo.get_mono_coordinates(), vo.get_true_coordinates()
            if np.isfinite(truth).all():
                errors.append(float(np.sum((position - truth) ** 2)))
            if args.show:
                cv2.imshow("Frame", vo.current_frame)
                cv2.imshow("Trajectory", draw_trajectory(vo.vo))
                if cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                    break
            if not vo.process_frame():
                break
    finally:
        if args.show:
            cv2.destroyAllWindows()
    print(f"Frames: {vo.id} | Distance: {vo.vo.distance:.3f} {vo.vo.units}")
    if errors:
        print(f"ATE RMSE in first-camera coordinates: {np.sqrt(np.mean(errors)):.4f} m")
        print("Ground-truth distances were supplied to VO for this evaluation.")


if __name__ == "__main__":
    main()
