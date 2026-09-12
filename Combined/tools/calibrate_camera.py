import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Calibrate a pinhole camera using checkerboard photographs")
    parser.add_argument("images", type=Path)
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25)
    parser.add_argument("--output", type=Path, default=Path("camera_calibration.json"))
    args = parser.parse_args()
    if min(args.columns, args.rows) < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("Use positive square size and at least 3 x 3 inner corners")
    grid = np.zeros((args.rows * args.columns, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    objects, corners, names = [], [], []
    size = None
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image_size = image.shape[1], image.shape[0]
        if size is not None and image_size != size:
            raise ValueError("All calibration images must have the same resolution")
        size = image_size
        found, points = cv2.findChessboardCornersSB(image, (args.columns, args.rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            objects.append(grid.copy())
            corners.append(points.astype(np.float32))
            names.append(path.name)
    if len(corners) < 12:
        raise ValueError(f"Found {len(corners)} checkerboards; at least 12 varied views are required")
    rms, camera, distortion, rotations, translations = cv2.calibrateCamera(objects, corners, size, None, None)
    errors = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, r, t, camera, distortion)[0] - pts) ** 2, axis=2))))
              for obj, pts, r, t in zip(objects, corners, rotations, translations)]
    if not np.isfinite(rms) or not np.isfinite(camera).all() or not np.isfinite(distortion).all():
        raise RuntimeError("Calibration failed")
    result = {"camera_matrix": camera.tolist(), "distortion_coefficients": distortion.ravel().tolist(),
              "image_size": list(size), "reprojection_rms_px": float(rms), "views_used": len(corners),
              "per_view_rms_px": dict(zip(names, errors)), "checkerboard_inner_corners": [args.columns, args.rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output} | {len(corners)} views | reprojection RMS {rms:.3f} px")


if __name__ == "__main__":
    main()
