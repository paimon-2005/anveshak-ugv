"""
Synthetic KITTI Dataset Generator for Immediate Verification.
Generates a valid KITTI-structured sequence (Sequence 99) with camera calibration,
ground truth poses, and realistic perspective-projected synthetic frames.
Allows zero-setup testing without downloading 20GB+ KITTI archives.
"""

import math
from pathlib import Path

import cv2
import numpy as np


def generate_sample_kitti_sequence(
    output_base: Path = Path("data/kitti/dataset"),
    seq_id: str = "99",
    num_frames: int = 80,
    img_w: int = 1241,
    img_h: int = 376
):
    """
    Generate synthetic sequence `seq_id` conforming to the KITTI format.
    """
    output_base = Path(output_base)
    seq_str = str(seq_id).zfill(2)
    seq_dir = output_base / "sequences" / seq_str
    img_dir = seq_dir / "image_0"
    pose_dir = output_base / "poses"

    img_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating synthetic KITTI dataset in: {output_base}")
    print(f"    Sequence ID: {seq_str} ({num_frames} frames)")

    # 1. Calibration Matrix (KITTI-like focal length and principal point)
    fx, fy = 718.856, 718.856
    cx, cy = img_w / 2.0, img_h / 2.0
    P0 = np.array([
        [fx,  0.0, cx,  0.0],
        [0.0, fy,  cy,  0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    calib_file = seq_dir / "calib.txt"
    with open(calib_file, "w") as f:
        p0_str = " ".join(f"{val:.12e}" for val in P0.flatten())
        f.write(f"P0: {p0_str}\n")
        f.write(f"P1: {p0_str}\n")
        f.write(f"P2: {p0_str}\n")
        f.write(f"P3: {p0_str}\n")
    print(f"    [+] Created: {calib_file}")

    # 2. Generate 3D World Landmarks (Features scattered along the road corridor)
    np.random.seed(42)
    num_landmarks = 1500
    # Landmarks distributed: X in [-15, 15], Y in [-5, 5], Z in [5, 250]
    landmarks = np.zeros((num_landmarks, 3), dtype=np.float64)
    landmarks[:, 0] = np.random.uniform(-18.0, 18.0, num_landmarks)
    landmarks[:, 1] = np.random.uniform(-4.0, 3.0, num_landmarks)   # Road / roadside objects
    landmarks[:, 2] = np.random.uniform(5.0, 220.0, num_landmarks)

    # 3. Simulate Vehicle Trajectory (Smooth S-Curve moving along Z)
    # Forward speed = 1.0 m/frame
    poses_3x4 = []
    cam_positions = []
    cam_rotations = []

    for i in range(num_frames):
        t = i * 0.08  # Parameter
        # Forward displacement
        pos_z = i * 1.2
        # Gentle S-curve lateral displacement
        pos_x = 8.0 * math.sin(t * 0.8)
        # Gentle ground elevation variation
        pos_y = 0.3 * math.sin(t * 0.4)

        # Yaw heading angle (tangent to trajectory)
        dx = 8.0 * 0.8 * math.cos(t * 0.8) * 0.08
        dz = 1.2
        yaw = math.atan2(dx, dz)  # Rotation around Y axis

        # Rotation matrix around Y axis in KITTI camera coords:
        # [ cos(yaw)  0  sin(yaw) ]
        # [    0      1     0     ]
        # [-sin(yaw)  0  cos(yaw) ]
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c]
        ], dtype=np.float64)

        cam_pos = np.array([pos_x, pos_y, pos_z], dtype=np.float64)
        cam_positions.append(cam_pos)
        cam_rotations.append(R)

        # Pose in KITTI format: Camera-to-World transformation [R | t]
        pose = np.hstack([R, cam_pos.reshape(3, 1)])
        poses_3x4.append(pose.flatten())

    # Write poses/99.txt
    pose_file = pose_dir / f"{seq_str}.txt"
    with open(pose_file, "w") as f:
        for p in poses_3x4:
            f.write(" ".join(f"{val:.6e}" for val in p) + "\n")
    print(f"    [+] Created: {pose_file}")

    # 4. Render Synthetic Camera Frames via Perspective Projection
    print(f"    [+] Rendering {num_frames} frames...")
    for frame_idx in range(num_frames):
        img = np.full((img_h, img_w), 40, dtype=np.uint8)

        # Draw road / horizon
        horizon_y = int(cy)
        img[horizon_y:, :] = 55

        # Road lanes
        cv2.line(img, (0, img_h), (int(cx - 40), horizon_y), 75, 2)
        cv2.line(img, (img_w, img_h), (int(cx + 40), horizon_y), 75, 2)

        # Transform world landmarks to camera coordinates: P_cam = R^T * (P_world - t)
        R = cam_rotations[frame_idx]
        t = cam_positions[frame_idx]

        diff = landmarks - t.reshape(1, 3)
        pts_cam = (R.T @ diff.T).T

        # Keep points in front of the camera (Z > 1.0)
        valid = pts_cam[:, 2] > 1.5
        pts_valid = pts_cam[valid]

        # Project to 2D image plane
        u = (fx * (pts_valid[:, 0] / pts_valid[:, 2]) + cx).astype(np.int32)
        v = (fy * (pts_valid[:, 1] / pts_valid[:, 2]) + cy).astype(np.int32)

        in_frame = (u >= 4) & (u < img_w - 4) & (v >= 4) & (v < img_h - 4)
        u_in = u[in_frame]
        v_in = v[in_frame]
        z_in = pts_valid[in_frame, 2]

        for ui, vi, zi in zip(u_in, v_in, z_in):
            # Size and brightness inversely proportional to distance
            radius = max(1, int(4.0 / max(zi * 0.05, 1.0)))
            brightness = min(255, int(200 - zi * 0.6))
            brightness = max(110, brightness)
            cv2.circle(img, (ui, vi), radius, brightness, -1)
            # Add micro-texture corner
            cv2.rectangle(img, (ui - radius, vi - radius), (ui + radius, vi + radius), brightness, 1)

        # Save frame
        frame_filename = img_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(frame_filename), img)

    print(f"    [+] Finished generating synthetic dataset in {output_base}")
    return seq_dir


if __name__ == "__main__":
    generate_sample_kitti_sequence()
