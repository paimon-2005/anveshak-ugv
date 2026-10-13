from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, default_config
from src.utils import make_transform_matrix

class VisualOdometry:
    def __init__(self, intrinsic_matrix: np.ndarray, config: Optional[AppConfig] = None):
        self.config = config or default_config
        self.K = intrinsic_matrix.astype(np.float64)
        self.focal = float(self.K[0, 0])
        self.pp = (float(self.K[0, 2]), float(self.K[1, 2]))
        self.prev_image = None
        self.prev_points = None
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        self.trajectory: List[np.ndarray] = [self.cur_t.flatten().copy()]
        self.poses: List[np.ndarray] = [make_transform_matrix(self.cur_R, self.cur_t)]

        cfg = self.config.tracker
        self.lk_params = dict(
            winSize=cfg.win_size,
            maxLevel=cfg.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.max_iters, cfg.epsilon),
        )
        self.detector = self._create_detector()
        self.ransac_method = getattr(cv2, self.config.pose.estimator, cv2.RANSAC)
        self.roi_mask = None
        self.last_valid_t = np.zeros((3, 1), dtype=np.float64)

    def _create_detector(self):
        cfg = self.config.detector
        return cv2.FastFeatureDetector_create(
            threshold=cfg.fast_threshold,
            nonmaxSuppression=cfg.fast_nonmax_suppression,
        ) if cfg.detector_type == "FAST" else None

    def _get_roi_mask(self, h: int, w: int) -> np.ndarray:
        if self.roi_mask is not None and self.roi_mask.shape == (h, w):
            return self.roi_mask
        cfg = self.config.detector
        mask = np.full((h, w), 255, dtype=np.uint8)
        if cfg.mask_sky:
            mask[:int(h * cfg.sky_top_fraction)] = 0
        if cfg.mask_vehicle_hood:
            mask[int(h * (1 - cfg.hood_bottom_fraction)):] = 0
        self.roi_mask = mask
        return mask

    def detect_features(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = self._get_roi_mask(h, w)
        cfg = self.config.detector

        if cfg.detector_type == "FAST" and self.detector is not None:
            kps = self.detector.detect(image, mask)
            if not kps:
                return np.empty((0, 1, 2), dtype=np.float32)

            if cfg.use_grid_bucketing:
                rows, cols = cfg.grid_rows, cfg.grid_cols
                cw, ch = w / cols, h / rows
                buckets = [[] for _ in range(rows * cols)]
                for kp in kps:
                    c = min(int(kp.pt[0] / cw), cols - 1)
                    r = min(int(kp.pt[1] / ch), rows - 1)
                    buckets[r * cols + c].append(kp)

                points = []
                limit = cfg.max_features_per_cell
                for cell in buckets:
                    if len(cell) > limit:
                        cell.sort(key=lambda k: k.response, reverse=True)
                        cell = cell[:limit]
                    points.extend(k.pt for k in cell)
                return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

            kps = self.detector.detect(image, mask)
            if len(kps) > cfg.target_num_features:
                step = len(kps) / cfg.target_num_features
                kps = [kps[int(i * step)] for i in range(cfg.target_num_features)]
            return np.asarray([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)

        pts = cv2.goodFeaturesToTrack(
            image, maxCorners=cfg.target_num_features, qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance, mask=mask,
        )
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def track_features(self, prev_img: np.ndarray, curr_img: np.ndarray, prev_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if prev_pts is None or not len(prev_pts):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,), dtype=bool)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **self.lk_params)
        valid = status.ravel().astype(bool)

        if self.config.tracker.enable_fb_check and valid.any():
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_img, prev_img, curr_pts, None, **self.lk_params
            )
            fb_err = np.sqrt(np.sum((prev_pts - back_pts) ** 2, axis=2)).ravel()
            valid &= back_status.ravel().astype(bool) & (fb_err < self.config.tracker.max_fb_error)

        h, w = curr_img.shape[:2]
        p = curr_pts.reshape(-1, 2)
        valid &= (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        return prev_pts[valid].reshape(-1, 2), p[valid], valid

    @staticmethod
    def _is_rotation_valid(R: np.ndarray) -> bool:
        return np.linalg.norm(R @ R.T - np.eye(3), "fro") <= 0.1 and np.linalg.det(R) >= 0.9

    def step(self, curr_image: np.ndarray, scale: Optional[float] = None) -> dict:
        cfg = self.config
        scale = cfg.pose.default_scale if scale is None or scale <= 0 else scale

        if self.prev_image is None:
            self.prev_image = curr_image
            self.prev_points = self.detect_features(curr_image)
            return {
                "success": True,
                "position": self.cur_t.flatten().copy(),
                "rotation": self.cur_R.copy(),
                "tracked_prev": np.empty((0, 2)),
                "tracked_curr": np.empty((0, 2)),
                "inliers_mask": np.empty((0,), dtype=bool),
                "num_features": len(self.prev_points),
                "num_inliers": 0,
            }

        pts_prev, pts_curr, _ = self.track_features(self.prev_image, curr_image, self.prev_points)
        inliers_mask = np.zeros(len(pts_curr), dtype=bool)
        num_inliers = 0
        disp = np.linalg.norm(pts_curr - pts_prev, axis=1) if len(pts_curr) else np.array([0.0])
        stationary = scale < 0.05 and float(np.median(disp)) < cfg.pose.min_stationary_disp

        if not stationary and len(pts_curr) >= cfg.pose.min_inliers:
            # Estimate the motion from previous -> current image.
            # Passing points in this order keeps the recovered translation/rotation
            # convention consistent with the trajectory integration below.
            E, mask = cv2.findEssentialMat(
                pts_prev, pts_curr, focal=self.focal, pp=self.pp,
                method=self.ransac_method, prob=cfg.pose.ransac_prob,
                threshold=cfg.pose.ransac_threshold,
            )
            if E is not None:
                # RANSAC may return multiple 3x3 essential matrices stacked
                # vertically. Try each candidate until one gives a valid pose.
                candidates = [E[i:i+3, :] for i in range(0, E.shape[0] - 2, 3)
                              if E[i:i+3, :].shape == (3, 3)]
                if not candidates and E.shape == (3, 3):
                    candidates = [E]
                num, R, t, mask_pose = 0, np.eye(3), np.zeros((3, 1)), None
                for Ec in candidates:
                    try:
                        n_i, R_i, t_i, m_i = cv2.recoverPose(
                            Ec, pts_prev, pts_curr, focal=self.focal, pp=self.pp, mask=mask
                        )
                        if int(n_i) > num:
                            num, R, t, mask_pose = int(n_i), R_i, t_i, m_i
                    except cv2.error:
                        continue
                num_inliers = int(num)
                inliers_mask = mask_pose.ravel() > 0 if mask_pose is not None else np.zeros(len(pts_curr), dtype=bool)
                e_inliers = int(np.sum(mask)) if mask is not None else 0
                valid_pose = (
                    num_inliers >= cfg.pose.min_inliers
                    and e_inliers >= cfg.pose.min_inliers
                    and self._is_rotation_valid(R)
                )
                if valid_pose:
                    test_t = self.cur_t + scale * (self.cur_R @ t)
                    if np.linalg.norm(test_t - self.last_valid_t) < 5.0:
                        self.cur_t = test_t
                        self.cur_R = self.cur_R @ R
                        self.last_valid_t = self.cur_t.copy()
                        success = True
                    else:
                        success = False
                else:
                    success = False
            else:
                success = False
        else:
            success = True

        self.trajectory.append(self.cur_t.flatten().copy())
        self.poses.append(make_transform_matrix(self.cur_R, self.cur_t))

        self.prev_points = (
            self.detect_features(curr_image)
            if len(pts_curr) < cfg.detector.min_num_features
            else pts_curr.reshape(-1, 1, 2)
        )
        self.prev_image = curr_image

        return {
            "success": success,
            "position": self.cur_t.flatten().copy(),
            "rotation": self.cur_R.copy(),
            "tracked_prev": pts_prev,
            "tracked_curr": pts_curr,
            "inliers_mask": inliers_mask,
            "num_features": len(self.prev_points),
            "num_inliers": num_inliers,
        }
