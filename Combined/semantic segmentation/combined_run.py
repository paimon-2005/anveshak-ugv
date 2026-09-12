import argparse
import csv
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import CameraCalibration, FrameSource, ScaleSamples
from navigation import astar_search, draw_planned_path, nearest_free_cell, plan_shortest_path
from segmentation import SegmentationEngine, load_segmentation_model
from web_dashboard import DashboardServer


BASE_DIRECTORY = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIRECTORY / "bisenetv2.py"
WEIGHTS_FILE = BASE_DIRECTORY / "model_final.pth"
STREAM_URL = "http://192.168.4.1:81/stream"


class FrameVisualOdometry:
    def __init__(self, width=640, height=480, focal_length=554.0, principal_point_x=320.0,
                 principal_point_y=240.0, intrinsic_matrix=None, distortion=None,
                 max_features=1200, metric_scale=False, max_gap=0.75):
        self.K = np.array(intrinsic_matrix if intrinsic_matrix is not None else
                          [[focal_length, 0, principal_point_x], [0, focal_length, principal_point_y], [0, 0, 1]], dtype=np.float64)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Invalid camera intrinsics")
        self.distortion = None if distortion is None else np.asarray(distortion, dtype=np.float64)
        self.focal = float((self.K[0, 0] + self.K[1, 1]) / 2)
        self.max_features = int(max_features)
        if self.max_features < 40:
            raise ValueError("At least 40 features are required")
        self.minimum_features = max(40, int(self.max_features * 0.65))
        self.minimum_inliers = 20
        self.max_gap = float(max_gap)
        self.metric_scale = bool(metric_scale)
        self.rotation = np.eye(3)
        self.translation = np.zeros((3, 1))
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.lk_parameters = dict(winSize=(21, 21), maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.motion_threshold = 1.25
        self.bootstrap_votes = 0
        self.bootstrap_direction = None
        self.ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        self.distance = 0.0
        self.last_step = 1.0
        self.scale_restarts = 0
        self.trajectory = deque([np.zeros(3)], maxlen=5000)
        self.reset_tracking()

    @property
    def units(self):
        return "m" if self.metric_scale else "relative units"

    def reset_tracking(self):
        self.previous_gray = None
        self.previous_points = np.empty((0, 1, 2), dtype=np.float32)
        self.landmarks = np.empty((0, 3))
        self.last_timestamp = None
        self.pending_scale = 0.0
        self.held_frames = 0
        self.tracked_features = 0
        self.tracked_inliers = 0
        self.status = "INITIALIZING"
        self.bootstrap_votes = 0
        self.bootstrap_direction = None

    def detect_features(self, gray, existing=None):
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        mask[:int(h * 0.18)] = 0
        mask[int(h * 0.84):] = 0
        mask[:, :5] = 0
        mask[:, -5:] = 0
        if existing is not None:
            for x, y in existing.reshape(-1, 2):
                cv2.circle(mask, (round(float(x)), round(float(y))), 8, 0, -1)
        budget = self.max_features - (len(existing) if existing is not None else 0)
        if budget <= 0:
            return np.empty((0, 1, 2), np.float32)
        buckets = [[] for _ in range(18)]
        for kp in self.detector.detect(gray, mask):
            index = min(2, int(kp.pt[1] * 3 / h)) * 6 + min(5, int(kp.pt[0] * 6 / w))
            buckets[index].append(kp)
        points = []
        for bucket in buckets:
            count = 0
            for kp in sorted(bucket, key=lambda k: k.response, reverse=True):
                x, y = round(kp.pt[0]), round(kp.pt[1])
                if not mask[y, x]:
                    continue
                points.append(kp.pt)
                cv2.circle(mask, (x, y), 7, 0, -1)
                count += 1
                if count >= math.ceil(budget / 12) or len(points) >= budget:
                    break
            if len(points) >= budget:
                break
        if len(points) < budget:
            extra = cv2.goodFeaturesToTrack(gray, maxCorners=budget - len(points), qualityLevel=0.01, minDistance=8, mask=mask)
            if extra is not None:
                points.extend(extra.reshape(-1, 2).tolist())
        return np.asarray(points, np.float32).reshape(-1, 1, 2)

    def track_features(self, previous_gray, current_gray, previous_points):
        empty = np.empty((0, 2), np.float32)
        no_indices = np.empty(0, np.int64)
        if previous_points is None or not len(previous_points):
            return empty, empty.copy(), no_indices
        current, status, error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None, **self.lk_parameters)
        if current is None or status is None:
            return empty, empty.copy(), no_indices
        p = current.reshape(-1, 2)
        h, w = current_gray.shape
        valid = status.ravel().astype(bool) & np.isfinite(p).all(axis=1)
        valid &= (p[:, 0] >= 4) & (p[:, 0] < w - 4) & (p[:, 1] >= 4) & (p[:, 1] < h - 4)
        if error is not None:
            valid &= np.isfinite(error.ravel()) & (error.ravel() < 35)
        indices = np.flatnonzero(valid)
        if not len(indices):
            return empty, empty.copy(), indices
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, current[indices], None, **self.lk_parameters)
        if back is None or back_status is None:
            return empty, empty.copy(), no_indices
        fb = np.linalg.norm(previous_points[indices].reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        indices = indices[back_status.ravel().astype(bool) & np.isfinite(fb) & (fb <= 0.7)]
        return previous_points[indices].reshape(-1, 2), p[indices], indices

    def _normalize(self, points):
        return cv2.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2), self.K, self.distortion).reshape(-1, 2)

    @staticmethod
    def is_rotation_valid(rotation):
        return (np.isfinite(rotation).all() and np.linalg.norm(rotation @ rotation.T - np.eye(3)) < 1e-3
                and abs(np.linalg.det(rotation) - 1) < 1e-3)

    def _rotation_only(self, previous, current):
        a = np.column_stack((previous, np.ones(len(previous))))
        b = np.column_stack((current, np.ones(len(current))))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        keep = np.ones(len(a), bool)
        for _ in range(4):
            if keep.sum() < self.minimum_inliers:
                return None
            u, _, vt = np.linalg.svd(a[keep].T @ b[keep])
            rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
            projected = a @ rotation.T
            error = np.linalg.norm(projected[:, :2] / projected[:, 2:3] - current, axis=1) * self.focal
            keep = error < max(0.5, float(np.percentile(error, 70)))
        keep = error < 0.65
        if keep.mean() >= 0.85 and np.median(error[keep]) < 0.3:
            return rotation, np.zeros((3, 1)), keep, "ROTATION ONLY"
        return None

    def _pnp(self, current, landmarks):
        indices = np.flatnonzero(np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] > 0))
        if len(indices) < self.minimum_inliers:
            return None
        objects, points = np.ascontiguousarray(landmarks[indices]), np.ascontiguousarray(current[indices])
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(objects, points, np.eye(3), None, iterationsCount=150,
            reprojectionError=1.5 / self.focal, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < max(self.minimum_inliers, 0.5 * len(indices)):
            return None
        selected = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(objects[selected], points[selected], np.eye(3), None, rvec, tvec)
        rotation = cv2.Rodrigues(rvec)[0]
        projected = objects @ rotation.T + tvec.reshape(1, 3)
        error = np.linalg.norm(projected[:, :2] / projected[:, 2:3] - points, axis=1) * self.focal
        good = (projected[:, 2] > 0) & (error < 1.5)
        step = float(np.linalg.norm(tvec))
        if good.sum() < self.minimum_inliers or np.median(error[good]) > 0.8 or not np.isfinite(step) or step > max(5 * self.last_step, 0.1):
            return None
        keep = np.zeros(len(current), bool)
        keep[indices[good]] = True
        return rotation, tvec, keep, "TRACKING PNP"

    def _triangulate(self, previous, current, rotation, translation):
        count = len(previous)
        result = np.full((count, 3), np.nan)
        if not count or np.linalg.norm(translation) < 1e-9:
            return result
        homogeneous = cv2.triangulatePoints(np.column_stack((np.eye(3), np.zeros(3))),
            np.column_stack((rotation, translation.reshape(3))), previous.T, current.T)
        valid = np.abs(homogeneous[3]) > 1e-10
        xyz = np.full((count, 3), np.nan)
        xyz[valid] = (homogeneous[:3, valid] / homogeneous[3, valid]).T
        next_xyz = xyz @ rotation.T + translation.reshape(1, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            e0 = np.linalg.norm(xyz[:, :2] / xyz[:, 2:3] - previous, axis=1) * self.focal
            e1 = np.linalg.norm(next_xyz[:, :2] / next_xyz[:, 2:3] - current, axis=1) * self.focal
            a = np.column_stack((previous, np.ones(count)))
            b = np.column_stack((current, np.ones(count))) @ rotation
            cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        valid &= np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (next_xyz[:, 2] > 0)
        valid &= (e0 < 1.5) & (e1 < 1.5) & (parallax > 0.15)
        result[valid] = xyz[valid]
        return result

    def _essential(self, previous, current, landmarks, scale):
        essential, mask = cv2.findEssentialMat(previous, current, np.eye(3), method=self.ransac_method,
            prob=0.999, threshold=1.0 / self.focal, maxIters=2000)
        if essential is None or mask is None or np.count_nonzero(mask) < self.minimum_inliers:
            return None
        best = None
        for matrix in np.asarray(essential).reshape(-1, 3, 3):
            count, rotation, translation, pose_mask, _ = cv2.recoverPose(matrix, previous, current,
                np.eye(3), distanceThresh=10000.0, mask=mask.copy())
            if count < max(self.minimum_inliers, len(previous) * 0.35):
                continue
            unit_points = self._triangulate(previous, current, rotation, translation)
            keep = pose_mask.ravel().astype(bool) & np.isfinite(unit_points).all(axis=1)
            if keep.sum() < self.minimum_inliers:
                continue
            step, mode = scale, "TRACKING ESSENTIAL"
            if step is None:
                known = keep & np.isfinite(landmarks).all(axis=1) & (landmarks[:, 2] > 0)
                if known.sum() >= 8:
                    ratios = landmarks[known, 2] / unit_points[known, 2]
                    median = float(np.median(ratios))
                    consistent = np.abs(ratios - median) < median * 0.3
                    step = float(np.median(ratios[consistent])) if consistent.sum() >= 8 else None
                if step is None or not np.isfinite(step) or step <= 0:
                    if self.distance > 0:
                        continue
                    step, mode = self.last_step, "RELATIVE INITIALIZATION"
                if step > max(5 * self.last_step, 0.1):
                    continue
            candidate = rotation, translation * step, keep, mode
            if best is None or keep.sum() > best[2].sum():
                best = candidate
        return best

    def _set_reference(self, gray, points=None, landmarks=None):
        points = np.empty((0, 1, 2), np.float32) if points is None else points.reshape(-1, 1, 2).astype(np.float32)
        landmarks = np.full((len(points), 3), np.nan) if landmarks is None else landmarks
        if len(points) < self.minimum_features:
            extra = self.detect_features(gray, points)
            points = np.concatenate((points, extra))
            landmarks = np.concatenate((landmarks, np.full((len(extra), 3), np.nan)))
        self.previous_gray, self.previous_points, self.landmarks = gray, points, landmarks
        self.pending_scale, self.held_frames = 0.0, 0

    def update(self, frame, scale=None, timestamp=None):
        if scale is not None and (not np.isfinite(scale) or scale < 0):
            raise ValueError("Scale must be a nonnegative measured inter-frame distance")
        if self.metric_scale and scale is None:
            raise ValueError("Metric VO needs a distance measurement for every frame")
        if scale is not None and not self.metric_scale:
            if self.distance > 0:
                raise ValueError("Reset VO before switching from relative to metric units")
            self.metric_scale = True
        gray = np.ascontiguousarray(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")
        if self.last_timestamp is not None:
            gap = timestamp - self.last_timestamp
            if gap <= 0 or gap > self.max_gap or gray.shape != self.previous_gray.shape:
                self.reset_tracking()
                self.status = "TRACKING RESET"
        self.last_timestamp = timestamp
        self.tracked_inliers = 0
        if self.previous_gray is None:
            self._set_reference(gray)
            self.tracked_features = len(self.previous_points)
            return self.get_coordinates()
        if scale is not None:
            self.pending_scale += scale
        previous, current, indices = self.track_features(self.previous_gray, gray, self.previous_points)
        self.tracked_features = len(current)
        if len(current) < self.minimum_inliers:
            self.status = "INSUFFICIENT FEATURES"
            self._set_reference(gray)
            return self.get_coordinates()
        if np.median(np.linalg.norm(current - previous, axis=1)) < self.motion_threshold:
            self.status = "STATIONARY / LOW MOTION"
            self.bootstrap_votes = 0
            self.bootstrap_direction = None
            self.held_frames += 1
            if self.held_frames >= 15:
                self._set_reference(gray)
            return self.get_coordinates()
        p0, p1 = self._normalize(previous), self._normalize(current)
        old = self.landmarks[indices]
        measured_scale = self.pending_scale if self.metric_scale else None
        pose = None
        try:
            pose = self._pnp(p1, old)
            if pose is not None:
                depths = old[np.isfinite(old).all(axis=1), 2]
                if len(depths) and np.linalg.norm(pose[1]) * self.focal / np.median(depths) < 0.4:
                    rotation_pose = self._rotation_only(p0, p1)
                    if rotation_pose is not None:
                        pose = rotation_pose
            if pose is None:
                pose = self._rotation_only(p0, p1)
            if pose is None:
                pose = self._essential(p0, p1, old, measured_scale)
        except (cv2.error, np.linalg.LinAlgError):
            pose = None
        if pose is None or not self.is_rotation_valid(pose[0]):
            self.status = "POSE REJECTED"
            self._set_reference(gray, current)
            return self.get_coordinates()
        rotation, translation, keep, mode = pose
        if mode == "RELATIVE INITIALIZATION":
            direction = -(rotation.T @ translation).ravel()
            direction /= max(np.linalg.norm(direction), 1e-9)
            stable = self.bootstrap_direction is not None and np.dot(direction, self.bootstrap_direction) > 0.95
            self.bootstrap_votes = self.bootstrap_votes + 1 if stable else 1
            self.bootstrap_direction = direction
            cells = np.floor(current[keep] / np.array([gray.shape[1] / 6, gray.shape[0] / 3])).astype(int)
            coverage = len(np.unique(cells, axis=0))
            if self.bootstrap_votes < 3 or coverage < 6 or keep.mean() < 0.6:
                self.status = "VERIFYING MOTION"
                return self.get_coordinates()
        step = float(np.linalg.norm(translation))
        if measured_scale is not None and step > 1e-9:
            translation *= measured_scale / step
            step = measured_scale
        if measured_scale == 0:
            translation, step = np.zeros((3, 1)), 0.0
        if mode == "RELATIVE INITIALIZATION" and self.distance > 0:
            self.scale_restarts += 1
        self.translation += self.rotation @ (-(rotation.T @ translation))
        self.rotation = self.rotation @ rotation.T
        if step > 1e-9:
            self.last_step = step
            self.distance += step
        self.status = mode
        new_landmarks = old @ rotation.T + translation.reshape(1, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            error = np.linalg.norm(new_landmarks[:, :2] / new_landmarks[:, 2:3] - p1, axis=1) * self.focal
        reliable = np.isfinite(old).all(axis=1) & (new_landmarks[:, 2] > 0) & (error < 1.5)
        new_landmarks[~reliable] = np.nan
        triangulated = self._triangulate(p0, p1, rotation, translation)
        new = np.isfinite(triangulated).all(axis=1) & ~reliable
        new_landmarks[new] = triangulated[new] @ rotation.T + translation.reshape(1, 3)
        keep |= reliable | new
        self.tracked_inliers = int(keep.sum())
        self._set_reference(gray, current[keep], new_landmarks[keep])
        self.trajectory.append(self.get_coordinates())
        return self.get_coordinates()

    def get_coordinates(self):
        return self.translation.reshape(3).copy()

    def get_pose(self):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = self.rotation, self.get_coordinates()
        return pose


class PerceptionWorker:
    def __init__(self, engine, valid_mask, asynchronous=True, clearance=16):
        self.engine, self.valid_mask = engine, valid_mask
        self.asynchronous, self.clearance = asynchronous, clearance
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.running = True
        self.thread = None
        if asynchronous:
            self.thread = threading.Thread(target=self._run, name="perception-worker", daemon=True)
            self.thread.start()

    def _process(self, frame, packet):
        started = time.perf_counter()
        color, traversable = self.engine.predict(frame)
        traversable[self.valid_mask == 0] = 0
        path, status = plan_shortest_path(traversable, clearance=self.clearance)
        return {"frame": frame, "color": color, "traversable": traversable, "path": path, "status": status,
                "captured_at": packet.captured_at, "sequence": packet.sequence, "epoch": packet.epoch,
                "inference_ms": (time.perf_counter() - started) * 1000}

    def submit(self, frame, packet):
        if not self.asynchronous:
            self.result = self._process(frame, packet)
        else:
            with self.condition:
                self.pending = frame, packet
                self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or not self.running)
                if not self.running:
                    return
                frame, packet = self.pending
                self.pending = None
            try:
                result = self._process(frame, packet)
                with self.condition:
                    self.result = result
            except Exception as error:
                with self.condition:
                    self.error, self.running = error, False
                return

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Perception worker failed") from self.error
            return self.result

    def close(self):
        with self.condition:
            self.running, self.pending = False, None
            self.condition.notify()
        if self.thread:
            self.thread.join(timeout=10)


def draw_trajectory(vo, width=640, height=640):
    image = np.full((height, width, 3), (24, 22, 19), np.uint8)
    points = np.asarray(vo.trajectory)[:, [0, 2]]
    low, high = points.min(axis=0), points.max(axis=0)
    span = max(float(np.max(high - low)), 0.05)
    scale = min(width - 100, height - 150) / (span * 1.2)
    centre = (high + low) * 0.5
    pixels = np.column_stack(((points[:, 0] - centre[0]) * scale + width / 2,
                              height / 2 + 25 - (points[:, 1] - centre[1]) * scale)).astype(np.int32)
    for x in range(50, width - 30, 60):
        cv2.line(image, (x, 85), (x, height - 50), (46, 43, 37), 1)
    for y in range(100, height - 40, 60):
        cv2.line(image, (40, y), (width - 40, y), (46, 43, 37), 1)
    if len(pixels) > 1:
        cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, (85, 225, 125), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(pixels[0]), 5, (255, 190, 60), -1)
    cv2.circle(image, tuple(pixels[-1]), 7, (80, 200, 255), -1)
    heading = vo.rotation[:, 2][[0, 2]].copy()
    heading /= max(float(np.linalg.norm(heading)), 1e-9)
    end = pixels[-1] + np.array([heading[0], -heading[1]]) * 25
    cv2.arrowedLine(image, tuple(pixels[-1]), tuple(end.astype(int)), (80, 200, 255), 2, tipLength=0.4)
    labels = ["VISUAL ODOMETRY", f"{vo.status} | {vo.units}",
              f"Distance: {vo.distance:.2f} | scale resets: {vo.scale_restarts}"]
    for index, label in enumerate(labels):
        cv2.putText(image, label, (20, 27 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 230, 235), 1, cv2.LINE_AA)
    cv2.line(image, (40, height - 28), (140, height - 28), (210, 210, 210), 2)
    cv2.putText(image, f"{100 / scale:.3g} {vo.units}", (150, height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    return image


def create_views(frame, result, vo, fps, status, age_ms):
    h, w = frame.shape[:2]
    raw = frame.copy()
    seg = frame.copy() if result is None else cv2.addWeighted(result["frame"], 0.55, result["color"], 0.45, 0)
    astar = np.zeros_like(frame) if result is None else cv2.cvtColor(result["traversable"], cv2.COLOR_GRAY2BGR)
    start = (w // 2, h - 5)
    target = (w // 2, int(h * 0.15))
    cv2.circle(astar, start, 8, (0, 220, 255), -1)
    cv2.drawMarker(astar, target, (0, 80, 255), cv2.MARKER_CROSS, 20, 2)
    if result is not None:
        draw_planned_path(astar, result["path"])
    planner_status = result['status'] if result is not None else 'WAITING'
    labels = [(raw, f"CAMERA | {fps:.1f} FPS"), (seg, "SEGMENTATION"),
              (astar, f"A*: {planner_status} | age {age_ms:.0f} ms")]
    for panel, text in labels:
        cv2.rectangle(panel, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(panel, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    if result is not None and not result["path"]:
        cv2.putText(astar, "No connected free route from the rover start", (12, max(70, h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, cv2.LINE_AA)
    cv2.putText(astar, f"Control: {status}", (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 160, 255), 1, cv2.LINE_AA)
    trajectory = draw_trajectory(vo, max(480, w), max(480, h))
    tile = cv2.resize(trajectory, (w, h), interpolation=cv2.INTER_AREA)
    combined = np.vstack((np.hstack((raw, seg)), np.hstack((tile, astar))))
    return {"all": combined, "vo": trajectory, "segmentation": seg, "astar": astar}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ANVESHAK segmentation, visual odometry, A* and dashboard")
    parser.add_argument("--source", default=os.environ.get("STREAM_URL", STREAM_URL))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_FILE)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--seg-every", type=int, default=4)
    parser.add_argument("--seg-scale", type=float, default=1.0)
    parser.add_argument("--seg-confidence", type=float, default=0.65)
    parser.add_argument("--floor-fraction", type=float, default=0.45,
                        help="Bottom fraction where sky becomes traversable; 0 disables the override")
    parser.add_argument("--seg-tta", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--clearance", type=int, default=16)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--show-local-window", action="store_true", default=os.environ.get("SHOW_LOCAL_WINDOW", "0") == "1")
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8080")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scale-csv", type=Path)
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args(argv)
    if args.seg_every < 1 or args.threads < 1 or args.max_frames < 0 or args.clearance < 0:
        parser.error("Invalid frame interval, thread count, frame limit, or clearance")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    calibration = CameraCalibration(args.width, args.height, args.calibration)
    source = FrameSource(args.source)
    if args.scale_csv and source.live:
        raise ValueError("--scale-csv requires prerecorded video; use update(scale=...) for live measurements")
    scales = ScaleSamples(args.scale_csv) if args.scale_csv else None
    vo = FrameVisualOdometry(intrinsic_matrix=calibration.K, max_features=args.max_features, metric_scale=scales is not None)
    model = load_segmentation_model(MODEL_FILE, args.weights, device)
    engine = SegmentationEngine(model, device, args.seg_confidence, args.seg_scale, args.seg_tta, args.fp16, args.floor_fraction)
    dashboard = worker = output_handle = writer = None
    frame_number, sequence, epoch = 0, -1, None
    intervals = deque(maxlen=60)
    last_frame_at, last_render_at = time.monotonic(), 0.0
    session_revision = 0
    was_active = False
    playback_started = time.monotonic()
    started = time.perf_counter()
    print(f"Device: {device} | Camera: {'calibrated' if calibration.calibrated else 'approximate intrinsics'} | VO: {vo.units}")
    print(f"Sky-to-traversable override: bottom {args.floor_fraction:.0%} of the frame")
    try:
        if not args.no_dashboard:
            dashboard = DashboardServer(args.host, args.port, dry_run=args.dry_run or not source.live)
            dashboard.start()
            print(f"Dashboard: http://localhost:{dashboard.port}")
        worker = PerceptionWorker(engine, calibration.valid_mask, source.live, args.clearance)
        if args.trajectory_csv:
            args.trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
            output_handle = args.trajectory_csv.open("w", newline="")
            writer = csv.writer(output_handle)
            writer.writerow(["frame", "timestamp_s", "x", "y", "z", "units", "vo_status", "inliers"])
        source.start()
        if args.show_local_window:
            cv2.namedWindow("ANVESHAK", cv2.WINDOW_NORMAL)
        print("Running. Press Ctrl+C to stop.")
        while not args.max_frames or frame_number < args.max_frames:
            active = True
            if dashboard:
                pending = dashboard.take_source()
                if pending is not None:
                    source.close()
                    worker.close()
                    selected = args.source if pending[0] is None else pending[0]
                    source = FrameSource(selected)
                    source.start()
                    scales = ScaleSamples(args.scale_csv) if pending[0] is None and args.scale_csv else None
                    worker = PerceptionWorker(engine, calibration.valid_mask, source.live, args.clearance)
                    sequence, epoch = -1, None
                    frame_number = 0
                    playback_started = time.monotonic()
                    with dashboard.controller.lock:
                        dashboard.controller.dry_run = True
                revision, active = dashboard.session()
                if revision != session_revision:
                    vo = FrameVisualOdometry(intrinsic_matrix=calibration.K, max_features=args.max_features, metric_scale=scales is not None)
                    session_revision = revision
                    dashboard.state.update_telemetry({"position": [0, 0, 0], "distance": 0, "vo_status": "INITIALIZING"})
                    if not source.live:
                        source.close()
                        source = FrameSource(source.source)
                        source.start()
                        sequence, epoch = -1, None
                        worker.close()
                        worker = PerceptionWorker(engine, calibration.valid_mask, False, args.clearance)
                        if scales:
                            scales.previous_distance = None
                        playback_started = time.monotonic()
                if active and not was_active:
                    vo.reset_tracking()
                if not active:
                    dashboard.controller.stop_motion()
                    vo.status = "WAITING FOR WEBSITE"
                    if not source.live:
                        time.sleep(0.03)
                        continue
                was_active = active
            packet = source.read(sequence)
            now = time.monotonic()
            if packet is None:
                worker.latest()
                if dashboard and (not source.connected or now - last_frame_at > 0.5):
                    dashboard.camera_disconnected()
                if source.ended:
                    if dashboard and not args.max_frames:
                        dashboard.state.update_telemetry({"planner_status": "VIDEO FINISHED", "source_status": "Upload another video or return to camera"})
                        time.sleep(0.03)
                        continue
                    break
                if args.show_local_window and cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                    break
                continue
            if not source.live:
                wait = playback_started + packet.timestamp - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, 0.1))
            sequence = packet.sequence
            if epoch != packet.epoch:
                vo.reset_tracking()
                epoch = packet.epoch
                if dashboard:
                    dashboard.controller.stop_motion()
            frame = calibration.prepare(packet.frame)
            position = vo.update(frame, scale=scales.step(packet.timestamp) if scales else None, timestamp=packet.timestamp) if active else vo.get_coordinates()
            if frame_number % args.seg_every == 0:
                worker.submit(frame, packet)
            result = worker.latest()
            if result is not None and result["epoch"] != epoch:
                result = None
            now = time.monotonic()
            age = now - result["captured_at"] if result is not None else float("inf")
            fresh_camera = not source.live or source.connected and now - packet.captured_at <= 0.5
            usable = result is not None and age <= 0.8 and fresh_camera
            path = result["path"] if usable else []
            status = result["status"] if usable else "WAITING" if result is None else "STALE PERCEPTION"
            if dashboard:
                if not fresh_camera:
                    dashboard.camera_disconnected()
                else:
                    dashboard.controller.update_autonomous(path if active else [], args.width, result["captured_at"] if usable and active else None)
            intervals.append(max(now - last_frame_at, 1e-6))
            last_frame_at = now
            fps = len(intervals) / sum(intervals)
            frame_number += 1
            if writer:
                writer.writerow([sequence, packet.timestamp, *position.tolist(), vo.units, vo.status, vo.tracked_inliers])
                if frame_number % 30 == 0:
                    output_handle.flush()
            telemetry = {"camera_connected": bool(fresh_camera), "planner_status": status, "fps": fps, "tracking_active": active,
                "source_kind": "camera" if source.live else "test video",
                "tracked_features": vo.tracked_features, "tracked_inliers": vo.tracked_inliers,
                "position": position.tolist(), "vo_status": vo.status, "vo_units": vo.units,
                "distance": vo.distance, "scale_restarts": vo.scale_restarts, "calibrated": calibration.calibrated,
                "path_points": len(path), "frame_number": frame_number, "floor_fraction": args.floor_fraction,
                "perception_age_ms": age * 1000 if math.isfinite(age) else None,
                "inference_ms": result["inference_ms"] if result else None}
            if dashboard:
                dashboard.state.update_telemetry(telemetry)
            if (dashboard or args.show_local_window) and now - last_render_at >= 0.1:
                views = create_views(frame, result, vo, fps, status, age * 1000 if math.isfinite(age) else 0)
                if dashboard:
                    dashboard.state.publish_views(views, telemetry)
                if args.show_local_window:
                    cv2.imshow("ANVESHAK", views["all"])
                last_render_at = now
            if args.show_local_window and cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if dashboard:
            dashboard.stop()
        source.close()
        if worker:
            worker.close()
        if output_handle:
            output_handle.close()
        if args.show_local_window:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    metrics = {"frames": frame_number, "elapsed_s": elapsed, "processing_fps": frame_number / max(elapsed, 1e-9),
        "position": vo.get_coordinates().tolist(), "distance": vo.distance, "units": vo.units,
        "scale_restarts": vo.scale_restarts, "vo_status": vo.status, "calibrated": calibration.calibrated,
        "floor_fraction": args.floor_fraction, "device": str(device)}
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Processed {frame_number} frames | {metrics['processing_fps']:.1f} FPS | distance {vo.distance:.3f} {vo.units}")
    return metrics


if __name__ == "__main__":
    main()
