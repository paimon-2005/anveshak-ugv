"""Modern Material-style OpenCV dashboard for visual odometry."""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, default_config
from src.utils import FPSCounter, compute_trajectory_length


class VODashboard:
    """Modern visual-odometry dashboard with camera, trajectory and telemetry."""

    # Material UI-inspired palette, stored as OpenCV BGR tuples.
    MUI_BG = (246, 247, 249)          # #F6F7F9
    MUI_SURFACE = (255, 255, 255)     # #FFFFFF
    MUI_SURFACE_ALT = (247, 249, 252)
    MUI_BORDER = (224, 229, 236)
    MUI_TEXT = (38, 43, 51)           # #262B33
    MUI_MUTED = (103, 113, 128)
    MUI_PRIMARY = (210, 118, 25)      # #1976D2
    MUI_PRIMARY_DARK = (156, 84, 18)  # #1257A0
    MUI_SUCCESS = (50, 125, 46)       # #2E7D32
    MUI_WARNING = (2, 108, 237)        # #ED6C02
    MUI_ERROR = (47, 47, 211)         # #D32F2F
    MUI_INFO = (214, 82, 2)            # #0288D1
    MUI_GRID = (232, 235, 240)
    MUI_EST = (50, 125, 46)
    MUI_GT = (120, 125, 135)
    MUI_UGV = (210, 118, 25)
    MUI_CAMERA_BG = (31, 35, 42)

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or default_config
        self.fps_counter = FPSCounter()
        self.w = self.config.gui.canvas_width
        self.h = self.config.gui.canvas_height
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        self.min_x = -40.0
        self.max_x = 40.0
        self.min_z = -20.0
        self.max_z = 80.0

        cv2.namedWindow(self.config.gui.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.gui.window_title, self.w, self.h)

    @staticmethod
    def _text(img, text, pos, size=0.5, color=MUI_TEXT, thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _rounded_rect(img, p1, p2, radius, fill, border=None, thickness=1):
        x1, y1 = p1
        x2, y2 = p2
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), fill, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), fill, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, fill, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, fill, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, fill, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, fill, -1)
        if border is not None:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), border, thickness)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), border, thickness)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border, thickness)

    def _draw_header(self, frame_idx, total_frames, fps, is_paused, is_completed):
        header_h = 72
        cv2.rectangle(self.canvas, (0, 0), (self.w, header_h), self.MUI_SURFACE, -1)
        cv2.line(self.canvas, (0, header_h - 1), (self.w, header_h - 1), self.MUI_BORDER, 1)

        # App mark
        cv2.circle(self.canvas, (28, 36), 17, self.MUI_PRIMARY, -1)
        self._text(self.canvas, "VO", (15, 42), 0.43, self.MUI_SURFACE, 1)
        self._text(self.canvas, "Visual Odometry", (55, 33), 0.62, self.MUI_TEXT, 1)
        self._text(self.canvas, "Trajectory & camera telemetry", (55, 53), 0.38, self.MUI_MUTED, 1)

        # Status chip
        if is_completed:
            label, color = "COMPLETED", self.MUI_SUCCESS
        elif is_paused:
            label, color = "PAUSED", self.MUI_WARNING
        else:
            label, color = "TRACKING", self.MUI_SUCCESS
        chip_w = 112 if label != "COMPLETED" else 122
        bx = self.w - 255
        self._rounded_rect(self.canvas, (bx, 20), (bx + chip_w, 51), 15, color)
        self._text(self.canvas, "●", (bx + 11, 41), 0.42, self.MUI_SURFACE, 1)
        self._text(self.canvas, label, (bx + 28, 41), 0.39, self.MUI_SURFACE, 1)

        frame_str = f"FRAME  {frame_idx + 1:04d} / {total_frames:04d}" if total_frames else f"FRAME  {frame_idx + 1:04d}"
        self._text(self.canvas, frame_str, (self.w - 128, 34), 0.36, self.MUI_MUTED, 1)
        self._text(self.canvas, f"{fps:.1f} FPS", (self.w - 128, 52), 0.36, self.MUI_TEXT, 1)

        if total_frames > 0:
            fill_w = int(self.w * min(1.0, (frame_idx + 1) / total_frames))
            cv2.rectangle(self.canvas, (0, header_h - 3), (fill_w, header_h), self.MUI_PRIMARY, -1)

    def _draw_metric_card(self, x, y, w, h, title, value, subtitle, accent):
        self._rounded_rect(self.canvas, (x, y), (x + w, y + h), 10, self.MUI_SURFACE, self.MUI_BORDER, 1)
        cv2.rectangle(self.canvas, (x, y + 8), (x + 4, y + h - 8), accent, -1)
        self._text(self.canvas, title.upper(), (x + 16, y + 22), 0.32, self.MUI_MUTED, 1)
        self._text(self.canvas, value, (x + 16, y + 48), 0.62, self.MUI_TEXT, 1)
        self._text(self.canvas, subtitle, (x + 16, y + h - 12), 0.31, self.MUI_MUTED, 1)

    def _draw_camera(self, image, step_result, x0, y0, w, h):
        if len(image.shape) == 2:
            cam = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            cam = image.copy()
        cam = cv2.resize(cam, (w, h))

        pts_prev = step_result.get("tracked_prev", np.empty((0, 2)))
        pts_curr = step_result.get("tracked_curr", np.empty((0, 2)))
        inliers = step_result.get("inliers_mask", np.empty((0,), dtype=bool))
        sx, sy = w / image.shape[1], h / image.shape[0]

        if self.config.gui.show_feature_points:
            for i, p in enumerate(pts_curr):
                cx, cy = int(p[0] * sx), int(p[1] * sy)
                ok = inliers[i] if i < len(inliers) else True
                color = self.MUI_SUCCESS if ok else self.MUI_ERROR
                if self.config.gui.show_flow_lines and i < len(pts_prev):
                    px, py = int(pts_prev[i][0] * sx), int(pts_prev[i][1] * sy)
                    cv2.line(cam, (px, py), (cx, cy), color, 1, cv2.LINE_AA)
                cv2.circle(cam, (cx, cy), 2, color, -1)

        self._rounded_rect(self.canvas, (x0, y0), (x0 + w, y0 + h), 12, self.MUI_CAMERA_BG, self.MUI_BORDER, 1)
        self.canvas[y0:y0 + h, x0:x0 + w] = cam

        # Camera overlay
        overlay = self.canvas[y0:y0 + 54, x0:x0 + w]
        overlay[:] = cv2.addWeighted(overlay, 0.35, np.full_like(overlay, self.MUI_CAMERA_BG), 0.65, 0)
        self._text(self.canvas, "LIVE CAMERA", (x0 + 16, y0 + 23), 0.40, self.MUI_SURFACE, 1)
        self._text(self.canvas, "●", (x0 + 16, y0 + 42), 0.42, self.MUI_SUCCESS, 1)
        self._text(self.canvas, "Optical flow tracking", (x0 + 31, y0 + 42), 0.34, (225, 230, 238), 1)

        n_feats = step_result.get("num_features", 0)
        n_inliers = step_result.get("num_inliers", 0)
        stats = f"{n_inliers}/{n_feats} inliers"
        self._text(self.canvas, stats, (x0 + w - 112, y0 + 27), 0.34, (225, 230, 238), 1)

    def _draw_trajectory_map(self, est_trajectory, gt_trajectory, cur_pos, gt_pos, cur_rot, x0, y0, w, h):
        self._rounded_rect(self.canvas, (x0, y0), (x0 + w, y0 + h), 12, self.MUI_SURFACE, self.MUI_BORDER, 1)
        inner_x, inner_y = x0 + 1, y0 + 1
        inner_w, inner_h = w - 2, h - 2

        if not est_trajectory:
            self._text(self.canvas, "Waiting for trajectory data...", (x0 + 24, y0 + 60), 0.45, self.MUI_MUTED, 1)
            return

        est_arr = np.array(est_trajectory)
        all_x, all_z = list(est_arr[:, 0]), list(est_arr[:, 2])
        if gt_trajectory:
            gt_arr = np.array(gt_trajectory)
            all_x.extend(gt_arr[:, 0])
            all_z.extend(gt_arr[:, 2])

        pad = self.config.gui.map_padding_meters
        min_x, max_x = min(min(all_x) - pad, self.min_x), max(max(all_x) + pad, self.max_x)
        min_z, max_z = min(min(all_z) - pad, self.min_z), max(max(all_z) + pad, self.max_z)
        self.min_x, self.max_x, self.min_z, self.max_z = min_x, max_x, min_z, max_z

        span_x, span_z = max(max_x - min_x, 10.0), max(max_z - min_z, 10.0)
        scale = min((inner_w - 44) / span_x, (inner_h - 82) / span_z)
        center_x, center_z = (min_x + max_x) / 2, (min_z + max_z) / 2
        origin_u = x0 + w // 2 - int(center_x * scale)
        origin_v = y0 + h // 2 - int(center_z * scale)  # forward is -Z -> drawn up

        def pixel(xm, zm):
            return int(origin_u + xm * scale), int(origin_v + zm * scale)

        grid_step = 20.0 if span_z < 60 else 50.0 if span_z <= 500 else 100.0
        z = math.floor(min_z / grid_step) * grid_step
        while z <= max_z:
            _, v = pixel(min_x, z)
            if y0 + 45 <= v <= y0 + h:
                cv2.line(self.canvas, (x0 + 8, v), (x0 + w - 8, v), self.MUI_GRID, 1)
            z += grid_step
        x = math.floor(min_x / grid_step) * grid_step
        while x <= max_x:
            u, _ = pixel(x, min_z)
            if x0 <= u <= x0 + w:
                cv2.line(self.canvas, (u, y0 + 42), (u, y0 + h - 8), self.MUI_GRID, 1)
            x += grid_step

        # Paths
        if gt_trajectory and len(gt_trajectory) > 1:
            pts = [pixel(p[0], p[2]) for p in gt_trajectory]
            cv2.polylines(self.canvas, [np.array(pts, dtype=np.int32)], False, self.MUI_GT, 2, cv2.LINE_AA)
        if len(est_trajectory) > 1:
            pts = [pixel(p[0], p[2]) for p in est_trajectory]
            cv2.polylines(self.canvas, [np.array(pts, dtype=np.int32)], False, self.MUI_EST, 3, cv2.LINE_AA)

        start = pixel(0, 0)
        cv2.circle(self.canvas, start, 5, self.MUI_SUCCESS, -1)
        self._text(self.canvas, "START", (start[0] + 8, start[1] + 4), 0.31, self.MUI_SUCCESS, 1)

        cur_u, cur_v = pixel(cur_pos[0], cur_pos[2])
        fwd = cur_rot @ np.array([0.0, 0.0, 1.0])
        norm = math.hypot(float(fwd[0]), float(fwd[2]))
        if norm > 1e-6:
            tip = (int(cur_u + fwd[0] / norm * 24), int(cur_v + fwd[2] / norm * 24))
            cv2.arrowedLine(self.canvas, (cur_u, cur_v), tip, self.MUI_UGV, 3, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(self.canvas, (cur_u, cur_v), 8, self.MUI_SURFACE, -1)
        cv2.circle(self.canvas, (cur_u, cur_v), 6, self.MUI_UGV, -1)

        # Map header
        self._text(self.canvas, "TRAJECTORY MAP", (x0 + 16, y0 + 25), 0.40, self.MUI_TEXT, 1)
        self._text(self.canvas, "TOP-DOWN", (x0 + 16, y0 + 42), 0.30, self.MUI_MUTED, 1)

        lx = x0 + w - 205
        cv2.line(self.canvas, (lx, y0 + 19), (lx + 18, y0 + 19), self.MUI_EST, 3)
        self._text(self.canvas, "Estimated", (lx + 24, y0 + 23), 0.31, self.MUI_TEXT, 1)
        cv2.line(self.canvas, (lx + 92, y0 + 19), (lx + 110, y0 + 19), self.MUI_GT, 2)
        self._text(self.canvas, "Actual", (lx + 116, y0 + 23), 0.31, self.MUI_TEXT, 1)

        dist = compute_trajectory_length(est_trajectory)
        card_w, card_h = 236, 66
        bx, by = x0 + w - card_w - 14, y0 + h - card_h - 14
        self._rounded_rect(self.canvas, (bx, by), (bx + card_w, by + card_h), 10, self.MUI_SURFACE, self.MUI_BORDER, 1)
        self._text(self.canvas, "CURRENT POSITION", (bx + 12, by + 19), 0.29, self.MUI_MUTED, 1)
        self._text(self.canvas, f"X  {cur_pos[0]:+.1f} m    Z  {cur_pos[2]:+.1f} m", (bx + 12, by + 40), 0.40, self.MUI_TEXT, 1)
        self._text(self.canvas, f"Travel  {dist:.1f} m", (bx + 12, by + 56), 0.30, self.MUI_PRIMARY_DARK, 1)

    def _draw_footer(self, y, is_paused, is_completed):
        self._text(self.canvas, "SPACE  Pause", (18, y), 0.32, self.MUI_MUTED, 1)
        self._text(self.canvas, "D  Step", (105, y), 0.32, self.MUI_MUTED, 1)
        self._text(self.canvas, "R  Reset", (177, y), 0.32, self.MUI_MUTED, 1)
        self._text(self.canvas, "+ / -  Speed", (254, y), 0.32, self.MUI_MUTED, 1)
        self._text(self.canvas, "Q / ESC  Close", (350, y), 0.32, self.MUI_MUTED, 1)
        state = "Playback complete — press R to restart" if is_completed else "Playback paused" if is_paused else "Tracking in progress"
        self._text(self.canvas, state, (self.w - 270, y), 0.32, self.MUI_PRIMARY_DARK if not is_completed else self.MUI_SUCCESS, 1)

    def render(self, image: np.ndarray, step_result: dict, est_trajectory: List[np.ndarray], gt_trajectory: Optional[List[np.ndarray]], gt_pos: Optional[np.ndarray], frame_idx: int, total_frames: int, is_paused: bool = False, is_completed: bool = False) -> int:
        fps = self.fps_counter.tick()
        self.canvas[:] = self.MUI_BG

        self._draw_header(frame_idx, total_frames, fps, is_paused, is_completed)

        content_y = 88
        gap = 12
        card_h = 74
        cards_y = content_y
        card_w = (self.w - 36 - gap * 3) // 4
        cur_pos = step_result.get("position", np.zeros(3))
        n_features = step_result.get("num_features", 0)
        n_inliers = step_result.get("num_inliers", 0)
        distance = compute_trajectory_length(est_trajectory)

        self._draw_metric_card(12, cards_y, card_w, card_h, "Position", f"{cur_pos[0]:+.1f} m", f"X coordinate   Z {cur_pos[2]:+.1f} m", self.MUI_PRIMARY)
        self._draw_metric_card(12 + card_w + gap, cards_y, card_w, card_h, "Features", str(n_features), f"{n_inliers} robust inliers", self.MUI_INFO)
        self._draw_metric_card(12 + (card_w + gap) * 2, cards_y, card_w, card_h, "Travel", f"{distance:.1f} m", "Estimated trajectory length", self.MUI_SUCCESS)
        self._draw_metric_card(12 + (card_w + gap) * 3, cards_y, card_w, card_h, "Rate", f"{fps:.1f} FPS", "Live processing rate", self.MUI_WARNING)

        panel_y = cards_y + card_h + 12
        footer_h = 28
        panel_h = self.h - panel_y - footer_h - 12
        cam_w = int((self.w - 36) * 0.47)
        map_w = self.w - cam_w - 36

        self._draw_camera(image, step_result, 12, panel_y, cam_w, panel_h)
        self._draw_trajectory_map(est_trajectory, gt_trajectory, cur_pos, gt_pos, step_result.get("rotation", np.eye(3)), 24 + cam_w, panel_y, map_w, panel_h)
        self._draw_footer(self.h - 12, is_paused, is_completed)

        cv2.imshow(self.config.gui.window_title, self.canvas)
        wait_ms = 30 if is_completed else 0 if is_paused else max(1, int(1000.0 / self.config.gui.target_fps))
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self):
        """Destroy OpenCV windows and cleanup."""
        cv2.destroyAllWindows()
