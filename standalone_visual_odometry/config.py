from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

@dataclass
class DatasetConfig:
    base_dir: Path = Path("data/kitti/dataset")
    sequence: str = "00"
    camera_dir: str = "image_0"

@dataclass
class DetectorConfig:
    detector_type: str = "FAST"
    fast_threshold: int = 20
    fast_nonmax_suppression: bool = True
    use_grid_bucketing: bool = True
    grid_rows: int = 3
    grid_cols: int = 6
    max_features_per_cell: int = 100
    quality_level: float = 0.01
    min_distance: float = 8.0
    target_num_features: int = 1800
    min_num_features: int = 1200
    mask_vehicle_hood: bool = True
    hood_bottom_fraction: float = 0.16
    mask_sky: bool = True
    sky_top_fraction: float = 0.18

@dataclass
class TrackerConfig:
    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    max_iters: int = 30
    epsilon: float = 0.01
    enable_fb_check: bool = True
    max_fb_error: float = 0.7

@dataclass
class PoseConfig:
    estimator: str = "USAC_MAGSAC"
    ransac_threshold: float = 0.3
    ransac_prob: float = 0.99999
    min_inliers: int = 20
    min_stationary_disp: float = 1.2
    default_scale: float = 1.0

@dataclass
class GUIConfig:
    window_title: str = "UGV Visual Odometry - Trajectory Tracker"
    canvas_width: int = 1280
    canvas_height: int = 720
    target_fps: int = 30
    map_padding_meters: float = 25.0
    show_flow_lines: bool = False
    show_feature_points: bool = False
    color_bg: Tuple[int, int, int] = (18, 20, 24)
    color_panel: Tuple[int, int, int] = (26, 29, 36)
    color_accent: Tuple[int, int, int] = (255, 191, 0)
    color_est_path: Tuple[int, int, int] = (80, 245, 120)
    color_gt_path: Tuple[int, int, int] = (210, 210, 210)
    color_ugv: Tuple[int, int, int] = (0, 140, 255)
    color_text: Tuple[int, int, int] = (245, 245, 245)
    color_grid: Tuple[int, int, int] = (40, 45, 55)

@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)

default_config = AppConfig()
