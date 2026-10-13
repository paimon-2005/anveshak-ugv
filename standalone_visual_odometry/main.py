import argparse
import sys
import time
from pathlib import Path

from config import default_config
from src.dataset import KittiDataset, VideoDataset
from src.gui import VODashboard
from src.utils import compute_ate_rmse, compute_trajectory_length
from src.visual_odometry import VisualOdometry
from tools.generate_sample_data import generate_sample_kitti_sequence


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight visual odometry for UGV with a real-time OpenCV GUI")
    parser.add_argument("-s", "--sequence", type=str, default="00")
    parser.add_argument("-v", "--video", type=str)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("-d", "--data_dir", type=str, default="data/kitti/dataset")
    parser.add_argument("--camera", type=str, default="image_0", choices=["image_0", "image_1", "image_2", "image_3"])
    parser.add_argument("--detector", type=str, default="FAST", choices=["FAST", "SHI_TOMASI"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--no_gui", action="store_true")
    return parser.parse_args(argv)


def run(video_path=None, argv=None):
    args = parse_args(argv)
    if video_path is not None:
        args.video = str(video_path)
    data_dir = Path(args.data_dir)
    if args.demo and args.video is None:
        demo_path = Path(__file__).resolve().parent / "demo" / "classroom_demo.mp4"
        if demo_path.exists():
            args.video = str(demo_path)
        else:
            print(f"[*] Demo video not found at {demo_path}; falling back to synthetic sequence 99")
            args.sequence = "99"

    if args.video is not None:
        sequence_name = "VIDEO"
    else:
        sequence_name = str(args.sequence).zfill(2)
        sequence_directory = data_dir / "sequences" / sequence_name
        if not sequence_directory.exists():
            seq99_dir = data_dir / "sequences" / "99"
            if sequence_name == "00" and seq99_dir.exists():
                print("[*] Sequence 00 not found; using available sequence 99...")
                sequence_name = "99"
                args.sequence = "99"
            elif sequence_name in ("00", "99") or not (data_dir / "sequences").exists():
                print("[*] Generating synthetic test sequence 99...")
                generate_sample_kitti_sequence(output_base=data_dir, seq_id="99", num_frames=80)
                sequence_name = "99"
                args.sequence = "99"
            else:
                print(f"[x] Error: Sequence directory does not exist: {sequence_directory}")
                return 1

    config = default_config
    config.dataset.base_dir = data_dir
    config.dataset.sequence = sequence_name
    config.dataset.camera_dir = args.camera
    config.detector.detector_type = args.detector
    config.gui.target_fps = args.fps
    print("\n=======================================================")
    print("       UGV REAL-TIME VISUAL ODOMETRY SYSTEM")
    print("=======================================================")
    print(f"[*] Sequence:       {config.dataset.sequence}")
    print(f"[*] Camera:         {config.dataset.camera_dir}")
    print(f"[*] Feature Engine: {config.detector.detector_type}")
    print(f"[*] Target FPS:     {config.gui.target_fps}")
    print("=======================================================\n")

    try:
        dataset = VideoDataset(args.video) if args.video is not None else KittiDataset(
            base_dir=config.dataset.base_dir,
            sequence=config.dataset.sequence,
            camera_dir=config.dataset.camera_dir,
        )
    except Exception as error:
        print(f"[x] Failed to load input: {error}")
        return 1

    total_frames = len(dataset)
    has_ground_truth = dataset.ground_truth_poses is not None
    source_description = "live stream" if getattr(dataset, "live", False) else f"{total_frames} frames"
    print(f"[+] Input loaded: {source_description}")
    print(f"[+] Ground truth: {'available' if has_ground_truth else 'not available'}")
    print(f"[+] Camera Matrix K:\n{dataset.K}\n")
    vo = VisualOdometry(intrinsic_matrix=dataset.K, config=config)
    dashboard = None if args.no_gui else VODashboard(config=config)
    ground_truth_trajectory = []
    is_paused = False
    step_once = False
    frame_index = 0
    processed_frames = 0
    image = None
    ground_truth_position = None
    step_result = {}
    started = time.perf_counter()

    try:
        while total_frames <= 0 or frame_index < total_frames:
            advance_frame = not is_paused or step_once
            if advance_frame:
                try:
                    image, ground_truth_pose = dataset[frame_index]
                except IOError as error:
                    print(f"[*] Input ended: {error}")
                    break
                scale = args.scale if args.scale is not None else dataset.get_ground_truth_scale(frame_index)
                step_result = vo.step(curr_image=image, scale=scale)
                ground_truth_position = ground_truth_pose[:3, 3] if ground_truth_pose is not None else None
                if ground_truth_position is not None:
                    ground_truth_trajectory.append(ground_truth_position.copy())
                processed_frames += 1
                step_once = False

            if dashboard is not None:
                key = dashboard.render(
                    image=image,
                    step_result=step_result,
                    est_trajectory=vo.trajectory,
                    gt_trajectory=ground_truth_trajectory if has_ground_truth else None,
                    gt_pos=ground_truth_position,
                    frame_idx=frame_index,
                    total_frames=total_frames,
                    is_paused=is_paused,
                )
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    is_paused = not is_paused
                elif key in (ord("d"), ord("D"), 83):
                    step_once = True
                elif key in (ord("+"), ord("=")):
                    config.gui.target_fps = min(120, config.gui.target_fps + 10)
                elif key in (ord("-"), ord("_")):
                    config.gui.target_fps = max(5, config.gui.target_fps - 5)
                elif key in (ord("r"), ord("R")):
                    vo = VisualOdometry(intrinsic_matrix=dataset.K, config=config)
                    ground_truth_trajectory.clear()
                    frame_index = 0
                    continue

            if advance_frame:
                frame_index += 1

            if total_frames > 0 and frame_index >= total_frames:
                if dashboard is None:
                    break
                restart = False
                while True:
                    key = dashboard.render(
                        image=image,
                        step_result=step_result,
                        est_trajectory=vo.trajectory,
                        gt_trajectory=ground_truth_trajectory if has_ground_truth else None,
                        gt_pos=ground_truth_position,
                        frame_idx=total_frames - 1,
                        total_frames=total_frames,
                        is_paused=True,
                        is_completed=True,
                    )
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("r"), ord("R")):
                        vo = VisualOdometry(intrinsic_matrix=dataset.K, config=config)
                        ground_truth_trajectory.clear()
                        frame_index = 0
                        restart = True
                        break
                if not restart:
                    break
    except KeyboardInterrupt:
        print("[*] Interrupted by user")
    finally:
        if dashboard is not None:
            dashboard.close()
        dataset.release()

    total_time = time.perf_counter() - started
    average_fps = processed_frames / total_time if total_time > 0 else 0.0
    estimated_distance = compute_trajectory_length(vo.trajectory)
    print("\n=======================================================")
    print("                FINAL EXECUTION METRICS")
    print("=======================================================")
    print(f"[*] Frames processed: {processed_frames}")
    print(f"[*] Total execution time: {total_time:.2f} s")
    print(f"[*] Average computation: {average_fps:.1f} FPS")
    print(f"[*] Total distance: {estimated_distance:.2f}")
    if has_ground_truth and ground_truth_trajectory:
        absolute_error = compute_ate_rmse(vo.trajectory, ground_truth_trajectory)
        ground_truth_distance = compute_trajectory_length(ground_truth_trajectory)
        drift = absolute_error / max(ground_truth_distance, 1e-3) * 100.0
        print(f"[*] Ground truth distance: {ground_truth_distance:.2f} m")
        print(f"[*] Absolute Error (ATE): {absolute_error:.3f} m")
        print(f"[*] Relative Drift: {drift:.2f} %")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
