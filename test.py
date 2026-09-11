import numpy as np
import cv2 as cv
from monovideoodometery import MonoVideoOdometery
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process paths for image and pose data.')
    parser.add_argument('--img_path', type=str, default='./images', help='Path to the image directory')
    parser.add_argument('--pose_path', type=str, default='./pose', help='Path to the pose file')
    args = parser.parse_args()

    if args.img_path == './images' or args.pose_path == './pose':
        print("Warning: Using default paths. Specify --img_path and --pose_path.")

    img_path = args.img_path
    pose_path = args.pose_path

    focal = 718.8560
    pp = (607.1928, 185.2157)

    lk_params = dict(winSize=(21, 21),
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    vo = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
    
    # Bigger canvas so the path stays visible longer
    traj = np.zeros((800, 1000, 3), dtype=np.uint8)

    while vo.hasNextFrame():
        frame = vo.current_frame
        cv.imshow('frame', frame)
        k = cv.waitKey(1)
        if k == 27:  # ESC to stop
            break

        vo.process_frame()

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()

        print("MSE Error: ", np.linalg.norm(mono_coord - true_coord))
        print(f"x: {mono_coord[0]:.2f}, y: {mono_coord[1]:.2f}, z: {mono_coord[2]:.2f}")
        print(f"true_x: {true_coord[0]:.2f}, true_y: {true_coord[1]:.2f}, true_z: {true_coord[2]:.2f}")

        draw_x = int(round(mono_coord[0]))
        draw_z = int(round(mono_coord[2]))
        true_x = int(round(true_coord[0]))
        true_z = int(round(true_coord[2]))

        # Center the trajectory and make positive Z go UP on the screen (looks forward)
        offset_x = 500
        offset_z = 400

        # Red = ground truth
        traj = cv.circle(traj, (true_x + offset_x, -true_z + offset_z), 2, (255, 87, 34), 3)
        # Green = estimated
        traj = cv.circle(traj, (draw_x + offset_x, -draw_z + offset_z), 2, (3, 169, 244), 3)

        cv.putText(traj, 'Actual Position: BLUE', (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 87, 34), 2)
        cv.putText(traj, 'Estimated Odometry: ORANGE', (30, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (3, 169, 244), 2)

        cv.imshow('trajectory', traj)

    cv.imwrite("./images/trajectory.png", traj)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()