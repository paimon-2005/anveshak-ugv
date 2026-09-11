import numpy as np
import cv2
import os


class MonoVideoOdometery(object):
    def __init__(self, 
                img_file_path,
                pose_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        
        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.id = 0
        self.n_features = 0

        try:
            files = os.listdir(img_file_path)
            if not any(f.lower().endswith('.png') for f in files):
                raise ValueError("img_file_path does not contain png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
        p0 = self.detector.detect(img)
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)

        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)

        if self.id < 2:
            self.R = R
            self.t = t
        else:
            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale * self.R.dot(t)
                self.R = R.dot(self.R)

        self.n_features = self.good_new.shape[0]


    def get_mono_coordinates(self):
        # Best mapping for KITTI (flips X and Z so direction matches ground truth)
        return np.array([-self.t[0][0], -self.t[1][0], -self.t[2][0]])


    def get_true_coordinates(self):
        return self.true_coord.flatten()


    def get_absolute_scale(self):
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
        return np.linalg.norm(true_vect - prev_vect)


    def process_frame(self):
        if self.id < 2:
            self.old_frame = cv2.imread(os.path.join(self.file_path, str(0).zfill(6) + '.png'), 0)
            self.current_frame = cv2.imread(os.path.join(self.file_path, str(1).zfill(6) + '.png'), 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(os.path.join(self.file_path, str(self.id).zfill(6) + '.png'), 0)
            self.visual_odometery()
            self.id += 1