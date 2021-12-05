import numpy as np
import os
import mediapipe as mp
import cv2
from torch.utils.data import Dataset


# We implement a PyTorch dataset loader that pre-processes the collected data to be loaded to the model

class IRIGesture(Dataset):
    def __init__(self, root_dir, num_gestures=8, is_for="train"):
        self.root_dir = root_dir

        # Choose the subjects for training set (default)
        if is_for == "train":
            self._subjects = [1, 3, 4, 5, 6, 7, 8, 9, 10]

        # Choose the subjects for test set
        elif is_for == "test":
            self._subjects = [2]

        # Two arrays storing all the paths for the 3D joints and the videos for later use
        self._paths = []
        self._videos = []
        for subject in self._subjects:
            subject_path = os.path.join(self.root_dir, "S"+str(subject))
            for data in os.listdir(subject_path):
                if data == "3Djoints":
                    _3Djoints_path = os.path.join(subject_path, "3Djoints")
                    _videos_path = os.path.join(subject_path, "videos")
                    for gesture in os.listdir(_3Djoints_path):
                        file_path = os.path.join(_3Djoints_path, gesture)
                        file_name = file_path.split("/")[-1].split(".")[0]
                        self._paths.append(file_path)
                        video_path = os.path.join(_videos_path, file_name + ".avi")
                        self._videos.append(video_path)

        # Number of gestures we are using from the dataset
        self.num_gestures = num_gestures

        self.mp_pose = mp.solutions.pose

    def __len__(self):
        return len(self._paths)
        # Length of the dataset

    def __getitem__(self, item):
        # Sample item of the dataset
        item_path = self._paths[item]
        video_path = self._videos[item]
        gesture_str = item_path.split("/")[-1].split(".")[0].split("_")[0]

        # We vectorize the gesture labelling of the samples
        gesture = np.zeros(self.num_gestures)

        if gesture_str == "attention":
            gesture[0] = 1

        if gesture_str == "right":
            gesture[1] = 1

        if gesture_str == "left":
            gesture[2] = 1

        if gesture_str == "stop":
            gesture[3] = 1

        if gesture_str == "yes":
            gesture[4] = 1

        if gesture_str == "shrug":
            gesture[5] = 1

        if gesture_str == "random":
            gesture[6] = 1

        if gesture_str == "static":
            gesture[7] = 1

        # In our use case we ignore dynamic gestures
        #if gesture_str == "greeting":
        #    gesture[8] = 1

        #if gesture_str == "continue":
        #    gesture[9] = 1

        #if gesture_str == "turnback":
        #    gesture[10] = 1

        #if gesture_str == "no":
        #    gesture[11] = 1

        #if gesture_str == "slowdown":
        #    gesture[12] = 1

        #if gesture_str == "come":
        #    gesture[13] = 1

        #if gesture_str == "back":
        #    gesture[14] = 1

        # We obtain the last frame of the corresponding sample video
        image = np.empty([])

        vs = cv2.VideoCapture(video_path)
        last_frame_num = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        vs.set(cv2.CAP_PROP_POS_FRAMES, last_frame_num - 1)
        _, frame = vs.read()
        image = frame
        vs.release()

        # Now we transform the data to pass the 3D joints to a numpy array of size (T, 99) where T = # frames
        # pose_list is an array of size T where every cell is an object containing the 33 3-dimensional coordinates
        pose_list = np.load(item_path, allow_pickle=True)
        T, _ = pose_list.shape

        landmarks = np.zeros((T, 99))
        # For every frame, we get the 99 coordinates and store them in the "landmarks" array corresponding frame row
        i = 0
        for pose in pose_list:
            landmarks[i][0] = pose[0].landmark[self.mp_pose.PoseLandmark.NOSE].x
            landmarks[i][1] = pose[0].landmark[self.mp_pose.PoseLandmark.NOSE].y
            landmarks[i][2] = pose[0].landmark[self.mp_pose.PoseLandmark.NOSE].z
            landmarks[i][3] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].x
            landmarks[i][4] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].y
            landmarks[i][5] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].z
            landmarks[i][6] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x
            landmarks[i][7] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y
            landmarks[i][8] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE].z
            landmarks[i][9] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].x
            landmarks[i][10] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].y
            landmarks[i][11] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].z
            landmarks[i][12] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].x
            landmarks[i][13] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].y
            landmarks[i][14] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].z
            landmarks[i][15] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x
            landmarks[i][16] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y
            landmarks[i][17] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].z
            landmarks[i][18] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x
            landmarks[i][19] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y
            landmarks[i][20] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z
            landmarks[i][21] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x
            landmarks[i][22] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y
            landmarks[i][23] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_EAR].z
            landmarks[i][24] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x
            landmarks[i][25] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y
            landmarks[i][26] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].z
            landmarks[i][27] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].x
            landmarks[i][28] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].y
            landmarks[i][29] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].z
            landmarks[i][30] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].x
            landmarks[i][31] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].y
            landmarks[i][32] = pose[0].landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].z
            landmarks[i][33] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
            landmarks[i][34] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            landmarks[i][35] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z
            landmarks[i][36] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            landmarks[i][37] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            landmarks[i][38] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z
            landmarks[i][39] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x
            landmarks[i][40] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
            landmarks[i][41] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].z
            landmarks[i][42] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x
            landmarks[i][43] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            landmarks[i][44] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z
            landmarks[i][45] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x
            landmarks[i][46] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            landmarks[i][47] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].z
            landmarks[i][48] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x
            landmarks[i][49] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            landmarks[i][50] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].z
            landmarks[i][51] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].x
            landmarks[i][52] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].y
            landmarks[i][53] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].z
            landmarks[i][54] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x
            landmarks[i][55] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y
            landmarks[i][56] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].z
            landmarks[i][57] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x
            landmarks[i][58] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y
            landmarks[i][59] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].z
            landmarks[i][60] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x
            landmarks[i][61] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y
            landmarks[i][62] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].z
            landmarks[i][63] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x
            landmarks[i][64] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y
            landmarks[i][65] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].z
            landmarks[i][66] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x
            landmarks[i][67] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y
            landmarks[i][68] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].z
            landmarks[i][69] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x
            landmarks[i][70] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y
            landmarks[i][71] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z
            landmarks[i][72] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x
            landmarks[i][73] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            landmarks[i][74] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z
            landmarks[i][75] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x
            landmarks[i][76] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y
            landmarks[i][77] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].z
            landmarks[i][78] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
            landmarks[i][79] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
            landmarks[i][80] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].z
            landmarks[i][81] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x
            landmarks[i][82] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
            landmarks[i][83] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].z
            landmarks[i][84] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x
            landmarks[i][85] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
            landmarks[i][86] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z
            landmarks[i][87] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x
            landmarks[i][88] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y
            landmarks[i][89] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].z
            landmarks[i][90] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].x
            landmarks[i][91] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].y
            landmarks[i][92] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].z
            landmarks[i][93] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
            landmarks[i][94] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            landmarks[i][95] = pose[0].landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z
            landmarks[i][96] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
            landmarks[i][97] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            landmarks[i][98] = pose[0].landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z
            i = i+1

        # Each sample is formed by last video frame (image), gesture label vector (gesture)
        # and 3D joints of last frame (landmarks)
        sample = {'image': image, 'gesture': gesture, 'landmarks': landmarks[-1]}

        return sample


if __name__ == "__main__":
    dataset = IRIGesture(root_dir="/home/ramon/rromero/PycharmProjects/gesture/dataset/BodyGestureDataset", is_for="train")
    print(f"dataset length: {len(dataset)} samples")
    #for i in range(7):
    #    print(dataset[i]["landmarks"])
    #    print(dataset[i]["gesture"])
    #    while True:
    #        cv2.imshow('last_frame_' + str(i), dataset[i]["image"])
    #        if cv2.waitKey(0) == 27:
    #            break

    # This is for visualizing some of the dataset samples

