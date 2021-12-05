# Most part of this code has been obtained from MediaPipe Pose Python solution API, from its GitHub webpage,
# and modified to perform the desired task of filming videos for our dataset and obtaining its 3D joints.

import cv2
import mediapipe as mp
import numpy as np


# Predictions for images
def image_file():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # For static images:
    IMAGE_FILES = []
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )
            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)


# Live predictions using webcam
def webcam():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    record = False

    # For webcam input: (-1 value may be changed to find the webcam input)
    cap = cv2.VideoCapture(-1)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image_ = image

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('MediaPipe Pose', image)

            k = cv2.waitKey(1)


            # Press SPACE to start filming and storing landmarks in numpy array
            if k == 32:
                print("Test")
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('results.avi', fourcc, 20.0, (640, 480))
                pose_array = np.array(results.pose_landmarks)

                record = True


            # Press ESC to save numpy array and end filming
            if k == 27:
                np.save('result.npy', pose_array)
                break


            if record:
                out.write(image_)
                pose_array = np.vstack((pose_array, results.pose_landmarks))


    cap.release()
    out.release()


if __name__ == "__main__":
    webcam()