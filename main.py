import os
import torch.nn as nn
from network.classifier import Classifier
import torch.cuda
import numpy as np
#import sys
import cv2
from dataset.dataset_loader import IRIGesture
from torch.utils.data import DataLoader
#np.set_printoptions(threshold=sys.maxsize)



class GestureModel():

    def __init__(self, root_dir=None, weights_path=None):
        # Init variables
        self._root_dir = root_dir
        self._epochs = 261

        # Create NN structure
        self.model = Classifier()

        # Add CUDA if available
        #if torch.cuda.is_available():
        #    self.model.cuda()

        # Define cost function criterion (Cross Entropy Loss for multi-classification)
        self.criterion = nn.CrossEntropyLoss()

        # Define optimization algorithm (Adam) and learning rate
        lr = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
            #weights_path = "/home/rromero/PycharmProjects/gesture/experiments/exp8/ep200.zip"


    def train(self):
        # Initialize variables
        self.model.train()

        # Load training dataset
        train_dataset = IRIGesture(root_dir=self._root_dir, is_for="train")

        # Create dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=216, shuffle=True, drop_last=True)

        loss_values = []
        accuracies = []

        # Keep track epoch-by-epoch of the training process
        for epoch in range(self._epochs):
            print(f"{epoch} epochs out of {self._epochs} completed.")
            for i, batch in enumerate(train_dataloader):
                landmarks = batch['landmarks'].float()
                gesture = batch['gesture']

                # Feed forward
                pred = self.model(landmarks)

                # Loss function
                loss = self.criterion(pred, torch.argmax(gesture, dim=-1))

                # Backprop
                self.optimizer.zero_grad()  # Sets the gradients of all optimized torch.tensor's to zero
                loss.backward()  # Computes the gradients

                # Optimize
                self.optimizer.step()  # Updates the parameters

                # Keep track of the loss epoch-by-epoch
                loss_values.append(loss.data.item())

            # We save the model weights every 5 epochs for later testing
            # We also save the accuracy
            if epoch%5 == 0:
                torch.save(self.model.state_dict(), os.getcwd() + "experiments/exp10/ep"+str(epoch))

                correct = 0
                total = 0

                # We test the accuracy of the model in the current epoch
                with torch.no_grad():
                    for data in train_dataloader:
                        landmarks = data['landmarks'].float()
                        gestures = data['gesture']
                        outputs = self.model(landmarks)
                        m = nn.Softmax(dim=1)
                        pred_label = m(outputs)
                        _, predicted = torch.max(outputs, 1)
                        _, correct_gesture = torch.max(gestures, 1)
                        total += gestures.size(0)

                        #print(outputs)
                        #print(f"predicted: {predicted.numpy()}")
                        #print(f"correct_gesture: {correct_gesture.numpy()}")
                        #print(f"softmax(predicted): {pred_label}")
                        #print(total)

                        correct += (predicted == correct_gesture).sum().item()

                    print('Accuracy of the network on the training set: %d %%' % (100 * correct / total))
                    accuracies.append(100 * correct / total)

        print(f"Loss values: {loss_values}")
        print(f"Accuracies: {accuracies}")


    def test(self):
        # Set model to evaluation mode
        self.model.eval()

        # Load test dataset
        test_dataset = IRIGesture(root_dir=self._root_dir, is_for="test")

        # Create dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=True, drop_last=True)

        test_loss_values = []
        test_accuracies = []

        # Load the saved model weights every 5 epochs for testing
        for epoch in range(self._epochs):
            if epoch%5 == 0:
                print(f"epoch number: {epoch}")
                self.model.load_state_dict(torch.load(os.getcwd() + "/experiments/exp10/ep"+str(epoch)))

                # Compute the loss and test the accuracy in the current epoch
                with torch.no_grad():
                    test_iter = iter(test_dataloader)
                    test_batch = next(test_iter)

                    test_landmarks = test_batch['landmarks'].float()
                    test_gesture = test_batch['gesture']
                    test_output = self.model(test_landmarks)

                    test_loss = self.criterion(test_output, torch.argmax(test_gesture, dim=-1))
                    test_loss_values.append(test_loss.data.item())

                    correct_test = 0
                    total_test = 0

                    for data_test in test_dataloader:
                        landmarks_test = data_test['landmarks'].float()
                        gestures_test = data_test['gesture']
                        #image_test = data_test['image'].numpy()
                        outputs_test = self.model(landmarks_test)
                        m_test = nn.Softmax(dim=1)
                        pred_label_test = m_test(outputs_test)
                        _, predicted_test = torch.max(pred_label_test, 1)
                        _, correct_gesture_test = torch.max(gestures_test, 1)
                        total_test += gestures_test.size(0)

                        print(f"predicted_test: {predicted_test}")
                        print(f"correct_gesture_test: {correct_gesture_test}")
                        #print(f"softmax(predicted_test): {pred_label_test}")
                        #print(total_test)

                        correct_test += (predicted_test == correct_gesture_test).sum().item()

                        print('Accuracy of the network on the test set: %d %%' % (100 * correct_test / total_test))
                        test_accuracies.append(100 * correct_test / total_test)

                        ## For visualization of some results every 100 epochs
                        #if epoch%100 == 0:
                        #    for i in range(4):
                        #        print(f"gesture prediction: {predicted_test[i]}")
                        #        print(f"correct gesture: {correct_gesture_test[i]}")
                        #        while True:
                        #            cv2.imshow('image' + str(i), image_test[i])
                        #            if cv2.waitKey(0) == 27:
                        #                cv2.destroyAllWindows()
                        #                break

                        # Compute confusion matrix at 200 epochs for metrics
                        if epoch == 200:
                            confusion_matrix = np.zeros((8, 8))

                            for i in range(24):
                                confusion_matrix[predicted_test[i].item()][correct_gesture_test[i].item()] += 1

                            print(confusion_matrix)

        print(f"Test loss values: {test_loss_values}")
        print(f"Test accuracies: {test_accuracies}")


    def live(self):
        import mediapipe as mp

        # Load a certain set of weights for live predictions
        weights_path = os.getcwd() + "/experiments/exp8/ep200.zip"
        self.model.load_state_dict(torch.load(weights_path))

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # For webcam input: (-1 value may be changed to find the webcam input)
        cap = cv2.VideoCapture(0)
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()

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
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Pose', image)

                k = cv2.waitKey(1)

                # Press ESC to end the process
                if k == 27:
                    break

                # Press SPACE to make a prediction on the current frame
                if k == 32:
                    pose_landmarks = results.pose_landmarks
                    landmarks = np.zeros(99)
                    landmarks[0] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x
                    landmarks[1] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y
                    landmarks[2] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].z
                    landmarks[3] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].x
                    landmarks[4] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].y
                    landmarks[5] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_INNER].z
                    landmarks[6] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x
                    landmarks[7] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].y
                    landmarks[8] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].z
                    landmarks[9] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].x
                    landmarks[10] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].y
                    landmarks[11] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER].z
                    landmarks[12] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].x
                    landmarks[13] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].y
                    landmarks[14] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_INNER].z
                    landmarks[15] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x
                    landmarks[16] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].y
                    landmarks[17] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].z
                    landmarks[18] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x
                    landmarks[19] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y
                    landmarks[20] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z
                    landmarks[21] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].x
                    landmarks[22] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y
                    landmarks[23] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].z
                    landmarks[24] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].x
                    landmarks[25] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].y
                    landmarks[26] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR].z
                    landmarks[27] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].x
                    landmarks[28] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].y
                    landmarks[29] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_LEFT].z
                    landmarks[30] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].x
                    landmarks[31] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].y
                    landmarks[32] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.MOUTH_RIGHT].z
                    landmarks[33] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x
                    landmarks[34] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
                    landmarks[35] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z
                    landmarks[36] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                    landmarks[37] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                    landmarks[38] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z
                    landmarks[39] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x
                    landmarks[40] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
                    landmarks[41] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].z
                    landmarks[42] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x
                    landmarks[43] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
                    landmarks[44] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].z
                    landmarks[45] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x
                    landmarks[46] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y
                    landmarks[47] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].z
                    landmarks[48] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x
                    landmarks[49] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
                    landmarks[50] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].z
                    landmarks[51] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].x
                    landmarks[52] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].y
                    landmarks[53] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_PINKY].z
                    landmarks[54] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].x
                    landmarks[55] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].y
                    landmarks[56] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY].z
                    landmarks[57] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].x
                    landmarks[58] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].y
                    landmarks[59] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX].z
                    landmarks[60] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].x
                    landmarks[61] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].y
                    landmarks[62] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX].z
                    landmarks[63] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].x
                    landmarks[64] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].y
                    landmarks[65] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_THUMB].z
                    landmarks[66] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].x
                    landmarks[67] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].y
                    landmarks[68] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB].z
                    landmarks[69] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x
                    landmarks[70] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y
                    landmarks[71] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z
                    landmarks[72] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x
                    landmarks[73] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y
                    landmarks[74] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z
                    landmarks[75] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x
                    landmarks[76] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y
                    landmarks[77] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].z
                    landmarks[78] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
                    landmarks[79] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
                    landmarks[80] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE].z
                    landmarks[81] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x
                    landmarks[82] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
                    landmarks[83] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].z
                    landmarks[84] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x
                    landmarks[85] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
                    landmarks[86] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z
                    landmarks[87] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x
                    landmarks[88] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y
                    landmarks[89] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].z
                    landmarks[90] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].x
                    landmarks[91] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].y
                    landmarks[92] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL].z
                    landmarks[93] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
                    landmarks[94] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
                    landmarks[95] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z
                    landmarks[96] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
                    landmarks[97] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
                    landmarks[98] = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z

                    with torch.no_grad():
                        landmarks_tensor = torch.Tensor(landmarks)
                        outputs_live = self.model(landmarks_tensor)
                        #print(outputs_live)
                        m_live = nn.Softmax(dim=0)
                        pred_label_live = m_live(outputs_live)
                        percentages = 100*pred_label_live
                        #print(pred_label_live)
                        print("PROBABILITY PERCENTAGES:")
                        print(f"Attention: {round(percentages[0].data.item(), 2)} %")
                        print(f"Right: {round(percentages[1].data.item(), 2)} %")
                        print(f"Left: {round(percentages[2].data.item(), 2)} %")
                        print(f"Stop: {round(percentages[3].data.item(), 2)} %")
                        print(f"Yes: {round(percentages[4].data.item(), 2)} %")
                        print(f"Shrug: {round(percentages[5].data.item(), 2)} %")
                        print(f"Random: {round(percentages[6].data.item(), 2)} %")
                        print(f"Static: {round(percentages[7].data.item(), 2)} %")

                        _, predicted_live = torch.max(pred_label_live, 0)
                        gesture_names = ["ATTENTION", "RIGHT", "LEFT", "STOP", "YES", "SHRUG", "RANDOM", "STATIC"]
                        print(f"Predicted gesture: {gesture_names[predicted_live]}")

        cap.release()

    def infer(self, landmarks):
        with torch.no_grad():
            landmarks_tensor_infer = torch.Tensor(landmarks)
            outputs_infer = self.model(landmarks_tensor_infer)
            # print(outputs_live)
            m_infer = nn.Softmax(dim=0)
            pred_label_infer = m_infer(outputs_infer)
            #print(pred_label_live)
            #_, predicted_live = torch.max(pred_label_live, 0)

            #gesture_names = ["ATTENTION", "RIGHT", "LEFT", "STOP", "YES", "SHRUG", "RANDOM", "STATIC"]
            #print(f"Predicted gesture: {gesture_names[predicted_live]}")
            return pred_label_infer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = GestureModel(root_dir=os.getcwd() + "/dataset/BodyGestureDataset")
    model.live()
