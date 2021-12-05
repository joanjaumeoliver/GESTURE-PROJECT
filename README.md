# GESTURE-PROJECT

In this project we can find the scripts of my final degree work.

We will propose a solution for human gesture classification using Deep Learning models, meant to be useful for non-verbal comunication between humans and robots. We have created a human body gesture dataset using a MediaPipe pose tracking solution in order to train the main model of this work. The MediaPipe solution uses Convolutional Neural Networks for body joints extraction and our model implements Deep Neural Networks for classification. This is meant to be an introductory work to the area of Deep Learning.

- In the "dataset" folder, find a dataset containing 450 sample videos (divided in 10 different subjects) of static and dynamic human gestures. It also contains numpy arrays for every video with frame-to-frame predicted 33 3D coordinates of body joints using MediaPipe BlazePose model. Also find the code to load the dataset in the main model.

- In the "experiments" folder, find the 10 different tests performed for the work.

- In the "network" folder, find the script for the Classification Network.

- In the "utils" folder, find the script for obtaining the data for the dataset.

- Finally, find the main script containing the code for training, test and live predictions.
