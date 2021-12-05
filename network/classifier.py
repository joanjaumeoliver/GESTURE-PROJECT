import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Define structure (4 fully connected layers of size 256, 128, 64 and 8)
        self.fc1 = nn.Linear(99, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 8)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        h_relu1 = self.ReLU(self.fc1(x))
        h_relu2 = self.ReLU(self.fc2(h_relu1))
        h_relu3 = self.ReLU(self.fc3(h_relu2))
        y_pred = self.fc4(h_relu3)
        return y_pred