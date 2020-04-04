from __main__ import *


class MNistCNN(NN.Module):

    def __init__(self):
        super(MNistCNN, self).__init__()
        self.conv1 = NN.Conv2d(1, 32, 3, 1)
        self.conv2 = NN.Conv2d(32, 64, 3, 1)
        self.dropout1 = NN.Dropout2d(0.25)
        self.dropout2 = NN.Dropout2d(0.25)
        self.fc1 = NN.Linear(9216, 128)
        self.fc2 = NN.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
