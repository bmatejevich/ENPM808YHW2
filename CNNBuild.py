from __main__ import *


class CNNBuild(NN.Module):

    def __init__(self):
        super(CNNBuild, self).__init__()

        self.conv1 = NN.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = NN.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = NN.Linear(4608, 64)
        self.fc2 = NN.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return (x)
