import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.num_classes = num_classes

        self.fc1_1 = nn.Linear(4096, 512)
        self.fc1_2 = nn.Linear(4096, 512)
        self.fc1_3 = nn.Linear(4096, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(1536, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(torch.cat((self.bn1(self.fc1_1(x[:,0,:])), self.bn2(self.fc1_2(x[:,1,:])), self.bn3(self.fc1_3(x[:,2,:]))), dim=1))
        x = self.bn4(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x
