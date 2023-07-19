# mypy: ignore-errors
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(210, 1024), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Linear(1024, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        return x
