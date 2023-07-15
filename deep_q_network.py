# mypy: ignore-errors
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # 2d conv network
        # kernel for convolution chosen with padding to keep original board size
        # channels is 1 because we dont have an RGB 3 channel image
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, (7, 5), padding=(3, 2)), nn.ReLU(inplace=True))
        # -> 21x10x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)), nn.ReLU(inplace=True))
        # -> 18x7x1
        # -> 21x10x1
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 1, (3, 3), padding=(1, 1)), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Linear(21*10, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # add one dimension / pseudo channel
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten 2d vector for linear layer
        # -1: determine batch size automatically
        x = x.view(-1, 21*10)
        x = self.conv4(x)

        return x
