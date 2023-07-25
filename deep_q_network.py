# mypy: ignore-errors
import torch
import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Layer to collapse each column into a single pixel with 64 feature channels
        self.collapse = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(1, 10))

        # More Convolutional layers
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1)

        # ReLU non-linearity
        self.relu = nn.ReLU()

        # Batch normalization and Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Convolutional layers with ReLU and BatchNorm
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv3(x))

        # Collapse columns into a single pixel
        x = self.collapse(x)

        # More Convolutional layers with ReLU and BatchNorm
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv6(x))

        # Global average pooling across spatial dimensions
        x = torch.mean(x, dim=(2, 3))

        # Fully connected layers with ReLU and Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Final fully connected layer
        x = self.fc3(x)

        return x
