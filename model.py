import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNBrain(nn.Module):
    def __init__(self, n_actions):
        super(DQNBrain, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.head = nn.Linear(1024, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
