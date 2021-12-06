import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class HotDogModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5, 2)
        self.conv3 = nn.Conv2d(12, 24, 5, 2)
        self.conv4 = nn.Conv2d(24, 12, 3)

        self.fc1 = nn.Linear(108, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu6(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu6(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu6(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu6(self.conv4(x)), 2)

        x = x.reshape(1, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.out(x)

        return x

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)