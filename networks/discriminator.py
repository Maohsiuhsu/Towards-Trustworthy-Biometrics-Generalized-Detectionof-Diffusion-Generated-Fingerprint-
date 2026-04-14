import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Discriminator_v1(nn.Module):
    def __init__(self, feature_size=2048):
        super(Discriminator_v1, self).__init__()
        self.feature_size = feature_size
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, feature_size//8),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size//8, 1),
        )

    def forward(self, x1, x2):
        # Calculate distance to the centroid
        x = torch.cat((x1, x2)).view(-1,2, self.feature_size)
        # print(f'x: {x.size()}')
        output = self.main(x)
        # print(f"out:{torch.squeeze(output).size()}")
        return torch.squeeze(output,-1)        
