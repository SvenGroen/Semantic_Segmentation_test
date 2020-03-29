import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

print(os.getcwd())


class Ball_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        # The output of of the conv. layers can be calculated by:
        # o_height = (n_h + 2p - f) / s + 1, where p = padding, f = kernel_size, s = stride, n_h = height of input
        # o_width = (n_w + 2p - f) / s + 1, where p = padding, f = kernel_size, s = stride, n_w = width of input
        # for square input images o_height = o_width

    
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, )
        self.deconv3 = nn.ConvTranspose2d(8, 2, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 1)  # kernel size 3, stride 1
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 1)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 1)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = F.softmax(x, dim = 1)
        return x
