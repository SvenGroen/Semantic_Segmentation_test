import torch.nn as nn
import torch.nn.functional as F



class Ball_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool1 = nn.MaxPool2d(3, 1, 1, return_indices=True)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv_out = nn.Conv2d(3, 1, 1)

        # The output of of the conv. layers can be calculated by:
        # o_height = (n_h + 2p - f) / s + 1, where p = padding, f = kernel_size, s = stride, n_h = height of input
        # o_width = (n_w + 2p - f) / s + 1, where p = padding, f = kernel_size, s = stride, n_w = width of input
        # for square input images o_height = o_width
        self.deconv0 = nn.ConvTranspose2d(8, 3, 3, 1)
        self.unpool = nn.MaxUnpool2d(3, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, )
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, idx = self.pool1(x)
        x = self.deconv0(x)
        # x = self.unpool(x,idx)
        x = self.conv_out(x)
        # x = F.log_softmax(x, 1)
        return x

        # x = self.conv1(x)
        # x = F.relu(x)
        # x, idx_1 = F.max_pool2d(x, 3, 1, return_indeces=True)  # kernel size 3, stride 1
        # x = self.conv2(x)
        # x = F.relu(x)
        # x, idx_2 = F.max_pool2d(x, 3, 1, return_indeces=True)
        #
        # x = self.conv3(x)
        # x = F.relu(x)
        # x, idx_3 = F.max_pool2d(x, 3, 1, return_indeces=True)
        #
        # x = self.deconv1(x)
        # x = F.max_unpool2d(x, idx_1, 3, 1)
        # x = self.deconv2(x)
        # x = self.deconv3(x)
        # x = F.softmax(x, dim=1)
        # return x
