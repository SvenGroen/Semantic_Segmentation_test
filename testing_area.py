import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import preprocessing
from Ball_net import Ball_net



net = Ball_net()

path = "./videos/random_dot.avi"

x_train, y_train, x_test, y_test = preprocessing.get_ball_train_test(path, split_percentage=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.01)

total_loss = 0


for epoch in range(3):
    for image, label in zip(x_train, y_train):
        image= torch.unsqueeze(image, 0)
        label= torch.unsqueeze(label, 0)
        pred = net(image)
        loss = F.binary_cross_entropy(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("epoch: {}, \t loss: {}".format(epoch, total_loss))
