import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import preprocessing
from Ball_net import Ball_net
from torch.utils.data import Dataset, DataLoader
import numpy as np

net = Ball_net()

path = "./videos/random_dot.avi"

# x_train, y_train, x_test, y_test = preprocessing.get_ball_train_test(path, split_percentage=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.01)
# criterion = nn.NLLLoss2d()

dataset = preprocessing.BallDataset(path, False)
train_loader = DataLoader(dataset=dataset, batch_size=10)

for epoch in range(3):
    total_loss = 0
    batch_count = 0
    for batch in train_loader:
        images, labels = batch
        # labels = labels.squeeze(1)
        pred = net(images)
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_count += 1
        total_loss += loss.item()
        print(batch_count)
    print("epoch: {}, \t batch: {}, \t loss: {}".format(epoch, batch_count, total_loss))
