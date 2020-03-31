import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import preprocessing
from Ball_net import Ball_net
from torch.utils.data import  DataLoader


net = Ball_net()

path = "./videos/random_dot.avi"

# x_train, y_train, x_test, y_test = preprocessing.get_ball_train_test(path, split_percentage=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.01)
# criterion = nn.NLLLoss2d()

dataset = preprocessing.BallDataset(path)
train_loader = DataLoader(dataset=dataset, batch_size=10)

for epoch in range(2):
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
    print("epoch: {}, \t batch: {}, \t loss: {}".format(epoch, batch_count, total_loss))

train_loader2 = DataLoader(dataset=dataset, batch_size=1,shuffle=True)
img, label = next(iter(train_loader2))
pred_img = net(img)
img = img.squeeze(0)
to_pil = transforms.ToPILImage()
img = to_pil(img)
img.show(title="loaded Image from dataset")

pred_img = pred_img.squeeze(0)
pred_img = to_pil(pred_img)
pred_img.show(title="predicted Image")
