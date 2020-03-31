import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class BallDataset(Dataset):

    def __init__(self, path, grey):
        self.cap = cv2.VideoCapture(path)
        self.transform = transforms.Compose([transforms.ToTensor()])
        #, transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_gry = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()])
#, transforms.Normalize((0.5,), (0.5,))
        self.grey = grey
        self.frames, self.labels = self.get_all_frames(self, self.transform, self.grey)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return (self.frames[idx], self.labels[idx])

    @staticmethod
    def get_all_frames(self, transform, grey):
        frames = []
        labels = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if grey:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                label = frame
            else:
                # label = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                label = frame
            frame = Image.fromarray(frame)
            frame = self.transform(frame)

            label = Image.fromarray(label)
            # label = torch.Tensor(label).unsqueeze(0)
            trf = transforms.ToPILImage()
            # label = trf(label)
            label = self.transform_gry(label)
            labels.append(label)
            frames.append(frame)
            # np.moveaxis(frame, -1, 0))  # changes channel order for pytorch (n_frames, channels, height, width)

        self.cap.release()
        return frames, labels  # np.asarray(frames, dtype=np.uint8)

    # def get_ball_train_test(path, split_percentage, grey=False):
    #     cap = cv2.VideoCapture(path)
    #     frames = get_all_frames(cap, grey=False)
    #
    #     # 80% train-test split (0.8-0.2)
    #     # frames_train = torch.Tensor(frames[:int(len(frames) * split_percentage)])
    #     # frames_test = torch.Tensor(frames[int(len(frames) * split_percentage):])
    #     # labels_train = torch.Tensor(frames[:int(len(frames) * split_percentage)])
    #     # labels_test = torch.Tensor(frames[int(len(frames) * split_percentage):])
    #
    #     # transform = transforms.Compose([transforms.Normalize()])
    #
    #     return frames_train, labels_train, frames_test, labels_test
