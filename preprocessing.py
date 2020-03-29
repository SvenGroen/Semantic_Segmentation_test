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


def get_all_frames(cap, grey=False):
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(
            np.moveaxis(frame, -1, 0))  # changes channel order for pytorch (n_frames, channels, height, width)

    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def get_ball_train_test(path, split_percentage, grey=False):
    cap = cv2.VideoCapture(path)
    frames = get_all_frames(cap, grey=False)

    # 80% train-test split (0.8-0.2)
    frames_train = torch.Tensor(frames[:int(len(frames) * split_percentage)])
    frames_test = torch.Tensor(frames[int(len(frames) * split_percentage):])
    labels_train = torch.Tensor(frames[:int(len(frames) * split_percentage)])
    labels_test = torch.Tensor(frames[int(len(frames) * split_percentage):])

    return frames_train, labels_train, frames_test, labels_test
