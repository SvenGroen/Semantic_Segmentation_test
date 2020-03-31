import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class BallDataset(Dataset):

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform_gry = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()])
        self.frames, self.labels = self.get_all_frames(self)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

    @staticmethod
    def get_all_frames(self):
        frames = []
        labels = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            label = frame
            label = Image.fromarray(label)
            label = self.transform_gry(label)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            labels.append(label)
            frames.append(frame)

            # label = torch.Tensor(label).unsqueeze(0)
            # trf = transforms.ToPILImage()
            # label = trf(label)
            # np.moveaxis(frame, -1, 0))  # changes channel order for pytorch (n_frames, channels, height, width)

        self.cap.release()
        return frames, labels  # np.asarray(frames, dtype=np.uint8)


