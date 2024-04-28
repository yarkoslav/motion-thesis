import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SingleImageTrainDataset(Dataset):
    def __init__(self, file_path, size):
        self.image = Image.open(file_path)
        self.image = self.image.resize((size, size), Image.Resampling.BILINEAR)
        self.image = np.array(self.image).astype(np.uint8)
        self.image = np.transpose(self.image, (2, 0, 1))
        self.image = self.image.astype(np.float32) / 255.0

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.image


class SingleImageValDataset(Dataset):
    def __init__(self, file_path, size):
        self.image = Image.open(file_path)
        self.image = self.image.resize((size, size), Image.Resampling.BILINEAR)
        self.image = np.array(self.image).astype(np.uint8)
        self.image = np.transpose(self.image, (2, 0, 1))
        self.image = self.image.astype(np.float32) / 255.0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image
