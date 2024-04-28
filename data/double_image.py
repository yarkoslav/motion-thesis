import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def process_image(image_path, size):
    image = Image.open(image_path)
    width, height = image.size
    mid = (width // 2, height // 2)
    crop_size = np.min([width, height])
    image = image.crop([mid[0] - crop_size // 2, mid[1] - crop_size // 2, mid[0] + crop_size // 2, mid[1] + crop_size // 2])
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    image = np.array(image).astype(np.uint8)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    return image


class DoubleImageTrainDataset(Dataset):
    def __init__(self, file_path1, file_path2, size):
        self.image1 = process_image(file_path1, size)
        self.image2 = process_image(file_path2, size)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return {"image1": self.image1,
                "image2": self.image2}


class DoubleImageValDataset(Dataset):
    def __init__(self, file_path1, file_path2, size):
        self.image1 = process_image(file_path1, size)
        self.image2 = process_image(file_path2, size)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return {"image1": self.image1,
                "image2": self.image2}
