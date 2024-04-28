import glob
import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset


class PlacesDataset(Dataset):
    def __init__(self, indir, split, out_size):
        self.__prepare_in_files(indir, split)
        self.transform = get_transforms(split, out_size)
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        self.iter_i += 1
        return img

    def __prepare_in_files(self, indir, split):
        if split == "train":
            self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        elif split =="val":
            mask_filenames = list(glob.glob(os.path.join(indir, '**', '*mask*.png'), recursive=True))
            self.in_files = [fname.rsplit('_mask', 1)[0] + ".png" for fname in mask_filenames]
        else:
            self.in_files = []


def get_transforms(split, out_size):
    if split == 'train':
        return A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    if split == 'val':
        return A.Compose([
            A.CenterCrop(height=out_size, width=out_size),
            A.ToFloat()
        ])
    return None


if __name__ == "__main__":
    train_data_config = {
        "indir": "/mnt/data/datasets/places_standard_dataset/train",
        "split": "train",
        "out_size": 256
    }
    train_dataset = PlacesDataset(**train_data_config)
    img = train_dataset.__getitem__(0)
    img = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(f"train.png")

    val_data_config = {
        "indir": "/mnt/data/datasets/places_standard_dataset/val",
        "split": "val",
        "out_size": 256
    }
    val_dataset = PlacesDataset(**val_data_config)
    img = train_dataset.__getitem__(0)
    img = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)
    Image.fromarray(img).save(f"val.png")
