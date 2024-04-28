import os
import decord
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


decord.bridge.set_bridge("torch")


class WebVidTrain(Dataset):
    def __init__(self,
                 video_params,
                 data_dir,
                 metadata_path=None,
                 subsample=1,
                 use_optical_flow_score=False
                 ):
        self.video_params = video_params
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transforms = get_tsfms("train", input_res=self.video_params["input_res"])
        self.subsample = subsample
        self.use_optical_flow_score = use_optical_flow_score
        self._load_metadata()

    def _load_metadata(self):
        metadata = pd.read_csv(self.metadata_path)
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample, random_state=42)
        self.metadata = metadata[["videoid", "page_dir"]]
        if self.use_optical_flow_score:
            optical_flow_score = torch.clip(torch.tensor(metadata["optical_flow"]), min=0, max=2)
            self.optical_flow_score = torch.softmax(optical_flow_score, dim=0).numpy()

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['page_dir']), str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.use_optical_flow_score:
            item = np.random.choice(len(self.metadata), p=self.optical_flow_score)
        else:
            item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = self._get_video_path(sample)

        try:
            image1, image2 = read_frames(video_fp, max_offset=self.video_params["max_offset"], sampling_mode=self.video_params["sampling_mode"])
        except Exception as e:
            default_image = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
            default_image = transforms.ToTensor()(default_image)
            image1, image2 = default_image.clone(), default_image.clone()

        if self.transforms is not None:
            image1, image2 = self.transforms(torch.stack([image1, image2]))

        return {"image1": image1,
                "image2": image2}


class WebVidVal(Dataset):
    def __init__(self,
                 video_params,
                 data_dir,
                 metadata_path=None,
                 num_val_videos=1000,
                 val_offsets=(1, 2, 4, 8, 16, 32, 64)
                 ):
        self.video_params = video_params
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transforms = get_tsfms("val", input_res=self.video_params["input_res"])
        self.num_val_videos = num_val_videos
        self._val_offsets = val_offsets
        self._load_metadata()
        self._prepare_pairs()

    def _load_metadata(self):
        metadata = pd.read_csv(self.metadata_path)
        if self.num_val_videos is not None:
            metadata = metadata.sample(self.num_val_videos, random_state=42)
        else:
            self.num_val_videos = len(metadata)
        self.metadata = metadata[["videoid", "page_dir"]]

    def _prepare_pairs(self):
        self.pairs = []
        for video_idx in range(self.num_val_videos):
            video_sample = self.metadata.iloc[video_idx]
            video_fp = self._get_video_path(video_sample)
            for offset in self._val_offsets:
                frame_idxs = [0, offset]
                self.pairs.append((video_fp, frame_idxs))

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['page_dir']), str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __len__(self):
        return self.num_val_videos * len(self._val_offsets)

    def __getitem__(self, item):
        item = item % self.__len__()
        video_fp, frame_idxs = self.pairs[item]

        try:
            image1, image2 = read_frames(video_fp, frame_idxs=frame_idxs)
        except Exception as e:
            default_image = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
            default_image = transforms.ToTensor()(default_image)
            image1, image2 = default_image.clone(), default_image.clone()

        if self.transforms is not None:
            image1, image2 = self.transforms(torch.stack([image1, image2]))

        return {"image1": image1,
                "image2": image2}


class WebVid(Dataset):
    def __init__(self, data_dir, metadata_path=None, default_res=256):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.default_res = default_res
        self._load_metadata()
        self._prepare_pairs()

    def _load_metadata(self):
        metadata = pd.read_csv(self.metadata_path)
        self.num_videos = len(metadata)
        self.metadata = metadata[["videoid", "page_dir"]]

    def _prepare_pairs(self):
        self.pairs = []
        for video_idx in range(self.num_videos):
            video_sample = self.metadata.iloc[video_idx]
            frame_idxs = [0, 1]
            self.pairs.append((video_sample, frame_idxs))

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(str(sample['page_dir']), str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __len__(self):
        return self.num_videos

    def __getitem__(self, item):
        video_sample, frame_idxs = self.pairs[item]
        video_fp = self._get_video_path(video_sample)

        try:
            image1, image2 = read_frames(video_fp, frame_idxs=frame_idxs)
        except Exception as e:
            default_image = Image.new('RGB', (self.default_res, self.default_res), (0, 0, 0))
            default_image = transforms.ToTensor()(default_image)
            image1, image2 = default_image.clone(), default_image.clone()

        return {"image1": image1,
                "image2": image2,
                "videoid": video_sample["videoid"],
                "page_dir": video_sample["page_dir"]}


def sample_frames(vlen, max_offset=5*24, sampling_mode="uniform"):
    offset = max_offset + 1
    while offset > max_offset or offset > vlen - 1:
        if sampling_mode == "exponential":
            offset = 1 + int(abs(np.random.exponential(max_offset / 4)))
        else:
            offset = 1 + int(np.random.uniform(0, max_offset - 1))
    first_idx = np.random.randint(0, vlen-offset)
    return [first_idx, first_idx + offset]


def read_frames(video_path, max_offset=5*24, frame_idxs=None, sampling_mode="uniform"):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    if frame_idxs is None:
        frame_idxs = sample_frames(vlen, max_offset, sampling_mode=sampling_mode)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float()
    frames = frames.permute(0, 3, 1, 2)
    return frames


def get_tsfms(split,
              input_res=256,
              center_crop=256,
              randcrop_scale=(0.5, 1.0),
              randcrop_ratio=(0.9, 1.1),
              color_jitter=(0, 0, 0)):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, ratio=randcrop_ratio),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
        ])
    if split == "val":
        return transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
        ])
    return None


if __name__ == "__main__":
    train_data_config = {
        "video_params": {
            "input_res": 256,
            "max_offset": 120
        },
        "data_dir": "/mnt/data/yaroslav/webvid/data",
        "metadata_path": "/mnt/data/yaroslav/webvid/results_05M_train.csv",
        "subsample": 1
    }
    train_webvid_dataset = WebVidTrain(**train_data_config)
    train_data = train_webvid_dataset.__getitem__(0)
    for key, val in train_data.items():
        img = (val.permute(1, 2, 0).numpy()).astype(np.uint8)
        Image.fromarray(img).save(f"train_{key}.png")

    val_data_config = {
        "video_params": {
            "input_res": 256
        },
        "data_dir": "/mnt/data/yaroslav/webvid/data",
        "metadata_path": "/mnt/data/yaroslav/webvid/results_2M_val.csv",
        "num_val_videos": 1000
    }
    val_webvid_dataset = WebVidVal(**val_data_config)
    val_data = val_webvid_dataset.__getitem__(0)
    for key, val in val_data.items():
        img = (val.permute(1, 2, 0).numpy()).astype(np.uint8)
        Image.fromarray(img).save(f"val_{key}.png")
