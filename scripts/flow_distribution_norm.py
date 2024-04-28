import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from FlowFormerPlusPlus.flowformerpp_wrapper import build_flow_model
from data.video_dataset import WebVid


DATA_DIR = "/mnt/data/yaroslav/webvid/data"
METADATA_PATH = "/mnt/data/yaroslav/webvid/results_01M_train.csv"
DEFAULT_RES = 256


if __name__ == "__main__":
    flow_model = build_flow_model().cuda()
    flow_model.eval()
    dataset = WebVid(data_dir=DATA_DIR,
                     metadata_path=METADATA_PATH,
                     default_res=DEFAULT_RES)

    df = {"videoid": [],
          "page_dir": [],
          "optical_flow": []}
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = dataset.__getitem__(i)
            image1, image2 = sample["image1"].unsqueeze(0).cuda(), sample["image2"].unsqueeze(0).cuda()
            videoid, page_dir = sample["videoid"], sample["page_dir"]
            flow = flow_model(image1, image2)[0]  # [B, C, H, W]
            flow = flow.permute(0, 2, 3, 1).reshape((-1, 2))  # [B*H*W, C]
            flow_norm = torch.sqrt(torch.sum(flow ** 2, dim=1)) # [B*H*W]
            average_flow_norm = torch.mean(flow_norm).detach().cpu().numpy()

            df["videoid"].append(videoid)
            df["page_dir"].append(page_dir)
            df["optical_flow"].append(average_flow_norm)
    df = pd.DataFrame(df)
    df.to_csv('flow_labeled.csv', index=False)