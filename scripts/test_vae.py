import os
import argparse
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.video_dataset import WebVidVal
from FlowFormerPlusPlus.flowformerpp_wrapper import viz_flow
from utils import instantiate_from_config


def save_image(out_dir, name, img):
    img = torch.clamp(img, 0, 1)
    img = (img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    Image.fromarray(img).save(os.path.join(out_dir, name))


if __name__ == "__main__":
    # command line arguments and configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--ckpt_path",
        type=str,
        help="path to checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="path to output directory"
    )

    opt, unknown = parser.parse_known_args()
    config = OmegaConf.load(opt.config_path)

    # model and load
    model = instantiate_from_config(config.model)
    state_dict = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    # data
    batch_size = 1
    data_config = config.data.params.validation.params
    data_config.num_val_videos = None

    dataset = WebVidVal(**data_config)
    dataloader = DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=2*batch_size,
                          shuffle=False)

    # test loop
    out_dir = opt.out_dir
    os.makedirs(out_dir, exist_ok=True)
    epe_list = []
    for idx, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
            input_flow, decoded_flow, _ = model(batch, sample_posterior=False)
            image1, image2 = batch["image1"] / 255.0, batch["image2"] / 255.0
            # input_flow, decoded_flow = decode_flow(input_flow), decode_flow(decoded_flow)
            if torch.isinf(input_flow).any() or torch.isinf(decoded_flow).any():
                continue

            # epe
            epe = torch.sum((input_flow - decoded_flow) ** 2, dim=1).sqrt()
            epe_list.append(epe.view(-1).detach().cpu().numpy())

            # visualization
            base_name = str(idx).zfill(5)
            save_image(out_dir, base_name + "_flow_gt.png", viz_flow(input_flow))
            save_image(out_dir, base_name + "_flow_reconstructed.png", viz_flow(decoded_flow))
            save_image(out_dir, base_name + "_image1.png", image1)
            save_image(out_dir, base_name + "_image2.png", image2)


    print(f"EPE is {np.mean(np.concatenate(epe_list))}")
