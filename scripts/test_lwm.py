import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
    data_config.num_val_videos = 1000

    dataset = WebVidVal(**data_config)
    dataloader = DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=2*batch_size,
                          shuffle=False)

    # test loop
    out_dir = opt.out_dir
    os.makedirs(out_dir, exist_ok=True)
    mse_list, recon_list = [], []
    for idx, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
            z1_warped, z2, _, _, _, occlusion_map, flow_back = model(batch)
            occlusion_map_resized = F.interpolate(occlusion_map, size=z1_warped.shape[2:], mode="nearest")
            image1_warped_recon = model.image_vae.decode(z1_warped)
            image1, image2 = batch["image1"] / 255.0, batch["image2"] / 255.0

            # mse and recon metrics calculation
            mse = F.mse_loss(z1_warped * (1 - occlusion_map_resized), z2 * (1 - occlusion_map_resized))
            mse_list.append(mse.detach().cpu().numpy())

            recon = F.mse_loss(image1_warped_recon * (1 - occlusion_map), image2 * (1 - occlusion_map))
            recon_list.append(recon.detach().cpu().numpy())

            # visualization
            flow_back = viz_flow(flow_back)
            base_name = str(idx).zfill(5)
            save_image(out_dir, base_name + "_warped.png", image1_warped_recon)
            save_image(out_dir, base_name + "_image1.png", image1)
            save_image(out_dir, base_name + "_image2.png", image2)
            save_image(out_dir, base_name + "_flow_back.png", flow_back)
            save_image(out_dir, base_name + "_occlusion.png", occlusion_map)

    print(f"MSE is {np.mean(mse_list)}, recon is {np.mean(recon_list)}")
