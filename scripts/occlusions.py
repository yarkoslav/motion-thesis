import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from models.lwm import convert_flow
from FlowFormerPlusPlus.flowformerpp_wrapper import build_flow_model, viz_flow


def open_image(img_path):
    img = Image.open(img_path)
    W, H = img.size
    transform = transforms.Compose([
        transforms.Resize((H // 2, W // 2)),
        transforms.ToTensor()
    ])
    tensor = transform(img) * 255.0
    img = tensor.unsqueeze(0)
    return img


if __name__ == "__main__":
    flow_model = build_flow_model().cuda()
    flow_model.eval()

    img1 = open_image("FlowFormerPlusPlus/demo-frames/000016.png").cuda()
    img2 = open_image("FlowFormerPlusPlus/demo-frames/000025.png").cuda()
    with torch.no_grad():
        flow12 = flow_model(img1, img2)[0]
        flow21 = flow_model(img2, img1)[0]
        flow = flow21
    flow_viz = (viz_flow(flow)[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    flow_viz = Image.fromarray(flow_viz)
    flow_viz.save("flow.png")

    flow = convert_flow(flow)
    warped_img1 = F.grid_sample(img1, flow.permute(0, 2, 3, 1))
    warped_img1 = warped_img1[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(warped_img1).save("warped_img1.png")

    # first image coords + flow
    B, _, H, W = flow.shape
    x_coord = torch.arange(W).view(1, 1, 1, -1).repeat(B, 1, H, 1).to(flow.device)
    y_coord = torch.arange(H).view(1, 1, -1, 1).repeat(B, 1, 1, W).to(flow.device)
    coord = torch.cat([x_coord, y_coord], dim=1)
    flow_coord = coord + flow12
    warped_coord = F.grid_sample(flow_coord, flow.permute(0, 2, 3, 1), mode="nearest")
    diff = coord - warped_coord
    diff[:, 0, :, :] /= W
    diff[:, 1, :, :] /= H
    occlusion_map = torch.sum(torch.abs(diff), dim=1)[0]
    occlusion_map = ((occlusion_map > 0.05).to(torch.float32).detach().cpu().numpy() * 255.0).astype(np.uint8)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    occlusion_map = cv2.erode(occlusion_map, kernel, iterations=1)
    Image.fromarray(occlusion_map).save("occlusion.png")
