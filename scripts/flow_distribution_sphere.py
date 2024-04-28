import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from data.video_dataset import WebVidVal
from FlowFormerPlusPlus.flowformerpp_wrapper import build_flow_model, viz_flow


def encode_flow(flow, d=1):
    B, C = flow.shape
    homogeneous_flow = torch.cat([flow, d * torch.ones((B, 1)).to(flow.device)], dim=1)  # lifting up vectors
    projection_on_the_sphere = homogeneous_flow / torch.sum(homogeneous_flow ** 2, dim=1,
                                                            keepdim=True)  # projecting point on the unit sphere
    x, y, z = projection_on_the_sphere.permute(1, 0)  # splitting up and getting x, y, z coordinates
    theta = (torch.atan2(y, x) / np.pi + 1) / 2  # theta will be in range [0, 1]
    phi = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2)) / (np.pi / 2)  # phi will be in range [0, 1]
    spherical_encoding = torch.stack([theta, phi], dim=1)  # spherical encoding of flow
    return spherical_encoding


def generate_grid(start_value, end_value, grid_size):
    grid = np.exp(np.linspace(np.log(start_value), np.log(end_value), grid_size))
    return grid


# def kl_divergence(samples):
#     hist, _  = torch.histogramdd(samples.detach().cpu(), bins=[250, 250], range=[0.0, 1.0, 0.0, 1.0], density=True)
#     uniform_distribution = torch.ones_like(hist) / hist.size(0)
#     kl_divergence = F.kl_div(F.log_softmax(hist.view(-1), dim=0), F.softmax(uniform_distribution.view(-1), dim=0), reduction='sum')
#     return kl_divergence


def kl_divergence(samples):
    theta, phi = samples.permute(1, 0).detach().cpu()
    hist, _ = torch.histogram(phi, bins=1000, range=(0.0, 1.0), density=True)
    uniform_distribution = torch.ones_like(hist) / hist.size(0)
    kl_divergence = F.kl_div(F.log_softmax(hist, dim=0), F.softmax(uniform_distribution, dim=0), reduction='sum')
    return kl_divergence


if __name__ == "__main__":
    flow_model = build_flow_model().cuda()
    flow_model.eval()
    dataset = WebVidVal(video_params={"input_res": 256},
                        data_dir="/datasets1/romanus/webvid/data",
                        metadata_path="/datasets1/romanus/webvid/results_01M_train.csv",
                        num_val_videos=10000)
    with torch.no_grad():
        indices = np.random.choice(len(dataset), 1000, replace=False)
        flows = []
        for i in tqdm(indices):
            sample = dataset.__getitem__(i)
            image1, image2 = sample["image1"].unsqueeze(0).cuda(), sample["image2"].unsqueeze(0).cuda()
            flow = flow_model(image1, image2)[0]  # [B, C, H, W]
            flow = flow.permute(0, 2, 3, 1).reshape((-1, 2))  # [B*H*W, C]
            # flow = flow.detach().cpu().numpy()
            flows.append(flow)
        flows = torch.cat(flows, dim=0)
        # d_grid = generate_grid(start_value=1e-2, end_value=1e2, grid_size=500)
        d_grid = generate_grid(start_value=1e0, end_value=1e2, grid_size=500)
        min_kl, min_d = 1e10, 0
        for d in d_grid:
            spherical_encoding = encode_flow(flows, d)
            kl = kl_divergence(spherical_encoding)
            if kl < min_kl:
                min_kl = kl
                min_d = d
            print(f"for lift parameter {d} kl divergence is {kl}")
        print(f"minimum kl divergence {min_kl} is for lift parameter {min_d}")

        # max_var, max_d = 0, 0
        # for d in d_grid:
        #     spherical_encoding = encode_flow(flows, d)
        #     var = torch.var(spherical_encoding)
        #     if var > max_var:
        #         max_var = var
        #         max_d = d
        #     print(f"for lift parameter {d} variance is {var}")
        # print(f"maximum variance {max_var} is for lift parameter {max_d}")

        # flows = np.concatenate(flows, axis=0)
        # x, y = flow[:, 0], flow[:, 1]
        # plt.scatter(x, y)
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title('Sample Plot')
        # plt.savefig('sample_plot.png')
        # exit(0)
