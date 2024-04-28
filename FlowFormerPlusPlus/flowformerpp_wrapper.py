import torch
import numpy as np
from FlowFormerPlusPlus.configs.submissions import get_cfg
from FlowFormerPlusPlus.core.FlowFormer import build_flowformer
from FlowFormerPlusPlus.core.utils.flow_viz import flow_to_image


def build_flow_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = build_flowformer(cfg)
    state_dict = {key[7:]: val for key, val in torch.load(cfg.model, map_location="cpu").items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def viz_flow(flows):
    flows = flows.detach().permute(0, 2, 3, 1).cpu().numpy()
    flow_imgs = torch.stack([torch.tensor(np.transpose(flow_to_image(flow), (2, 0, 1)).astype(np.float32) / 255.0) for flow in flows])
    return flow_imgs
