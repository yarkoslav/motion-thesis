import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_kmeans import KMeans
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from utils import instantiate_from_config
from omegaconf import OmegaConf
import robust_loss_pytorch.general


def convert_flow(flow):
    _, _, H, W = flow.shape
    x, y = flow.permute(1, 0, 2, 3)
    x = x + torch.arange(W).view(1, 1, -1).to(flow.device)
    y = y + torch.arange(H).view(1, -1, 1).to(flow.device)
    x = x / W
    y = y / H
    flow = torch.stack([x, y], dim=1)
    flow = 2 * flow - 1
    return flow


def extract_flow_patches(flow, patch_size):
    B, C, H, W = flow.shape # [B, C, H, W]
    flow = flow.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # [B, C, H/ps, W/ps, ps, ps]
    flow = flow.reshape(B, C, -1, patch_size, patch_size) # [B, C, num_patches, ps, ps]
    flow = flow.view(*flow.shape[:3], -1) # [B, C, num_patches, ps**2]
    flow = flow.permute(0, 2, 3, 1) # [B, num_patches, ps**2, C]
    flow = flow.reshape(-1, *flow.shape[2:]) # [B * num_patches, ps**2, C
    return flow


def flow_kmeans(flow, num_flows):
    # flow in shape [B * num_patches, ps**2, C]
    B, S, C = flow.shape
    model = KMeans(n_clusters=num_flows, seed=42)
    result = model(flow)

    # sort clusters according to the number of belonging points
    # labels = []
    # for label in range(num_flows):
    #     labels.append(torch.sum(result.labels == label, dim=1))
    # labels = torch.stack(labels, dim=1) # [B * num_patches, num_flows]
    # order = torch.argsort(labels, dim=1, descending=True) # [B * num_patches, num_flows]
    # order_expanded = order.unsqueeze(-1).expand(-1, -1, C) # [B * num_patches, num_flows, C]
    # centers = torch.gather(result.centers, 1, order_expanded) # [B*num_patches, num_flows, C]

    # without sorting
    centers = result.centers # [B*num_patches, num_flows, C]
    return centers


def calculate_flow_loss(gt_flows, flows):
    # gt_flows - [num_flows, B, 2, H, W], flows - list([B, 2, H, W]), len = num_flows
    flows = torch.stack(flows)
    num_flows, B, _, H, W = gt_flows.shape
    gt_flows = gt_flows.permute(1, 3, 4, 0, 2).reshape(-1, 1, num_flows, 2) # [B*H*W, 1, num_flows, 2]
    flows = flows.permute(1, 3, 4, 0, 2).reshape(-1, 1, num_flows, 2) # [B*H*W, 1, num_flows, 2]

    permutations = itertools.permutations(range(num_flows))
    permutations = torch.tensor(list(permutations)).to(gt_flows.device) # [num_permutations, num_flows]
    num_permutations = permutations.shape[0]

    gt_flows_expanded = gt_flows.repeat(1, num_permutations, 1, 1) # [B*H*W, num_permutations, num_flows, 2]
    flows_expanded = flows.repeat(1, num_permutations, 1, 1) # [B*H*W, num_permutations, num_flows, 2]
    permutations_expanded = permutations.view(1, num_permutations, num_flows, 1).expand(B*H*W, -1, -1, 2) # [B*H*W, num_permutations, num_flows, 2]
    gt_flows_permuted = torch.gather(gt_flows_expanded, dim=2, index=permutations_expanded) # [B*H*W, num_permutations, num_flows, 2]
    loss = F.mse_loss(flows_expanded, gt_flows_permuted, reduction="none").mean(dim=[2, 3]) # [B*H*W, num_permutations]
    loss = torch.mean(torch.min(loss, dim=1)[0])
    return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LFM(nn.Module):
    def __init__(self,
                 temp_dim=32,
                 num_flows=4):
        super().__init__()
        self.num_flows = num_flows
        self.flow_net = nn.Sequential(
            ConvBlock(in_channels=4, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=temp_dim, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=temp_dim, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=temp_dim, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=temp_dim, out_channels=2 * self.num_flows, kernel_size=1, stride=1)
        )

        self.weight_net = nn.Sequential(
            ConvBlock(in_channels=self.num_flows * (4 + 2), out_channels=temp_dim, kernel_size=1, stride=1),
            ConvBlock(in_channels=temp_dim, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=temp_dim, out_channels=temp_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=temp_dim, out_channels=self.num_flows, kernel_size=1, stride=1)
            nn.Conv2d(in_channels=temp_dim, out_channels=self.num_flows * 4, kernel_size=1, stride=1)
        )

    def forward(self, image, flow, gt_flows=None):
        # image - [B, C, H, W], flow - [B, C, H, W], gt_flows - list([B, 2, H, W]), len = num_flows
        flows = self.flow_net(flow)  # [B, 2*num_flows, H, W]
        flows = flows.split(2, dim=1)  # list([B, 2, H, W]), len = num_flows
        warp_flows = flows

        images_warped = []
        for flow in warp_flows:
            flow = convert_flow(flow) # normalize flow to [-1, 1] range
            flow = flow.permute(0, 2, 3, 1) # [B, H, W, 2]
            images_warped.append(F.grid_sample(image, flow, mode="nearest"))
        # image_warped - list([B, C, H, W]), len = num_flows

        weight_feats = torch.cat([*images_warped, *flows], dim=1) # [B, num_flows * (C+2), H, W]
        weights = self.weight_net(weight_feats) # [B, num_flows, H, W] or [B, C*num_flows, H, W]

        # general weight prediction
        # weights = torch.softmax(weights, dim=1) # [B, num_flows, H, W]
        # weights = weights.permute(1, 0, 2, 3) # [num_flows, B, H, W]

        # channel-wise weight prediction
        weights = weights.split(self.num_flows, dim=1)  # list([B, num_flows, H, W]), len = C
        weights = torch.stack([torch.softmax(weight, dim=1) for weight in weights], dim=-1) # [B, num_flows, H, W, C]
        weights = weights.permute(1, 0, 4, 2, 3) # [num_flows, B, C, H, W]

        out = torch.zeros_like(image) # [B, C, H, W]
        for image_warped, weight in zip(images_warped, weights):
            out += image_warped * weight
        return out, flows


class MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 d_in,
                 d_out,
                 num_head=8):
        super().__init__()
        self.d_model = d_model
        self.d_in = d_in
        self.num_head = num_head

        self.hidden_dim = d_model // num_head
        self.T = self.hidden_dim**0.5

        self.linear_Q = nn.Linear(d_in["Q"], d_model)
        self.linear_K = nn.Linear(d_in["K"], d_model)
        self.linear_V = nn.Linear(d_in["V"], d_model)

        self.projection = nn.Linear(d_model, d_out)


    def forward(self, Q, K, V):
        # Q - [B, T, C_Q], K - [B, T, C_K], V = [B, T, C_V]
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        B = Q.shape[0]

        # Linear projections
        Q = self.linear_Q(Q) # [B, T, C]
        K = self.linear_K(K) # [B, T, C]
        V = self.linear_V(V) # [B, T, C]

        # Scale
        Q = Q / self.T

        # Multi-head
        Q = Q.view(B, -1, num_head, hidden_dim).permute(0, 2, 1, 3) # [B, T, C] -> [B, T, num_head, hidden_dim] -> [B, num_head, T, hidden_dim]
        K = K.view(B, -1, num_head, hidden_dim).permute(0, 2, 3, 1) # [B, T, C] -> [B, T, num_head, hidden_dim] -> [B, num_head, hidden_dim, T]
        V = V.view(B, -1, num_head, hidden_dim).permute(0, 2, 1, 3) # [B, T, C] -> [B, T, num_head, hidden_dim] -> [B, num_head, T, hidden_dim]

        # Multiplication
        QK = Q @ K # [B, num_head, T, T]

        # Activation
        attn = torch.softmax(QK, dim=-1)

        # Weighted sum
        outputs = attn @ V # [B, num_head, T, h_dim]

        # Restore shape
        outputs = outputs.permute(0, 2, 1, 3).reshape(B, -1, self.d_model) # [B, num_head, T, h_dim] -> [B, T, num_head, hidden_dim] -> [B, T, C]
        outputs = self.projection(outputs) # [B, T, C_V]
        return outputs


class WarpAttention(nn.Module):
    def __init__(self,
                 attn_params,
                 latent_shape=(1024, 4),
                 hidden_dim=256):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.randn(*latent_shape).unsqueeze(0))
        self.norm_feat = nn.LayerNorm(latent_shape[-1])
        self.norm_pos = nn.LayerNorm(latent_shape[-1] * 2)
        self.attn = MultiheadAttention(**attn_params)
        self.flow_map = nn.Sequential(
            nn.Linear(latent_shape[-1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_shape[-1])
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_shape[-1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_shape[-1])
        )
        self.feat_map = nn.Sequential(
            nn.Linear(latent_shape[-1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_shape[-1])
        )
        self._init_weights()

    def forward(self, img1_feat, img2_feat, flow_feat):
        # img1_feat - [B, T, C], img2_feat - [B, T, C], img3_feat - [B, T ,C]
        B = img1_feat.shape[0]

        # add positional embedding
        pos_emb = self.pos_emb.repeat((B, 1, 1))
        flow_feat = self.flow_map(flow_feat)
        pos_emb_warped = pos_emb + flow_feat
        img1_pos = torch.cat([self.feat_map(img1_feat), pos_emb_warped], dim=-1) # [B, T, 2*C]
        img2_pos = torch.cat([self.feat_map(img2_feat), pos_emb], dim=-1) # [B, T, 2*C]
        # img1_pos = pos_emb_warped
        # img2_pos = pos_emb

        # normalize
        img1_pos, img2_pos = self.norm_pos(img1_pos), self.norm_pos(img2_pos)

        # multi-head attention
        _img2_feat = self.attn(img2_pos, img1_pos, img1_feat) # [B, T, C]

        # add & normalize
        img2_feat = img2_feat + _img2_feat
        _img2_feat = self.norm_feat(img2_feat)

        # feed forward net
        _img2_feat = self.feed_forward(_img2_feat)

        # add
        img2_feat = img2_feat + _img2_feat
        return img2_feat

    def _zero_init(self, module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def _init_weights(self):
        self._zero_init(self.flow_map)
        self._zero_init(self.feat_map)


class AttentionLFM(nn.Module):
    def __init__(self,
                 warp_attn_params,
                 num_att_blocks,
                 latent_dim=4):
        super().__init__()

        attn_blocks = []
        for _ in range(num_att_blocks):
            attn_blocks.append(WarpAttention(**warp_attn_params))
        self.attn_blocks = nn.Sequential(*attn_blocks)
        self.query = nn.Parameter(torch.randn(1, latent_dim, 1, 1))

    def forward(self, img1_feat, flow_feat):
        # img1_feat - [B, C, H, W], img2_feat - [B, C, H, W], img3_feat - [B, C, H, W]
        B, C, H, W = img1_feat.shape
        img2_feat = self.query.repeat(B, 1, H, W)

        img1_feat, img2_feat, flow_feat = self.to_attn(img1_feat), self.to_attn(img2_feat), self.to_attn(flow_feat)
        img2_feats = []
        for attn_block in self.attn_blocks:
            img2_feat = attn_block(img1_feat, img2_feat, flow_feat)
            img2_feats.append(img2_feat)
        img2_feats = [self.from_attn(_img2_feat, [H, W]) for _img2_feat in img2_feats]
        img2_feat = self.from_attn(img2_feat, [H, W])
        return img2_feat, None

    def to_attn(self, x):
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, -1, C)

    def from_attn(self, x, shape):
        B, T, C = x.shape
        return x.reshape(B, *shape, C).permute(0, 3, 1, 2)



class LWM(pl.LightningModule):
    def __init__(self,
                 image_config,
                 image_ckpt,
                 flow_config,
                 flow_ckpt,
                 lfm_config,
                 mode,
                 monitor=None,
                 ):
        super().__init__()
        self.image_vae = self.load_vae(image_config, image_ckpt).eval()
        self.image_vae.freeze()

        self.flow_vae = self.load_vae(flow_config, flow_ckpt).eval()
        self.flow_vae.freeze()
        self.mode = mode
        if self.mode == "warp":
            self.LFM = LFM(**lfm_config)
        else:
            self.LFM = AttentionLFM(**lfm_config)

        # self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=4, float_dtype=torch.float32, device="cuda:0")
        if monitor is not None:
            self.monitor = monitor

    def save_image(self, image, path):
        grid = torchvision.utils.make_grid(image, nrow=4)
        # grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.detach().cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        Image.fromarray(grid).save(path)

    def forward(self, input):
        image1, image2 = input["image1"] / 255.0, input["image2"] / 255.0
        with torch.no_grad():
            z1, z2 = self.image_vae.encode(image1).mode(), self.image_vae.encode(image2).mode()
            image1, image2 = image1 * 255.0, image2 * 255.0
            gt_flow = self.flow_vae.flow_model(image2, image1)[0]
            gt_flow_forw = self.flow_vae.flow_model(image1, image2)[0]
            z_flow = self.flow_vae.encode(gt_flow).mode()

            B, C, H, W = gt_flow.shape
            H, W = H // 8, W // 8
            occlusion_map = self.get_occlusion_map(gt_flow_forw, gt_flow)

        if self.mode == "warp":
            if self.LFM.num_flows == 1:
                # gt flow via interpolation
                gt_flows = F.interpolate(gt_flow / 8.0, size=z_flow.shape[2:]).unsqueeze(0) # [num_flows, B, C, H, W]
            else:
                # gt flow via k-means
                gt_flow_patches = extract_flow_patches(gt_flow, 8)  # [B * num_patches, ps**2, C]
                gt_flow_centroids = flow_kmeans(gt_flow_patches, self.LFM.num_flows)  # [B*num_patches, num_flows, C]
                gt_flows = gt_flow_centroids.reshape(B, H, W, -1, C).permute(3, 0, 4, 1, 2)  # [num_flows, B, C, H, W]
                gt_flows = gt_flows / 8.0

            z1_warped, flows = self.LFM(z1, z_flow, gt_flows)
        else:
            z1_warped, z1_warped_list = self.LFM(z1, z_flow)
        # return z1_warped, z2, z1_warped, (None, None), z1_warped_list, occlusion_map
        return z1_warped, z2, z1_warped, (gt_flows, flows), None, occlusion_map, gt_flow

    def loss(self, z1_warped, z2, gt_flows, flows, z1_warped_list, occlusion_map):
        mse_loss = 0
        if occlusion_map is not None:
            occlusion_map = F.interpolate(occlusion_map, size=z1_warped.shape[2:], mode="nearest")
            z1_warped = z1_warped * (1 - occlusion_map)
            z2 = z2 * (1 - occlusion_map)
        if z1_warped_list is None:
            # mse loss
            # mse_loss = F.mse_loss(z1_warped, z2)

            # robust loss
            B, C, H, W = z1_warped.shape
            mse_loss = torch.mean(self.adaptive.lossfun((z2 - z1_warped).permute(0, 2, 3, 1).reshape(-1, C)))
        else:
            gamma = 0.75
            cur_weight = 1
            normalizer = 0
            for z1_warped in z1_warped_list[::-1]:
                cur_weight *= gamma
                cur_mse_loss = F.mse_loss(z1_warped, z2)
                mse_loss += cur_mse_loss * cur_weight
                normalizer += cur_weight
            mse_loss = mse_loss / normalizer
        flow_loss = 0
        if gt_flows is not None:
            flow_loss = calculate_flow_loss(gt_flows, flows)
        return mse_loss, flow_loss

    def training_step(self, batch, batch_idx):
        z1_warped, z2, z1_warped_viz, (gt_flows, flows), z1_warped_list, occlusion_map = self(batch)
        mse_loss, flow_loss = self.loss(z1_warped, z2, gt_flows, flows, z1_warped_list, occlusion_map)
        total_loss = mse_loss + flow_loss
        image1_warped_recon, image2_recon = self.image_vae.decode(z1_warped), self.image_vae.decode(z2)
        if occlusion_map is not None:
            image1_warped_recon, image2_recon = image1_warped_recon * (1 - occlusion_map), image2_recon * (1 - occlusion_map)
        recon_loss = F.mse_loss(image1_warped_recon, image2_recon)

        self.log("total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("mse_loss", mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("flow_loss", flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("recon_loss", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        z1_warped, z2, z1_warped_viz, (gt_flows, flows), z1_warped_list, occlusion_map = self(batch)
        mse_loss, flow_loss = self.loss(z1_warped, z2, gt_flows, flows, z1_warped_list, occlusion_map)
        total_loss = mse_loss + flow_loss
        image1_warped_recon, image2_recon = self.image_vae.decode(z1_warped), self.image_vae.decode(z2)
        if occlusion_map is not None:
            image1_warped_recon, image2_recon = image1_warped_recon * (1 - occlusion_map), image2_recon * (1 - occlusion_map)
        recon_loss = F.mse_loss(image1_warped_recon, image2_recon)

        self.log("val/total_loss", total_loss)
        self.log("val/mse_loss", mse_loss)
        self.log("val/flow_loss", flow_loss)
        self.log("val/recon_loss", recon_loss)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.LFM.parameters(), lr=lr)
        return opt

    def load_vae(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(ckpt, strict=False)
        return model

    def get_occlusion_map(self, flow12, flow21, threshold=0.05):
        B, _, H, W = flow12.shape
        x_coord = torch.arange(W).view(1, 1, 1, -1).repeat(B, 1, H, 1).to(flow12.device)
        y_coord = torch.arange(H).view(1, 1, -1, 1).repeat(B, 1, 1, W).to(flow12.device)
        coord = torch.cat([x_coord, y_coord], dim=1)
        flow_coord = coord + flow12
        warped_coord = F.grid_sample(flow_coord, convert_flow(flow21).permute(0, 2, 3, 1), mode="nearest")
        diff = coord - warped_coord
        diff[:, 0, :, :] /= W
        diff[:, 1, :, :] /= H
        occlusion_map = torch.sum(torch.abs(diff), dim=1, keepdim=True)
        occlusion_map = (occlusion_map > threshold).to(torch.float32)
        return occlusion_map

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = {key: val.to(self.device) for key, val in batch.items()}
        if not only_inputs:
            z1_warped, z2, z1_warped_viz, (gt_flows, flows), z1_warped_list, occlusion_map = self(x)
            image1_warped = self.image_vae.decode(z1_warped_viz)
            image2_recon = self.image_vae.decode(z2)
            log["occlusions"] = occlusion_map
            log["image1_warped"] = image1_warped
            log["image2_recon"] = image2_recon
        log["image1"] = x["image1"] / 255.0
        log["image2"] = x["image2"] / 255.0
        return log