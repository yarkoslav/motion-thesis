import torch
import numpy as np
import pytorch_lightning as pl
from utils import instantiate_from_config
from modules.model import Encoder, Decoder
from modules.distributions import DiagonalGaussianDistribution
from FlowFormerPlusPlus.flowformerpp_wrapper import build_flow_model, viz_flow


# def encode_flow(x):
#     B, _, H, W = x.shape
#     homogeneous_x = torch.cat([x, torch.ones((B, 1, H, W)).to(x.device)], dim=1)
#     projection_on_the_sphere = homogeneous_x / torch.sum(homogeneous_x**2, dim=1, keepdim=True)
#     return projection_on_the_sphere
#
#
# def decode_flow(x):
#     flow = x[:, :2, :, :] / x[:, 2:, :, :]
#     return flow


def encode_flow(flow, d=1):
    B, _, H, W = flow.shape
    homogeneous_flow = torch.cat([flow, d*torch.ones((B, 1, H, W)).to(flow.device)], dim=1)  # lifting up vectors
    projection_on_the_sphere = homogeneous_flow / torch.sum(homogeneous_flow**2, dim=1, keepdim=True)  # projecting point on the unit sphere
    x, y, z = projection_on_the_sphere.permute(1, 0, 2, 3)  # splitting up and getting x, y, z coordinates
    theta = (torch.atan2(y, x) / np.pi + 1) / 2 # theta will be in range [0, 1]
    phi = torch.atan2(z, torch.sqrt(x**2 + y**2)) / (np.pi / 2)  # phi will be in range [0, 1]
    spherical_encoding = torch.stack([theta, phi], dim=1)  # spherical encoding of flow
    return spherical_encoding


def decode_flow(x, d=1):
    theta, phi = x.permute(1, 0, 2, 3)  # splitting up and getting theta, phi
    theta, phi = (2 * theta - 1) * np.pi, phi * (np.pi / 2)  # theta in [-pi, pi], phi in [0, pi/2]
    x, y, z = d * torch.cos(phi) * torch.cos(theta), d * torch.cos(phi) * torch.sin(theta), d*torch.sin(phi)  # getting our coordinates on the sphere
    scaling_factor = d / z  # scaling factor to reproject back from the sphere onto the flow plane
    x, y = x * scaling_factor, y * scaling_factor  # reprojection
    flow = torch.stack([x, y], dim=1)
    return flow


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None,
                 mode="image",
                 use_flow_encoding=False,
                 trigonometric_loss=False
                 ):
        super().__init__()
        self.mode = mode
        self.use_flow_encoding = use_flow_encoding
        self.trigonometric_loss = trigonometric_loss
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        if self.mode == "flow":
            self.flow_model = build_flow_model()

        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # if self.mode == "flow" and self.use_flow_encoding:
        #     x = encode_flow(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # if self.mode == "flow" and self.use_flow_encoding:
            # dec = decode_flow(dec)
            # dec = dec / torch.sum(dec**2, dim=1, keepdim=True)
        return dec

    def forward(self, input, sample_posterior=True):
        if self.mode == "flow":
            image1, image2 = input["image1"], input["image2"]
            with torch.no_grad():
                flow = self.flow_model(image1, image2)[0]
            input = flow
            if self.use_flow_encoding:
                input = encode_flow(input)

        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return input, dec, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        input, reconstructions, posterior = self(batch)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(input, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", trigonometric_loss=self.trigonometric_loss)
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(input, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train", trigonometric_loss=self.trigonometric_loss)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        input, reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(input, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", trigonometric_loss=self.trigonometric_loss)
        discloss, log_dict_disc = self.loss(input, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", trigonometric_loss=self.trigonometric_loss)
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        if self.mode == "image":
            log = dict()
            x = batch.to(self.device)
            if not only_inputs:
                _, xrec, posterior = self(x)
                log["samples"] = self.decode(torch.randn_like(posterior.sample()))
                log["reconstructions"] = xrec
            log["inputs"] = x
        if self.mode == "flow":
            log = dict()
            x = {key: val.to(self.device) for key, val in batch.items()}
            if not only_inputs:
                flow_gt, flow_rec, posterior = self(x)
                sampled_flow = self.decode(torch.randn_like(posterior.sample()))
                if self.use_flow_encoding:
                    flow_gt, flow_rec, sampled_flow = decode_flow(flow_gt), decode_flow(flow_rec), decode_flow(sampled_flow)
                log["flow_samples"] = viz_flow(sampled_flow)
                log["flow_reconstructions"] = viz_flow(flow_rec)
                log["flow_gt"] = viz_flow(flow_gt)
            log["image"] = x["image1"] / 255.0
        return log

    def _freeze_model_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def freeze(self):
        self._freeze_model_params(self.encoder)
        self._freeze_model_params(self.decoder)
        self.quant_conv.requires_grad = False
        # self.posterior.requires_grad = False
        if self.mode == "flow":
            self._freeze_model_params(self.flow_model)
