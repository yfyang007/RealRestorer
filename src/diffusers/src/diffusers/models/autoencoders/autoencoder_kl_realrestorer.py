from __future__ import annotations

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput
from ..realrestorer.autoencoder import AutoEncoder
from ..realrestorer.state_dict_utils import load_state_dict_into_module


class RealRestorerAutoencoderKL(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, int, int, int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 16,
        scale_factor: float = 0.3611,
        shift_factor: float = 0.1159,
    ) -> None:
        super().__init__()
        self.inner_model = AutoEncoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=list(ch_mult),
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
        )
        self.latent_channels = z_channels
        self.scaling_factor = scale_factor
        self.shift_factor = shift_factor

    @property
    def device(self) -> torch.device:
        return self.inner_model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.inner_model.dtype

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        latents = self.inner_model.encode(x)
        if not return_dict:
            return (latents,)
        return AutoencoderKLOutput(latent_dist=latents)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        sample = self.inner_model.decode(z)
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def load_weights(self, ckpt_path: str, strict: bool = True):
        _, summary = load_state_dict_into_module(
            self.inner_model,
            ckpt_path,
            device="cpu",
            strict=strict,
            assign=True,
        )
        return summary
