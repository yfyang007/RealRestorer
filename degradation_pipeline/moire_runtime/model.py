from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Mapping, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .blending import Blending
from .uformer import Uformer


def _ensure_checkpoint_compat() -> None:
    if "pytorch_lightning" in sys.modules:
        return

    pytorch_lightning = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    model_checkpoint = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class ModelCheckpoint:
        pass

    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = model_checkpoint
    pytorch_lightning.callbacks = callbacks

    sys.modules["pytorch_lightning"] = pytorch_lightning
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = model_checkpoint


class MoireBlendingInferenceModel(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        network_config: Mapping[str, Any],
        ckpt_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.network_config = dict(network_config)
        self.init_blending_args = self.network_config["init_blending_args"]
        self.blending_network_args = self.network_config["blending_network_args"]

        if self.model_name != "UniDemoire":
            raise ValueError(f"Unsupported moire model: {self.model_name}")

        self.init_blend = Blending(self.init_blending_args)
        self.refine_net = Uformer(
            embed_dim=self.blending_network_args["embed_dim"],
            depths=self.blending_network_args["depths"],
            win_size=self.blending_network_args["win_size"],
            modulator=self.blending_network_args["modulator"],
            shift_flag=self.blending_network_args["shift_flag"],
        )

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        *,
        ckpt_path: Union[str, Path, None] = None,
    ) -> "MoireBlendingInferenceModel":
        config = OmegaConf.load(config_path)
        model_params = config.model.params
        return cls(
            model_name=model_params.model_name,
            network_config=model_params.network_config,
            ckpt_path=ckpt_path,
        )

    def load_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        _ensure_checkpoint_compat()
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict, strict=False)

    def forward(self, moire_pattern: torch.Tensor, natural: torch.Tensor, real_moire: torch.Tensor):
        moire_pattern = moire_pattern.to(self.device)
        natural = natural.to(self.device)
        real_moire = real_moire.to(self.device)

        mib_result, _ = self.init_blend(natural, moire_pattern)
        refine_result = mib_result * self.refine_net(mib_result, real_moire)
        min_val = torch.min(refine_result)
        max_val = torch.max(refine_result)
        refine_result = (refine_result - min_val) / (max_val - min_val + 1e-6)
        return mib_result, refine_result

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
