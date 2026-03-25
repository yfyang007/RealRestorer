from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import build_generator


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    if all(key.startswith("netG.") for key in keys):
        return {key[len("netG."):]: value for key, value in state_dict.items()}
    return state_dict


class ReflectionSynthesisInferenceModel(nn.Module):
    def __init__(
        self,
        *,
        ckpt_path: Union[str, Path, None] = None,
        input_nc: int = 6,
        output_nc: int = 3,
        ngf: int = 64,
        which_model_netG: str = "resnet_9blocks",
        norm: str = "instance",
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.netG = build_generator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            which_model_netG=which_model_netG,
            norm=norm,
            use_dropout=use_dropout,
        )
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")
        state_dict = _strip_state_dict_prefix(checkpoint)
        self.netG.load_state_dict(state_dict, strict=True)

    def forward(self, transmission: torch.Tensor, reflection: torch.Tensor):
        transmission = transmission.to(self.device)
        reflection = reflection.to(self.device)

        concat_tr = torch.cat((transmission, reflection), dim=1)
        reflection_weight = self.netG(concat_tr)
        if reflection_weight.shape[2:] != reflection.shape[2:]:
            reflection_weight = F.interpolate(
                reflection_weight,
                size=reflection.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if reflection_weight.shape[1] == 1 and reflection.shape[1] > 1:
            reflection_weight = reflection_weight.expand(-1, reflection.shape[1], -1, -1)
        elif reflection_weight.shape[1] != reflection.shape[1]:
            raise ValueError(
                "Reflection weight channel mismatch: "
                f"{reflection_weight.shape[1]} vs {reflection.shape[1]}"
            )

        mix = reflection_weight * reflection + (1.0 - reflection_weight) * transmission
        return {
            "mix": mix,
            "reflection_weight": reflection_weight,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
