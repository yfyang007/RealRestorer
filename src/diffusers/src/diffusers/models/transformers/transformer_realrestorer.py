from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..realrestorer.model_edit import Step1XEdit, Step1XParams
from ..realrestorer.state_dict_utils import inspect_realrestorer_checkpoint, load_realrestorer_state_dict, load_state_dict_into_module


class RealRestorerTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        vec_in_dim: int = 768,
        context_in_dim: int = 4096,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 19,
        depth_single_blocks: int = 38,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        theta: int = 10_000,
        qkv_bias: bool = True,
        mode: str = "flash",
        version: str = "v1.1",
        guidance_embeds: bool = False,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        params = Step1XParams(
            in_channels=in_channels,
            out_channels=out_channels,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            depth=depth,
            depth_single_blocks=depth_single_blocks,
            axes_dim=list(axes_dims_rope),
            theta=theta,
            qkv_bias=qkv_bias,
            mode=mode,
            version=version,
            guidance_embed=guidance_embeds,
            use_mask_token=use_mask_token,
        )
        self.inner_model = Step1XEdit(params)

    @property
    def device(self) -> torch.device:
        return self.inner_model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.inner_model.dtype

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(module, "enable_gradient_checkpointing") and hasattr(module, "disable_gradient_checkpointing"):
            if value:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def _broadcast_token_ids(
        self,
        token_ids: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if token_ids is None:
            return None
        if token_ids.ndim == 2:
            token_ids = token_ids.unsqueeze(0).expand(batch_size, -1, -1)
        elif token_ids.ndim == 3 and token_ids.shape[0] == 1 and batch_size > 1:
            token_ids = token_ids.expand(batch_size, -1, -1)
        elif token_ids.ndim != 3:
            raise ValueError(f"Expected token ids with ndim 2 or 3, got shape {tuple(token_ids.shape)}")
        if token_ids.shape[0] != batch_size:
            raise ValueError(f"Token id batch size mismatch: expected {batch_size}, got {token_ids.shape[0]}")
        return token_ids.to(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del joint_attention_kwargs, text_embeddings, text_mask, kwargs

        batch_size = hidden_states.shape[0]
        model_device = hidden_states.device
        model_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for RealRestorerTransformer2DModel")
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask is required for RealRestorerTransformer2DModel")
        if timestep is None:
            raise ValueError("timestep is required for RealRestorerTransformer2DModel")

        if timestep.ndim == 0:
            timestep = timestep.view(1)
        if timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        timestep = timestep.to(device=model_device, dtype=model_dtype)

        img_ids = self._broadcast_token_ids(img_ids, batch_size, model_device, model_dtype)
        txt_ids = self._broadcast_token_ids(txt_ids, batch_size, model_device, model_dtype)

        if img_ids is None:
            raise ValueError("img_ids is required for RealRestorerTransformer2DModel")
        if txt_ids is None:
            raise ValueError("txt_ids is required for RealRestorerTransformer2DModel")

        sample = self.inner_model(
            img=hidden_states,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=timestep,
            llm_embedding=encoder_hidden_states.to(device=model_device, dtype=model_dtype),
            t_vec=timestep,
            mask=prompt_embeds_mask.to(device=model_device),
            guidance=None if guidance is None else guidance.to(device=model_device, dtype=model_dtype),
        )

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)

    @classmethod
    def from_realrestorer_checkpoint(
        cls,
        load_path: str,
        mode: str = "flash",
        version: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> "RealRestorerTransformer2DModel":
        ckpt_info = inspect_realrestorer_checkpoint(load_path)
        inferred_version = "v1.0" if ckpt_info["has_scale_factor"] else "v1.1"
        final_version = inferred_version if version == "auto" else version
        model = cls(
            mode=mode,
            version=final_version,
            guidance_embeds=bool(ckpt_info["has_guidance_in"]),
            use_mask_token=bool(ckpt_info["has_mask_token"]),
        )
        model.load_realrestorer_weights(load_path, strict=False)
        return model.to(device=device, dtype=dtype)

    def load_realrestorer_weights(self, load_path: str, strict: bool = False):
        self.inner_model, summary = load_realrestorer_state_dict(
            self.inner_model,
            load_path,
            strict=strict,
            assign=True,
        )
        return summary

    def load_preconverted_weights(self, ckpt_path: str, strict: bool = True):
        _, summary = load_state_dict_into_module(
            self.inner_model,
            ckpt_path,
            device="cpu",
            strict=strict,
            assign=True,
        )
        return summary
