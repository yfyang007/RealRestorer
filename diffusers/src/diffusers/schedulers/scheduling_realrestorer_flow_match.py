from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional, Sequence

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)

    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


class RealRestorerFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> None:
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self._begin_index = 0

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        timesteps: Optional[Sequence[float]] = None,
        sigmas: Optional[Sequence[float]] = None,
        image_seq_len: Optional[int] = None,
    ) -> None:
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of timesteps or sigmas can be provided.")
        if timesteps is not None:
            values = [float(t) for t in timesteps]
        elif sigmas is not None:
            values = [float(t) for t in sigmas]
            if not values or values[-1] != 0.0:
                values.append(0.0)
        else:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required when timesteps and sigmas are not provided.")
            if image_seq_len is None:
                raise ValueError("image_seq_len is required for RealRestorerFlowMatchScheduler.")
            values = get_schedule(
                num_inference_steps,
                image_seq_len,
                base_shift=self.config["base_shift"] if isinstance(self.config, dict) else self.config.base_shift,
                max_shift=self.config["max_shift"] if isinstance(self.config, dict) else self.config.max_shift,
                shift=self.config["shift"] if isinstance(self.config, dict) else self.config.shift,
            )
        self.timesteps = torch.tensor(values, device=device, dtype=torch.float32)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        if self.timesteps.numel() == 0:
            raise RuntimeError("Call set_timesteps before step.")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(float(timestep), device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep_values = self.timesteps.to(device=sample.device, dtype=sample.dtype)
        index = int(torch.argmin(torch.abs(timestep_values - timestep.reshape(()))).item())
        next_index = min(index + 1, timestep_values.numel() - 1)
        prev_timestep = timestep_values[next_index]
        prev_sample = sample + (prev_timestep - timestep) * model_output

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
