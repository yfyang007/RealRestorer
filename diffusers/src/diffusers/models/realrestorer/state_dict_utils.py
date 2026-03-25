from __future__ import annotations

import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file


def load_state_dict_file(ckpt_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if Path(ckpt_path).suffix == ".safetensors":
        return load_file(ckpt_path, device=device)
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type for {ckpt_path}")
    return payload


def load_state_dict_into_module(
    module,
    ckpt_path: str,
    device: str = "cpu",
    strict: bool = False,
    assign: bool = True,
):
    state_dict = load_state_dict_file(ckpt_path, device=device)
    missing, unexpected = module.load_state_dict(state_dict, strict=strict, assign=assign)
    return module, {"missing": missing, "unexpected": unexpected}


def normalize_state_dict_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        key = key[len("model.") :]
    return key


def extract_model_state_dict(payload: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError("Unsupported checkpoint payload format")


def list_model_checkpoint_files(load_path: str) -> List[str]:
    path = Path(load_path)
    if path.is_file():
        return [str(path)]

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {load_path}")

    patterns = ("TP*.safetensors", "TP*.pt", "*.safetensors", "*.pt")
    files: List[str] = []
    for pattern in patterns:
        for file_path in sorted(glob.glob(str(path / pattern))):
            filename = os.path.basename(file_path)
            if filename.endswith("_OPT.pt"):
                continue
            if filename in {"data.pt", "data.pkl"}:
                continue
            if filename.endswith(".meta.pt"):
                continue
            files.append(file_path)
        if files:
            break

    if not files:
        raise FileNotFoundError(f"No model checkpoint files found under {load_path}")

    return files


def load_checkpoint_file(file_path: str) -> Dict[str, torch.Tensor]:
    payload = load_state_dict_file(file_path, device="cpu")
    return {
        normalize_state_dict_key(key): value
        for key, value in extract_model_state_dict(payload).items()
        if torch.is_tensor(value)
    }


def inspect_realrestorer_checkpoint(load_path: str) -> Dict[str, object]:
    shard_files = list_model_checkpoint_files(load_path)
    first_state_dict = load_checkpoint_file(shard_files[0])
    return {
        "shard_files": shard_files,
        "has_scale_factor": "connector.scale_factor" in first_state_dict,
        "has_guidance_in": "guidance_in.in_layer.weight" in first_state_dict,
        "has_mask_token": "mask_token" in first_state_dict,
    }


def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in shape)


def can_concat_to_target(shards: Sequence[torch.Tensor], target_shape: Sequence[int], dim: int) -> bool:
    if not shards:
        return False
    ndim = shards[0].ndim
    if dim >= ndim or ndim != len(target_shape):
        return False
    if any(shard.ndim != ndim for shard in shards):
        return False

    for axis in range(ndim):
        if axis == dim:
            total = sum(int(shard.shape[axis]) for shard in shards)
            if total != int(target_shape[axis]):
                return False
        elif any(int(shard.shape[axis]) != int(target_shape[axis]) for shard in shards):
            return False
    return True


def merge_sharded_tensor(
    key: str,
    shards: Sequence[torch.Tensor],
    target_shape: Sequence[int],
) -> Tuple[Optional[torch.Tensor], str]:
    if not shards:
        return None, "missing"

    if len(shards) == 1:
        shard = shards[0]
        if _shape_tuple(shard.shape) == _shape_tuple(target_shape):
            return shard, "single"
        return None, "shape_mismatch"

    first = shards[0]
    same_shape = all(_shape_tuple(shard.shape) == _shape_tuple(first.shape) for shard in shards)
    if same_shape and _shape_tuple(first.shape) == _shape_tuple(target_shape):
        if all(torch.equal(first, shard) for shard in shards[1:]):
            return first, "replicated"
        return first, "same_shape_first"

    for dim in range(first.ndim):
        if can_concat_to_target(shards, target_shape, dim):
            return torch.cat(list(shards), dim=dim), f"concat_dim_{dim}"

    if key.endswith(".bias") and _shape_tuple(first.shape) == _shape_tuple(target_shape):
        return first, "bias_first"

    return None, "shape_mismatch"


def build_realrestorer_state_dict(
    load_path: str,
    target_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    shard_files = list_model_checkpoint_files(load_path)
    buckets: Dict[str, List[torch.Tensor]] = defaultdict(list)
    source_keys = set()

    for file_path in shard_files:
        shard_state_dict = load_checkpoint_file(file_path)
        for key, value in shard_state_dict.items():
            source_keys.add(key)
            if key in target_state_dict:
                buckets[key].append(value)

    merged_state_dict: Dict[str, torch.Tensor] = {}
    merge_stats: Dict[str, int] = defaultdict(int)
    skipped_shape: Dict[str, List[Tuple[int, ...]]] = {}

    for key, target_tensor in target_state_dict.items():
        if key not in buckets:
            continue
        merged_tensor, merge_mode = merge_sharded_tensor(key, buckets[key], target_tensor.shape)
        if merged_tensor is None:
            skipped_shape[key] = [_shape_tuple(t.shape) for t in buckets[key]]
            continue
        merge_stats[merge_mode] += 1
        merged_state_dict[key] = merged_tensor.to(dtype=target_tensor.dtype)

    summary = {
        "source_file_count": len(shard_files),
        "source_key_count": len(source_keys),
        "matched_key_count": len(merged_state_dict),
        "missing_target_keys": sorted(set(target_state_dict) - set(merged_state_dict)),
        "unused_source_keys": sorted(source_keys - set(target_state_dict)),
        "skipped_shape_keys": skipped_shape,
        "merge_stats": dict(merge_stats),
    }
    return merged_state_dict, summary


def load_realrestorer_state_dict(
    module,
    load_path: str,
    strict: bool = False,
    assign: bool = True,
):
    merged_state_dict, summary = build_realrestorer_state_dict(load_path, module.state_dict())
    missing, unexpected = module.load_state_dict(merged_state_dict, strict=strict, assign=assign)
    summary = dict(summary)
    summary["missing"] = missing
    summary["unexpected"] = unexpected
    return module, summary


def _first_existing_path(candidates: Sequence[Optional[str]]) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def resolve_model_assets(
    model_path: Optional[str],
    ae_path: Optional[str],
    qwen2vl_path: Optional[str],
) -> Tuple[str, str]:
    if ae_path is None:
        ae_path = _first_existing_path(
            [
                os.path.join(model_path, "ae.safetensors") if model_path else None,
                os.path.join(model_path, "vae.safetensors") if model_path else None,
                os.path.join(model_path, "FLUX.1-dev", "ae.safetensors") if model_path else None,
                os.path.join(model_path, "FLUX.1-dev", "vae.safetensors") if model_path else None,
                os.path.join(model_path, "vae", "diffusion_pytorch_model.safetensors") if model_path else None,
            ]
        )
    if qwen2vl_path is None:
        qwen2vl_path = _first_existing_path(
            [
                os.path.join(model_path, "Qwen2.5-VL-7B-Instruct") if model_path else None,
                os.path.join(model_path, "qwen25vl-7b-instruct") if model_path else None,
                model_path
                if model_path
                and os.path.isdir(model_path)
                and os.path.isdir(os.path.join(model_path, "text_encoder"))
                and os.path.isdir(os.path.join(model_path, "processor"))
                else None,
                os.path.join(model_path, "text_encoder")
                if model_path
                and os.path.isdir(os.path.join(model_path, "text_encoder"))
                and os.path.exists(os.path.join(model_path, "text_encoder", "config.json"))
                else None,
                model_path
                if model_path and os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))
                else None,
            ]
        )

    if ae_path is None:
        raise FileNotFoundError(
            "Could not find VAE weights. Pass ae_path or use model_path containing ae.safetensors, vae.safetensors, or vae/diffusion_pytorch_model.safetensors."
        )
    if qwen2vl_path is None:
        raise FileNotFoundError(
            "Could not find the Qwen2.5-VL model. Pass qwen2vl_path or use model_path containing Qwen2.5-VL-7B-Instruct or a diffusers bundle with text_encoder/ and processor/."
        )
    return ae_path, qwen2vl_path
