from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Prefer the vendored diffusers checkout when running from this repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIFFUSERS_SRC = _REPO_ROOT / "diffusers" / "src"
if _LOCAL_DIFFUSERS_SRC.is_dir() and str(_LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIFFUSERS_SRC))

import torch

if TYPE_CHECKING:
    from diffusers import RealRestorerPipeline

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def export_bundle_from_source(
    realrestorer_load: str,
    save_dir: str,
    model_path: str | None = None,
    ae_path: str | None = None,
    qwen2vl_path: str | None = None,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    mode: str = "flash",
    version: str = "auto",
    model_guidance: float = 3.5,
    max_length: int = 640,
    safe_serialization: bool = True,
) -> Path:
    from diffusers import RealRestorerPipeline

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pipe = RealRestorerPipeline.from_realrestorer_sources(
        realrestorer_load=realrestorer_load,
        model_path=model_path,
        ae_path=ae_path,
        qwen2vl_path=qwen2vl_path,
        device=device,
        dtype=torch_dtype,
        mode=mode,
        version=version,
        model_guidance=model_guidance,
        max_length=max_length,
    )

    pipe.to("cpu")
    pipe.save_pretrained(save_path, safe_serialization=safe_serialization)

    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a self-contained RealRestorer diffusers bundle using the installed diffusers package."
    )
    parser.add_argument("--load", type=str, required=True, help="Original RealRestorer checkpoint directory or file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Export directory.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional shared model root.")
    parser.add_argument("--ae_path", type=str, default=None, help="Explicit VAE weights path.")
    parser.add_argument("--qwen2vl_path", type=str, default=None, help="Explicit Qwen2.5-VL model path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device used while assembling the bundle.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
        help="Load dtype for the temporary assembly pipeline.",
    )
    parser.add_argument("--mode", type=str, default="flash", help="Attention mode.")
    parser.add_argument(
        "--version",
        type=str,
        default="auto",
        choices=["auto", "v1.0", "v1.1"],
        help="Checkpoint version.",
    )
    parser.add_argument("--model_guidance", type=float, default=3.5, help="Internal guidance embedding value.")
    parser.add_argument("--max_length", type=int, default=640, help="Qwen max token length.")
    parser.add_argument(
        "--disable_safe_serialization",
        action="store_true",
        help="Save weights with PyTorch binaries instead of safetensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = export_bundle_from_source(
        realrestorer_load=args.load,
        save_dir=args.save_dir,
        model_path=args.model_path,
        ae_path=args.ae_path,
        qwen2vl_path=args.qwen2vl_path,
        device=args.device,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        mode=args.mode,
        version=args.version,
        model_guidance=args.model_guidance,
        max_length=args.max_length,
        safe_serialization=not args.disable_safe_serialization,
    )
    print(f"Saved RealRestorer diffusers bundle to {save_path}")


if __name__ == "__main__":
    main()
