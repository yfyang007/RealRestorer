from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Prefer the vendored diffusers checkout when running from this repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIFFUSERS_SRC = _REPO_ROOT / "diffusers" / "src"
if _LOCAL_DIFFUSERS_SRC.is_dir() and str(_LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIFFUSERS_SRC))

import torch
from PIL import Image

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

DEFAULT_T2I_NEGATIVE_PROMPT = (
    "worst quality, wrong limbs, unreasonable limbs, normal quality, "
    "low quality, low res, blurry, text, watermark, logo, banner, "
    "extra digits, cropped, jpeg artifacts, signature, username, "
    "error, sketch ,duplicate, ugly, monochrome, horror, geometry, "
    "mutation, disgusting"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RealRestorer inference through the installed diffusers integration."
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Original RealRestorer checkpoint directory or file. Only needed when loading from source weights.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Packaged repo path for direct inference, or shared asset root when used together with --load.",
    )
    parser.add_argument("--ae_path", type=str, default=None, help="Explicit VAE weights path.")
    parser.add_argument("--qwen2vl_path", type=str, default=None, help="Explicit Qwen2.5-VL model path.")

    parser.add_argument("--prompt", type=str, required=True, help="Generation or editing prompt.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--image", type=str, default=None, help="Reference image path. Omit for text-to-image.")
    parser.add_argument("--output", type=str, required=True, help="Output image path.")

    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size_level", type=int, default=1024, help="Resize target for edit.")
    parser.add_argument("--height", type=int, default=1024, help="Output height for t2i.")
    parser.add_argument("--width", type=int, default=1024, help="Output width for t2i.")

    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
        help="Inference dtype.",
    )
    parser.add_argument("--mode", type=str, default="flash", help="Attention mode for source loading.")
    parser.add_argument(
        "--version",
        type=str,
        default="auto",
        choices=["auto", "v1.0", "v1.1"],
        help="Checkpoint version for source loading.",
    )
    parser.add_argument("--model_guidance", type=float, default=3.5, help="Internal guidance embedding value.")
    parser.add_argument("--max_length", type=int, default=640, help="Qwen max token length.")
    return parser.parse_args()


def configure_pipeline_memory(pipe: Any, device: str) -> None:
    if str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)


def main() -> None:
    args = parse_args()
    torch_dtype = DTYPE_MAP[args.torch_dtype]
    from diffusers import RealRestorerPipeline

    if args.load is None and args.model_path is None:
        raise ValueError(
            "Pass --model_path for direct inference from a packaged repo."
        )
    if args.load is not None and args.model_path is None and (args.ae_path is None or args.qwen2vl_path is None):
        raise ValueError(
            "When using --load without --model_path, you must pass both --ae_path and --qwen2vl_path."
        )

    if args.load is None:
        pipe = RealRestorerPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
        )
    else:
        source_device = "cpu" if str(args.device).startswith("cuda") else args.device
        pipe = RealRestorerPipeline.from_realrestorer_sources(
            realrestorer_load=args.load,
            model_path=args.model_path,
            ae_path=args.ae_path,
            qwen2vl_path=args.qwen2vl_path,
            device=source_device,
            dtype=torch_dtype,
            mode=args.mode,
            version=args.version,
            model_guidance=args.model_guidance,
            max_length=args.max_length,
        )
    configure_pipeline_memory(pipe=pipe, device=args.device)

    image = Image.open(args.image).convert("RGB") if args.image else None
    negative_prompt = args.negative_prompt
    if negative_prompt is None:
        negative_prompt = "" if image is not None else DEFAULT_T2I_NEGATIVE_PROMPT

    result = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        size_level=args.size_level,
        height=args.height,
        width=args.width,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)

    print(f"Saved output image to {output_path}")


if __name__ == "__main__":
    main()
