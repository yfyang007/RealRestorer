from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from degradation_pipeline import DegradationPipeline, SUPPORTED_DEGRADATIONS
else:
    from . import DegradationPipeline, SUPPORTED_DEGRADATIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone degradation synthesis pipeline.")
    parser.add_argument("--image", required=True, type=str, help="Input image path.")
    parser.add_argument("--degradation", required=True, choices=SUPPORTED_DEGRADATIONS, help="Degradation type.")
    parser.add_argument("--output", required=True, type=str, help="Output image path.")
    parser.add_argument("--metadata_output", type=str, default=None, help="Optional JSON metadata output path.")
    parser.add_argument("--device", type=str, default=None, help="Execution device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    parser.add_argument(
        "--midas_model_type",
        type=str,
        default="DPT_Large",
        help="MiDaS model type for depth-backed degradations such as haze and rain.",
    )
    parser.add_argument(
        "--midas_repo_or_dir",
        type=str,
        default=None,
        help="Optional MiDaS torch.hub repo or local MiDaS checkout path. Defaults to `isl-org/MiDaS`.",
    )
    parser.add_argument("--fog_texture_dir", type=str, default=None, help="Optional fog texture directory for haze.")
    parser.add_argument("--rain_texture_dir", type=str, default=None, help="Optional rain texture directory for rain.")
    parser.add_argument(
        "--disable_density_averaging",
        action="store_true",
        help="Disable the optional density averaging step in noise degradation.",
    )
    parser.add_argument(
        "--disable_realesrgan_degradation",
        action="store_true",
        help="Disable the optional RealESRGAN-style blur/jpeg tail in noise degradation.",
    )
    parser.add_argument(
        "--unidemoire_root",
        type=str,
        default=None,
        help="Optional legacy external UniDemoire root used only to resolve default moire config/checkpoint paths.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Moire config path. Defaults to the bundled moire runtime config.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Moire checkpoint directory. Defaults to the bundled moire asset directory.",
    )
    parser.add_argument(
        "--moire_pattern_dir",
        type=str,
        default=None,
        help="Directory with moire pattern images for moire degradation.",
    )
    parser.add_argument(
        "--real_moire_dir",
        type=str,
        default=None,
        help="Directory with real moire reference images for moire degradation.",
    )
    parser.add_argument("--model_input_size", type=int, default=512, help="UniDemoire input size.")
    parser.add_argument(
        "--reflection_ckpt_path",
        type=str,
        default=None,
        help="Reflection synthesis generator checkpoint path.",
    )
    parser.add_argument(
        "--reflection_dir",
        type=str,
        default=None,
        help="Directory with reflection layer images. Used when --reflection_image is not provided.",
    )
    parser.add_argument(
        "--reflection_image",
        type=str,
        default=None,
        help="Optional explicit reflection layer image. Overrides --reflection_dir sampling.",
    )
    parser.add_argument(
        "--reflection_type",
        type=str,
        default="random",
        choices=("random", "focused", "defocused", "ghosting"),
        help="Reflection rendering mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pipe = DegradationPipeline(
        device=args.device,
        midas_model_type=args.midas_model_type,
        midas_repo_or_dir=args.midas_repo_or_dir,
    )
    result = pipe(
        args.image,
        args.degradation,
        seed=args.seed,
        fog_texture_dir=args.fog_texture_dir,
        rain_texture_dir=args.rain_texture_dir,
        enable_density_averaging=not args.disable_density_averaging,
        enable_realesrgan_degradation=not args.disable_realesrgan_degradation,
        unidemoire_root=args.unidemoire_root,
        config_path=args.config_path,
        ckpt_dir=args.ckpt_dir,
        moire_pattern_dir=args.moire_pattern_dir,
        real_moire_dir=args.real_moire_dir,
        model_input_size=args.model_input_size,
        reflection_ckpt_path=args.reflection_ckpt_path,
        reflection_dir=args.reflection_dir,
        reflection_image=args.reflection_image,
        reflection_type=args.reflection_type,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)

    metadata_output = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(result.metadata[0], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved degraded image to {output_path}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
