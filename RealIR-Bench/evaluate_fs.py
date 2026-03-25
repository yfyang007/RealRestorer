#!/usr/bin/env python3
"""Evaluate paired restoration results with LPIPS, local Qwen3-VL, and FS."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


megfile = None
torch = None
Image = None
lpips = None
process_vision_info = None
AutoProcessor = None
Qwen3VLForConditionalGeneration = None
ToTensor = None

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp", "JPG")
CSV_FIELDS = ["ImageName", "ReferencePath", "PredictionPath", "LPIPS_Score", "VLM_Score_Diff", "FS"]
DEFAULT_VLM_TORCH_DTYPE = "auto"
DEFAULT_VLM_MAX_NEW_TOKENS = 32

WORKER_VLM_SCORER = None
WORKER_LPIPS_MODEL = None
WORKER_TO_TENSOR = None
WORKER_DEVICE = None


def import_runtime_dependencies() -> None:
    global megfile, torch, Image, lpips, process_vision_info, AutoProcessor, Qwen3VLForConditionalGeneration, ToTensor

    if all(
        dependency is not None
        for dependency in (
            megfile,
            torch,
            Image,
            lpips,
            process_vision_info,
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            ToTensor,
        )
    ):
        return

    import megfile as _megfile
    import torch as _torch
    import lpips as _lpips
    from PIL import Image as _Image
    from qwen_vl_utils import process_vision_info as _process_vision_info
    from torchvision.transforms import ToTensor as _ToTensor
    from transformers import AutoProcessor as _AutoProcessor, Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration

    megfile = _megfile
    torch = _torch
    Image = _Image
    lpips = _lpips
    process_vision_info = _process_vision_info
    AutoProcessor = _AutoProcessor
    Qwen3VLForConditionalGeneration = _Qwen3VLForConditionalGeneration
    ToTensor = _ToTensor


def setup_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU mode only supports float32 for the local Qwen3-VL model.")
    return dtype


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


class LocalQwenVLScorer:
    def __init__(self, model_path: str, device: torch.device, torch_dtype_name: str, max_new_tokens: int) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=resolve_torch_dtype(torch_dtype_name, device),
            local_files_only=True,
        ).to(device)
        self.model.eval()

    def score_image(self, image: Image.Image, task: str) -> Optional[float]:
        query = (
            f"Evaluate ONLY whether the image exhibits the degradation type “{task}” and how severe it is. "
            "Ignore any quality aspects unrelated to this degradation and any semantic content.\n"
            f"Method: Divide the image into regions. Inspect each region for “{task}”, estimate the proportion of "
            "affected area and its severity, then aggregate to a SINGLE INTEGER score. "
            "Do NOT output your reasoning.\n"
            "Scale (integers 1-5):\n"
            f"5 = No “{task}”\n"
            "4 = Mild degradation\n"
            "3 = Moderate degradation\n"
            "2 = Severe degradation\n"
            "1 = Extreme degradation\n"
            "Output: Return ONLY “退化分数：<1-5>”. Example: 退化分数：4"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = move_batch_to_device(model_inputs, self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_token_count = model_inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_token_count:]
        response_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return parse_vlm_score(response_text)


def parse_vlm_score(response_text: str) -> Optional[float]:
    response_text = response_text.strip()
    match = re.search(r"退化分数\s*[：:]\s*([1-5])", response_text)
    if match:
        return float(match.group(1))
    match = re.search(r"([1-5])\s*$", response_text)
    if match:
        return float(match.group(1))
    return None


def resize_for_vlm(image: Image.Image, target_pixels: int = 512 * 512) -> Image.Image:
    aspect_ratio = max(image.width / image.height, 1e-6)
    target_width = max(1, int((target_pixels * aspect_ratio) ** 0.5))
    target_height = max(1, int((target_pixels / aspect_ratio) ** 0.5))
    return image.resize((target_width, target_height), Image.LANCZOS)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def is_remote_path(path: str) -> bool:
    return "://" in path


def ensure_parent_dir(path: str) -> None:
    if is_remote_path(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def smart_write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(path)
    with megfile.smart_open(path, "w", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_dir(path: str) -> str:
    return path.rstrip("/\\")


def relative_key(path: str, root: str) -> str:
    normalized_root = normalize_dir(root)
    normalized_path = path.replace("\\", "/")
    normalized_root = normalized_root.replace("\\", "/")
    prefix = normalized_root + "/"
    if normalized_path.startswith(prefix):
        return normalized_path[len(prefix):]
    return os.path.basename(normalized_path)


def collect_image_map(root_dir: str) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    for pattern in IMAGE_PATTERNS:
        glob_pattern = megfile.smart_path_join(root_dir, "**", pattern)
        for image_path in megfile.smart_glob(glob_pattern, recursive=True, missing_ok=True):
            key = relative_key(image_path, root_dir)
            image_map[key] = image_path
    return image_map


def collect_items(reference_dir: str, prediction_dir: str) -> List[Dict[str, str]]:
    reference_map = collect_image_map(reference_dir)
    prediction_map = collect_image_map(prediction_dir)
    shared_keys = sorted(set(reference_map) & set(prediction_map))
    if not shared_keys:
        raise RuntimeError(
            "No matched image pairs found between --ref-dir and --pred-dir. "
            "Both directories must contain the same relative file names."
        )

    items: List[Dict[str, str]] = []
    for key in shared_keys:
        items.append(
            {
                "image_name": key,
                "reference_image_path": reference_map[key],
                "prediction_image_path": prediction_map[key],
            }
        )
    return items


def resolve_default_workers(requested_workers: Optional[int], device_names: Sequence[str]) -> int:
    if requested_workers is not None:
        return max(1, requested_workers)
    if device_names and device_names[0].startswith("cuda"):
        return len(device_names)
    return 1


def build_device_names(device: str, requested_workers: Optional[int]) -> List[str]:
    if device == "cpu":
        if requested_workers not in (None, 1):
            raise RuntimeError("CPU mode only supports one worker when using a local Qwen3-VL model.")
        return ["cpu"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")
    gpu_count = torch.cuda.device_count()
    worker_count = resolve_default_workers(requested_workers, [f"cuda:{idx}" for idx in range(gpu_count)])
    if worker_count > gpu_count:
        raise RuntimeError(
            f"Requested {worker_count} workers but only found {gpu_count} GPUs. "
            "Local Qwen3-VL loads one model copy per worker, so keep --num-workers <= GPU count."
        )
    return [f"cuda:{idx}" for idx in range(worker_count)]


def init_worker(device_queue, config: Dict[str, str]) -> None:
    global WORKER_DEVICE, WORKER_LPIPS_MODEL, WORKER_TO_TENSOR, WORKER_VLM_SCORER

    import_runtime_dependencies()
    device_name = device_queue.get()
    WORKER_DEVICE = torch.device(device_name)
    torch.set_num_threads(1)
    if WORKER_DEVICE.type == "cuda":
        torch.cuda.set_device(WORKER_DEVICE)

    WORKER_TO_TENSOR = ToTensor()
    WORKER_LPIPS_MODEL = lpips.LPIPS(net=config["lpips_net"]).to(WORKER_DEVICE)
    WORKER_LPIPS_MODEL.eval()
    WORKER_VLM_SCORER = LocalQwenVLScorer(
        model_path=config["vlm_model_path"],
        device=WORKER_DEVICE,
        torch_dtype_name=config["vlm_torch_dtype"],
        max_new_tokens=int(config["vlm_max_new_tokens"]),
    )


def process_single_item(item: Dict[str, str], task: str, max_retries: int) -> Optional[Dict[str, str]]:
    for attempt in range(max_retries):
        try:
            with megfile.smart_open(item["reference_image_path"], "rb") as handle:
                pil_reference = Image.open(handle).convert("RGB")
            with megfile.smart_open(item["prediction_image_path"], "rb") as handle:
                pil_prediction_raw = Image.open(handle).convert("RGB")

            pil_prediction = pil_prediction_raw.resize(pil_reference.size, Image.LANCZOS)

            with torch.no_grad():
                tensor_reference = WORKER_TO_TENSOR(pil_reference).unsqueeze(0).to(WORKER_DEVICE) * 2 - 1
                tensor_prediction = WORKER_TO_TENSOR(pil_prediction).unsqueeze(0).to(WORKER_DEVICE) * 2 - 1
                lpips_score = float(WORKER_LPIPS_MODEL(tensor_reference, tensor_prediction).item())

            reference_vlm_score = WORKER_VLM_SCORER.score_image(resize_for_vlm(pil_reference), task)
            prediction_vlm_score = WORKER_VLM_SCORER.score_image(resize_for_vlm(pil_prediction), task)
            if reference_vlm_score is None or prediction_vlm_score is None:
                raise RuntimeError("VLM returned an unparsable score.")

            vlm_score_diff = max(0.0, float(prediction_vlm_score - reference_vlm_score))
            fs_score = 0.2 * vlm_score_diff * (1.0 - lpips_score)

            return {
                "ImageName": item["image_name"],
                "ReferencePath": item["reference_image_path"],
                "PredictionPath": item["prediction_image_path"],
                "LPIPS_Score": format_float(lpips_score),
                "VLM_Score_Diff": format_float(vlm_score_diff),
                "FS": format_float(fs_score),
            }
        except Exception as error:
            if attempt >= max_retries - 1:
                print(
                    f"Failed to process {item['image_name']} after {max_retries} attempts: {error}",
                    file=sys.stderr,
                )
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _process_single_item_wrapper(payload: Tuple[Dict[str, str], str, int]) -> Optional[Dict[str, str]]:
    item, task, max_retries = payload
    return process_single_item(item=item, task=task, max_retries=max_retries)


def evaluate_items(
    items: Sequence[Dict[str, str]],
    task: str,
    device_names: Sequence[str],
    worker_config: Dict[str, str],
    num_workers: int,
    max_retries: int,
    ) -> List[Dict[str, str]]:
    if not items:
        return []

    manager = mp.Manager()
    device_queue = manager.Queue()
    for device_name in device_names[:num_workers]:
        device_queue.put(device_name)

    results: List[Dict[str, str]] = []
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(device_queue, worker_config),
    ) as pool:
        payloads = [(item, task, max_retries) for item in items]
        for index, result in enumerate(pool.imap_unordered(_process_single_item_wrapper, payloads), start=1):
            if result is not None:
                results.append(result)
            if index % 50 == 0 or index == len(items):
                print(f"Processed {index}/{len(items)} image pairs.", file=sys.stderr)
    return sorted(results, key=lambda row: row["ImageName"])


def build_summary(task: str, rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    return {
        "Task": task,
        "NumImages": str(len(rows)),
        "LPIPS_Score": format_float(mean(float(row["LPIPS_Score"]) for row in rows)),
        "VLM_Score_Diff": format_float(mean(float(row["VLM_Score_Diff"]) for row in rows)),
        "FS": format_float(mean(float(row["FS"]) for row in rows)),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a prediction directory against a reference directory with LPIPS and a local Qwen3-VL model, "
            "then compute FS = 0.2 * VLM_Score_Diff * (1 - LPIPS)."
        )
    )
    parser.add_argument("--ref-dir", required=True, help="Reference image directory.")
    parser.add_argument("--pred-dir", required=True, help="Prediction image directory.")
    parser.add_argument("--task", required=True, help="Degradation type, for example reflection, rain, blur, or noise.")
    parser.add_argument(
        "--vlm-model-path",
        default=os.environ.get("QWEN3_VL_MODEL_PATH"),
        help="Local Qwen3-VL checkpoint path or a cached local model id.",
    )
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device type for LPIPS.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes. Defaults to one worker per GPU or 1 on CPU. Keep this <= GPU count for local Qwen3-VL.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per image pair.")
    parser.add_argument("--seed", type=int, default=20, help="Random seed.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional per-image output CSV path.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if not args.vlm_model_path:
        parser.error("--vlm-model-path is required. You can also set QWEN3_VL_MODEL_PATH.")

    import_runtime_dependencies()
    setup_seed(args.seed)
    mp.set_start_method("spawn", force=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device_names = build_device_names(args.device, args.num_workers)
    num_workers = resolve_default_workers(args.num_workers, device_names)
    items = collect_items(reference_dir=args.ref_dir, prediction_dir=args.pred_dir)

    worker_config = {
        "vlm_model_path": args.vlm_model_path,
        "vlm_torch_dtype": DEFAULT_VLM_TORCH_DTYPE,
        "vlm_max_new_tokens": str(DEFAULT_VLM_MAX_NEW_TOKENS),
        "lpips_net": args.lpips_net,
    }
    rows = evaluate_items(
        items=items,
        task=args.task,
        device_names=device_names,
        worker_config=worker_config,
        num_workers=num_workers,
        max_retries=args.max_retries,
    )
    if not rows:
        print("No image pairs were successfully evaluated.", file=sys.stderr)
        return 1

    if args.output_csv:
        smart_write_csv(args.output_csv, rows, CSV_FIELDS)

    summary = build_summary(task=args.task, rows=rows)
    print(f"Task: {summary['Task']}")
    print(f"NumImages: {summary['NumImages']}")
    print(f"LPIPS_Score: {summary['LPIPS_Score']}")
    print(f"VLM_Score_Diff: {summary['VLM_Score_Diff']}")
    print(f"FS: {summary['FS']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
