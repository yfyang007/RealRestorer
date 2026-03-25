#!/usr/bin/env python3
"""Minimal RealIR benchmark evaluator for LPIPS, VLM_Score_Diff and FS."""

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from os.path import basename
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import megfile
import torch
from PIL import Image
import lpips
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from torchvision.transforms import ToTensor


DEFAULT_TASKS = [
    "blur",
    "compression",
    "demoire",
    "noise",
    "deflare",
    "hazy",
    "lowlight",
    "rain",
    "reflection",
]
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.webp")
CSV_FIELDS = ["Model", "Task", "ImageName", "LPIPS_Score", "VLM_Score_Diff", "FS"]
SUMMARY_FIELDS = ["Model", "Task", "NumImages", "LPIPS_Score_Mean", "VLM_Score_Diff_Mean", "FS_Mean"]
DEFAULT_VLM_TORCH_DTYPE = "auto"
DEFAULT_VLM_MAX_NEW_TOKENS = 32

WORKER_VLM_SCORER = None
WORKER_LPIPS_MODEL = None
WORKER_TO_TENSOR = None
WORKER_DEVICE = None


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
            f"Method: Divide the image into regions (e.g., a 3×3 grid). Inspect each region for “{task}”, "
            "estimate the proportion of affected area and its severity, then aggregate to a SINGLE INTEGER score. "
            "If the distribution is uneven, weight by region area. For borderline cases, choose the nearest level. "
            "Do NOT output your reasoning.\n"
            "Scale (integers 1–5):\n"
            f"5 = No “{task}”\n"
            "4 = Mild, small/localized presence (≈≤20% of areas)\n"
            "3 = Moderate, noticeable across multiple areas (≈20–50%)\n"
            "2 = Severe, large portion clearly affected (≈50–80%)\n"
            "1 = Extreme, nearly the entire image (≈>80%)\n"
            "Output: Return ONLY “退化分数：<1-5>”. No extra text, symbols, or line breaks, within 10 words. "
            "Example: 退化分数：4"
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
    target_width = max(1, int(math.sqrt(target_pixels * aspect_ratio)))
    target_height = max(1, int(math.sqrt(target_pixels / aspect_ratio)))
    return image.resize((target_width, target_height), Image.LANCZOS)


def is_remote_path(path: str) -> bool:
    return "://" in path


def ensure_parent_dir(path: str) -> None:
    if is_remote_path(path):
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def smart_read_csv(path: str) -> List[Dict[str, str]]:
    if not megfile.smart_exists(path):
        return []
    with megfile.smart_open(path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def smart_write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(path)
    with megfile.smart_open(path, "w", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def discover_models(data_root: str, tasks: Sequence[str], excluded_names: Set[str]) -> List[str]:
    discovered: List[str] = []
    seen: Set[str] = set()
    for task in tasks:
        task_root = megfile.smart_path_join(data_root, task)
        if not megfile.smart_isdir(task_root):
            continue
        for entry in sorted(megfile.smart_listdir(task_root)):
            if entry in excluded_names:
                continue
            model_dir = megfile.smart_path_join(task_root, entry)
            if megfile.smart_isdir(model_dir) and entry not in seen:
                seen.add(entry)
                discovered.append(entry)
    return discovered


def load_existing_results(output_csv: str) -> Tuple[List[Dict[str, str]], Set[Tuple[str, str, str]]]:
    rows = smart_read_csv(output_csv)
    existing_keys: Set[Tuple[str, str, str]] = set()
    for row in rows:
        model = row.get("Model")
        task = row.get("Task")
        image_name = row.get("ImageName")
        if model and task and image_name:
            existing_keys.add((model, task, image_name))
    return rows, existing_keys


def resolve_reference_model(model_name: str, bench_model_name: str, bench_gpt_model_name: str, gpt_bench_model_key: str) -> str:
    if model_name == gpt_bench_model_key:
        return bench_gpt_model_name
    return bench_model_name


def collect_items(args: argparse.Namespace, existing_keys: Set[Tuple[str, str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    excluded_names = {args.bench_model_name, args.bench_gpt_model_name}
    models = list(args.models) if args.models else discover_models(args.data_root, args.tasks, excluded_names)
    if not models:
        raise RuntimeError("No models found. Pass --models explicitly or check --data-root.")

    items: List[Dict[str, str]] = []
    for task in args.tasks:
        for model_name in models:
            model_dir = megfile.smart_path_join(args.data_root, task, model_name)
            if not megfile.smart_isdir(model_dir):
                continue
            reference_model = resolve_reference_model(
                model_name=model_name,
                bench_model_name=args.bench_model_name,
                bench_gpt_model_name=args.bench_gpt_model_name,
                gpt_bench_model_key=args.gpt_bench_model_key,
            )
            for pattern in IMAGE_PATTERNS:
                image_paths = megfile.smart_glob(megfile.smart_path_join(model_dir, pattern), recursive=False, missing_ok=True)
                for edited_path in image_paths:
                    image_name = basename(edited_path)
                    item_key = (model_name, task, image_name)
                    if item_key in existing_keys:
                        continue
                    original_path = megfile.smart_path_join(args.data_root, task, reference_model, image_name)
                    if not megfile.smart_exists(original_path):
                        continue
                    items.append(
                        {
                            "model_name": model_name,
                            "task": task,
                            "image_name": image_name,
                            "original_image_path": original_path,
                            "edited_image_path": edited_path,
                        }
                    )
    return models, items


def init_worker(device_queue, config: Dict[str, str]) -> None:
    global WORKER_DEVICE, WORKER_LPIPS_MODEL, WORKER_TO_TENSOR, WORKER_VLM_SCORER

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


def process_single_item(item: Dict[str, str], max_retries: int) -> Optional[Dict[str, str]]:
    for attempt in range(max_retries):
        try:
            with megfile.smart_open(item["original_image_path"], "rb") as handle:
                pil_original = Image.open(handle).convert("RGB")
            with megfile.smart_open(item["edited_image_path"], "rb") as handle:
                pil_edited_raw = Image.open(handle).convert("RGB")

            pil_edited = pil_edited_raw.resize(pil_original.size, Image.LANCZOS)

            with torch.no_grad():
                tensor_original = WORKER_TO_TENSOR(pil_original).unsqueeze(0).to(WORKER_DEVICE) * 2 - 1
                tensor_edited = WORKER_TO_TENSOR(pil_edited).unsqueeze(0).to(WORKER_DEVICE) * 2 - 1
                lpips_score = float(WORKER_LPIPS_MODEL(tensor_original, tensor_edited).item())

            original_vlm_score = WORKER_VLM_SCORER.score_image(resize_for_vlm(pil_original), item["task"])
            edited_vlm_score = WORKER_VLM_SCORER.score_image(resize_for_vlm(pil_edited), item["task"])
            if original_vlm_score is None or edited_vlm_score is None:
                raise RuntimeError("VLM returned an unparsable score.")

            vlm_score_diff = max(0.0, float(edited_vlm_score - original_vlm_score))
            fs_score = 0.2 * vlm_score_diff * (1.0 - lpips_score)

            return {
                "Model": item["model_name"],
                "Task": item["task"],
                "ImageName": item["image_name"],
                "LPIPS_Score": format_float(lpips_score),
                "VLM_Score_Diff": format_float(vlm_score_diff),
                "FS": format_float(fs_score),
            }
        except Exception as error:
            if attempt >= max_retries - 1:
                print(
                    f"Failed to process {item['edited_image_path']} after {max_retries} attempts: {error}",
                    file=sys.stderr,
                )
                return None
            time.sleep(2 * (attempt + 1))
    return None


def evaluate_items(
    items: Sequence[Dict[str, str]],
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
        for index, result in enumerate(
            pool.imap_unordered(_process_single_item_wrapper, [(item, max_retries) for item in items]),
            start=1,
        ):
            if result is not None:
                results.append(result)
            if index % 50 == 0 or index == len(items):
                print(f"Processed {index}/{len(items)} image pairs.")
    return results


def _process_single_item_wrapper(payload: Tuple[Dict[str, str], int]) -> Optional[Dict[str, str]]:
    item, max_retries = payload
    return process_single_item(item, max_retries=max_retries)


def sort_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda row: (row["Model"], row["Task"], row["ImageName"]))


def build_summary_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"LPIPS_Score": [], "VLM_Score_Diff": [], "FS": []}
    )
    for row in rows:
        key = (row["Model"], row["Task"])
        grouped[key]["LPIPS_Score"].append(float(row["LPIPS_Score"]))
        grouped[key]["VLM_Score_Diff"].append(float(row["VLM_Score_Diff"]))
        grouped[key]["FS"].append(float(row["FS"]))

    summary_rows: List[Dict[str, str]] = []
    per_model_totals: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"LPIPS_Score": [], "VLM_Score_Diff": [], "FS": []}
    )

    for (model_name, task), metrics in sorted(grouped.items()):
        row = {
            "Model": model_name,
            "Task": task,
            "NumImages": str(len(metrics["FS"])),
            "LPIPS_Score_Mean": format_float(mean(metrics["LPIPS_Score"])),
            "VLM_Score_Diff_Mean": format_float(mean(metrics["VLM_Score_Diff"])),
            "FS_Mean": format_float(mean(metrics["FS"])),
        }
        summary_rows.append(row)
        per_model_totals[model_name]["LPIPS_Score"].extend(metrics["LPIPS_Score"])
        per_model_totals[model_name]["VLM_Score_Diff"].extend(metrics["VLM_Score_Diff"])
        per_model_totals[model_name]["FS"].extend(metrics["FS"])

    for model_name in sorted(per_model_totals):
        metrics = per_model_totals[model_name]
        summary_rows.append(
            {
                "Model": model_name,
                "Task": "Average",
                "NumImages": str(len(metrics["FS"])),
                "LPIPS_Score_Mean": format_float(mean(metrics["LPIPS_Score"])),
                "VLM_Score_Diff_Mean": format_float(mean(metrics["VLM_Score_Diff"])),
                "FS_Mean": format_float(mean(metrics["FS"])),
            }
        )
    return summary_rows


def print_model_averages(summary_rows: Sequence[Dict[str, str]]) -> None:
    average_rows = [row for row in summary_rows if row["Task"] == "Average"]
    if not average_rows:
        print("No summary rows available.")
        return
    print("\nModel averages:")
    for row in average_rows:
        print(
            f"{row['Model']}: "
            f"LPIPS={row['LPIPS_Score_Mean']}  "
            f"VLM_Diff={row['VLM_Score_Diff_Mean']}  "
            f"FS={row['FS_Mean']}"
        )


def build_argparser() -> argparse.ArgumentParser:
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser = argparse.ArgumentParser(
        description="Evaluate RealIR outputs with LPIPS and a local Qwen3-VL model, then compute FS = 0.2 * VLM_Score_Diff * (1 - LPIPS)."
    )
    parser.add_argument("--data-root", default="s3://yfyang/evalu", help="Root directory of benchmark data.")
    parser.add_argument("--models", nargs="*", default=None, help="Model directory names to evaluate. Omit to auto-discover.")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="Tasks to evaluate.")
    parser.add_argument("--bench-model-name", default="bench", help="Reference image directory name for most models.")
    parser.add_argument("--bench-gpt-model-name", default="bench_gpt", help="Reference image directory name for the GPT benchmark model.")
    parser.add_argument("--gpt-bench-model-key", default="gpt_bench", help="Model name that should use --bench-gpt-model-name.")
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
    parser.add_argument("--force-recompute", action="store_true", help="Ignore existing CSV cache.")
    parser.add_argument(
        "--output-csv",
        default=os.path.join(default_output_dir, "fs_scores.csv"),
        help="Detailed per-image CSV path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(default_output_dir, "fs_summary.csv"),
        help="Per-task and per-model summary CSV path.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if not args.vlm_model_path:
        parser.error("--vlm-model-path is required. You can also set QWEN3_VL_MODEL_PATH.")

    setup_seed(args.seed)
    mp.set_start_method("spawn", force=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device_names = build_device_names(args.device, args.num_workers)
    num_workers = resolve_default_workers(args.num_workers, device_names)

    print(f"Using devices: {', '.join(device_names[:num_workers])}")
    print(f"Workers: {num_workers}")

    cached_rows: List[Dict[str, str]] = []
    existing_keys: Set[Tuple[str, str, str]] = set()
    if not args.force_recompute and megfile.smart_exists(args.output_csv):
        cached_rows, existing_keys = load_existing_results(args.output_csv)
        print(f"Loaded {len(cached_rows)} cached rows from {args.output_csv}")

    models, items = collect_items(args, existing_keys)
    print(f"Models: {', '.join(models)}")
    print(f"Queued image pairs: {len(items)}")

    worker_config = {
        "vlm_model_path": args.vlm_model_path,
        "vlm_torch_dtype": DEFAULT_VLM_TORCH_DTYPE,
        "vlm_max_new_tokens": str(DEFAULT_VLM_MAX_NEW_TOKENS),
        "lpips_net": args.lpips_net,
    }
    new_rows = evaluate_items(
        items=items,
        device_names=device_names,
        worker_config=worker_config,
        num_workers=num_workers,
        max_retries=args.max_retries,
    )

    all_rows = cached_rows + new_rows if not args.force_recompute else new_rows
    all_rows = sort_rows(all_rows)
    smart_write_csv(args.output_csv, all_rows, CSV_FIELDS)

    summary_rows = build_summary_rows(all_rows)
    smart_write_csv(args.summary_csv, summary_rows, SUMMARY_FIELDS)

    print(f"\nSaved detailed results to: {args.output_csv}")
    print(f"Saved summary results to: {args.summary_csv}")
    print_model_averages(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
