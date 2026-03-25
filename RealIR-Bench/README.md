# RealIR-Bench

This folder contains a simplified evaluator for paired restoration results.

## What It Expects

You only need two directories:

- one reference directory
- one prediction directory

Both directories must contain the same relative file names so the evaluator can match image pairs directly.

Example:

```text
ref_dir/
├── 0001.png
├── 0002.png
└── subset/0003.png

pred_dir/
├── 0001.png
├── 0002.png
└── subset/0003.png
```

## What It Outputs

The evaluator prints aggregated metrics for the selected task:

```text
LPIPS_Score = mean(LPIPS)
VLM_Score_Diff = mean(VLM_Score_Diff)
FS = mean(0.2 * VLM_Score_Diff * (1 - LPIPS))
```

If needed, you can also save per-image details with `--output-csv`.

## Installation

Install the extra benchmark dependencies on top of the repository root requirements:

```bash
cd /data/yfyang/project/RealRestorer-diffuser
python -m pip install -r RealIR-Bench/requirements.txt
```

You also need a local Qwen3-VL checkpoint, for example:

```bash
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

## Usage

Run the top-level entry:

```bash
python3 /data/yfyang/project/RealRestorer-diffuser/evaluate_realir_bench.py \
  --ref-dir /path/to/reference_dir \
  --pred-dir /path/to/prediction_dir \
  --task reflection \
  --vlm-model-path /path/to/Qwen3-VL-8B-Instruct
```

The script prints the task-level `LPIPS_Score`, `VLM_Score_Diff`, and `FS` to stdout.

Optional detailed CSV:

```bash
python3 /data/yfyang/project/RealRestorer-diffuser/evaluate_realir_bench.py \
  --ref-dir /path/to/reference_dir \
  --pred-dir /path/to/prediction_dir \
  --task reflection \
  --vlm-model-path /path/to/Qwen3-VL-8B-Instruct \
  --output-csv /path/to/results.csv
```

## Notes

- `--task` is the degradation type used by the Qwen3-VL scoring prompt, such as `reflection`, `rain`, `blur`, `noise`, or `moire`.
- Prediction images are resized to the reference image size before LPIPS and VLM scoring.
- The implementation stays in `RealIR-Bench/evaluate_fs.py`, while the release-facing script is `evaluate_realir_bench.py` at the repository root.
