# RealIR-Bench

这个目录是从 `RealIR-benchmark/qwen3_vl_evalu.py` 整理出来的精简版评估工具，只保留下面三项输出：

- `LPIPS_Score`
- `VLM_Score_Diff`
- `FS = 0.2 * VLM_Score_Diff * (1 - LPIPS_Score)`

不再包含 DINO、SSIM、NIQE、颜色一致性和画图逻辑。

## 环境

按你的要求，默认使用 `gedit_v2`，并且 VLM 评分改成加载本地 `Qwen3-VL`，不再走远程服务：

```bash
conda activate gedit_v2
cd /data/yfyang/project/RealRestorer-diffuser
python -m pip install -r requirements.txt
python -m pip install -r RealIR-Bench/requirements.txt
```

另外需要准备本地 `Qwen3-VL` 权重目录，例如：

```bash
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

## 目录约定

默认沿用原 benchmark 的目录结构：

```text
<data_root>/<task>/<reference_model>/<image_name>
<data_root>/<task>/<model_name>/<image_name>
```

默认参数下：

- 大多数模型的参考图目录是 `bench`
- 当模型名是 `gpt_bench` 时，参考图目录切到 `bench_gpt`

## 运行方式

最常用的命令：

```bash
conda run -n gedit_v2 python /data/yfyang/project/RealRestorer-diffuser/RealIR-Bench/evaluate_fs.py \
  --data-root s3://yfyang/evalu \
  --models qwen_new doubao_no_watermark kontext_new nano_new \
  --vlm-model-path /path/to/Qwen3-VL-8B-Instruct \
  --num-workers 8
```

如果不传 `--models`，脚本会自动从 `data_root/task/*` 下发现模型目录，并排除 `bench` 和 `bench_gpt`。

如果已经设置了 `QWEN3_VL_MODEL_PATH`，可以不再显式传 `--vlm-model-path`。

## 输出文件

默认输出到：

- `RealIR-Bench/results/fs_scores.csv`
- `RealIR-Bench/results/fs_summary.csv`

其中：

- `fs_scores.csv` 是逐图结果
- `fs_summary.csv` 是按 `Model + Task` 汇总后的均值，以及每个模型的 `Average`

## 主要参数

- `--data-root`: benchmark 根目录，支持本地路径和 `s3://`
- `--models`: 要评估的模型目录名列表
- `--tasks`: 要评估的任务列表
- `--vlm-model-path`: 本地 Qwen3-VL 权重路径，或已经缓存到本机的模型 id
- `--device`: `cuda` 或 `cpu`
- `--num-workers`: 并行 worker 数。因为每个 worker 都会各自加载一份本地 Qwen3-VL，建议不要超过 GPU 数量
- `--force-recompute`: 忽略已有 CSV 缓存并全量重算
- `--output-csv`: 逐图结果 CSV
- `--summary-csv`: 汇总结果 CSV

脚本内部默认使用：

- `vlm_torch_dtype=auto`
- `vlm_max_new_tokens=32`
