<div align="center">
  <h2>RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models</h2>
  <p>
    <a href="https://yfyang007.github.io/RealRestorer/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"/></a>
    <a href="https://arxiv.org/abs/2603.25502"><img src="https://img.shields.io/badge/arXiv-2603.25502-b31b1b.svg" alt="arXiv Paper"/></a>
    <a href="https://huggingface.co/RealRestorer/RealRestorer"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow.svg" alt="HuggingFace Model"/></a>
    <a href="https://huggingface.co/datasets/RealRestorer/RealIR-Bench"><img src="https://img.shields.io/badge/HuggingFace-RealIR--Bench-green.svg" alt="RealIR-Bench"/></a>
    <a href="https://huggingface.co/spaces/dericky286/RealRestorer-Demo"><img src="https://img.shields.io/badge/HuggingFace-Demo-orange.svg" alt="HuggingFace Demo"/></a>
  </p>
</div>

<p align="center">
  <img src="assets/teaser.png" alt="RealRestorer teaser" width="100%"/>
</p>

## News
- [2026/03/29] We have released the [Hugging Face demo](https://huggingface.co/spaces/dericky286/RealRestorer-Demo) for RealRestorer.
- [2026/03/26] We have released a new benchmark for real-world image degradation, [RealIR-Bench](https://huggingface.co/datasets/RealRestorer/RealIR-Bench).
- [2026/03/26] We have released the RealRestorer data pipeline and [model weights](https://huggingface.co/RealRestorer/RealRestorer).


## Community Contributions

A community implementation for ComfyUI is available here:

- [ComfyUI-RealRestorer](https://github.com/yfyang007/ComfyUI-RealRestorer)  
  Special thanks to [CCP6](https://huggingface.co/CCP6) for support and community contribution.
- [Comfyui_RealRestorer](https://github.com/StartHua/Comfyui_RealRestorer)  
  Thanks to [@StartHua](https://github.com/StartHua) for making this possible.


## TODO

- [x] Open-source the inference code, degradation pipeline code, and RealIR-Bench evaluation code.
- [x] Release RealIR-Bench on Hugging Face.
- [x] Open-source the RealRestorer model weights.
- [ ] Open-source the Qwen-Image-Edit-2511 version.


## Quick Start

### 1. Installation

This project relies on the patched local `diffusers/` checkout in this repository.

- Python: `3.12`

```bash
python3.12 -m pip install --upgrade pip

cd diffusers
python -m pip install -e .

cd ..
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r RealIR-Bench/requirements.txt
```

You can verify the environment with:

```bash
python -c "from diffusers import RealRestorerPipeline; print(RealRestorerPipeline.__name__)"
```

### 2. Recommended Inference Config

- Device: `cuda`
- Torch dtype: `bfloat16`
- Inference steps: `28`
- Guidance scale: `3.0`
- Recommended seed: `42`

For practical deployment, we recommend using inputs around `1024 x 1024`, which offers a good balance between restoration quality, runtime, and memory usage in the current release. Under the recommended configuration above, the peak GPU memory usage is about `34 GB`.

### 3. Task Prompts

| Task | English Prompt | 中文 Prompt |
| --- | --- | --- |
| Blur Removal | `Please deblur the image and make it sharper` | `请将图像去模糊，变得更清晰。` |
| Compression Artifact Removal | `Please restore the image clarity and artifacts.` | `请修复图像清晰度和伪影。` |
| Lens Flare Removal | `Please remove the lens flare and glare from the image.` | `请去除图像中的光晕和炫光。` |
| Moire Removal | `Please remove the moiré patterns from the image` | `请将图像中的摩尔条纹去除` |
| Dehazing | `Please dehaze the image` | `请将图像去雾。` |
| Low-light Enhancement | `Please restore this low-quality image, recovering its normal brightness and clarity.` | `请修复这张低质量图像，恢复其正常的亮度和清晰度。` |
| Denoising | `Please remove noise from the image.` | `请去除图像中的噪声。` |
| Rain Removal | `Please remove the rain from the image and restore its clarity.` | `请去除图像中的雨水并恢复图像清晰度` |
| Reflection Removal | `Please remove the reflection from the image.` | `请移除图像中的反光` |

## RealRestorer Inference

```bash
python3 infer_realrestorer.py \
  --model_path /path/to/realrestorer_bundle \
  --image /path/to/input.png \
  --prompt "Restore the details and keep the original composition." \
  --output /path/to/output.png \
  --device cuda \
  --torch_dtype bfloat16 \
  --num_inference_steps 28 \
  --guidance_scale 3.0 \
  --seed 42
```

## Degradation Pipeline

`degradation_pipeline/` is the synthetic degradation pipeline used by RealRestorer. It is released together with this repository and can be used to synthesize real-world degradations for training, analysis, and controlled evaluation.

The current pipeline covers common restoration settings including blur, haze, noise, rain, moire, and reflection.

Generation script:

```bash
python3 infer_degradation.py \
  --image /path/to/input.png \
  --degradation reflection \
  --output /path/to/degraded.png \
  --reflection_ckpt_path /path/to/130_net_G.pth
```

The script writes the degraded image and a JSON metadata file for the sampled degradation settings.

## Benchmark Evaluation

RealIR-Bench uses a VLM-based scoring protocol to compare restored images against references and reports `LPIPS_Score`, `VLM_Score_Diff`, and `FS`, where:

```text
FS = 0.2 * VLM_Score_Diff * (1 - LPIPS)
```

Evaluation script:

```bash
python3 evaluate_realir_bench.py \
  --ref-dir /path/to/reference_dir \
  --pred-dir /path/to/prediction_dir \
  --task reflection \
  --vlm-model-path /path/to/Qwen3-VL-8B-Instruct
```

For degradation-specific evaluation, set `--task` to the corresponding restoration target such as `blur`, `noise`, `rain`, `moire`, or `reflection`.



## Citation

If you find RealRestorer useful in your research, please star and cite:

```bibtex
@misc{yang2026realrestorergeneralizablerealworldimage,
      title={RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models},
      author={Yufeng Yang and Xianfang Zeng and Zhangqi Jiang and Fukun Yin and Jianzhuang Liu and Wei Cheng and jinghong lan and Shiyu Liu and Yuqi Peng and Gang YU and Shifeng Chen},
      year={2026},
      eprint={2603.25502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.25502},
}
```
