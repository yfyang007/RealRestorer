# RealRestorer-diffuser

Thin utilities and examples for running `RealRestorerPipeline` through a RealRestorer-enabled `diffusers` repository.

## Important

This project does **not** work with the upstream PyPI `diffusers` package.

You must install the **patched `diffusers` repo that already contains `RealRestorerPipeline`** before installing this repository.

- Correct: install your RealRestorer-enabled `diffusers` repo
- Wrong: `pip install diffusers`

Detailed setup instructions are available in [docs/INSTALL.md](docs/INSTALL.md).

## What This Repo Contains

The actual pipeline implementation lives in the custom `diffusers` repo.

This repository intentionally stays thin and only provides:

- direct usage examples
- an export CLI for building a diffusers bundle from original RealRestorer weights
- a minimal inference CLI for quick local testing

## Installation

### 1. Install the patched `diffusers` repo

Clone and install the `diffusers` repository that contains `RealRestorerPipeline`:

```bash
cd RealRestorer-diffuser/diffusers
pip install -e .
```

If you already installed the upstream `diffusers`, make sure the patched repo is the one that is actually imported in your environment.

### 2. Install this repo

```bash
cd RealRestorer-diffuser
pip install -r requirements.txt
pip install -e .
```

### 3. Verify the installation

```bash
python -c "from diffusers import RealRestorerPipeline; print(RealRestorerPipeline.__name__)"
```

If this import fails, your environment is still using the wrong `diffusers`.

The local runtime requirements for this repository are also listed in [requirements.txt](requirements.txt).

## Recommended Config

For image restoration and editing, the recommended starting point is:

- `device="cuda"`
- `torch_dtype=torch.bfloat16`
- `num_inference_steps=28`
- `guidance_scale=3.0`
- `seed=42`
- use `pipe.enable_model_cpu_offload(device=device)` on CUDA

`seed=42` is the default recommendation used in this repo.

## Quick Start

The main usage path is standard `diffusers` code. Load a packaged repo, enable model CPU offload, and run inference:

```python
import torch
from diffusers import RealRestorerPipeline

device = "cuda"
dtype = torch.bfloat16
seed = 42
model_path = "/path/to/packed_realrestorer_repo"

pipe = RealRestorerPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe.enable_model_cpu_offload(device=device)

image = pipe(
    image="/path/to/input.png",
    prompt="Please remove the reflection from the image.",
    num_inference_steps=28,
    guidance_scale=3.0,
    seed=seed,
).images[0]
image.save("output.png")
```

More examples:

- [examples/basic_pretrained.py](examples/basic_pretrained.py)
- [examples/basic_source.py](examples/basic_source.py)

## Loading Modes

### 1. Load from a packaged diffusers repo

Use this when you already exported a self-contained model repo:

```python
import torch
from diffusers import RealRestorerPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = RealRestorerPipeline.from_pretrained(
    "/path/to/packed_realrestorer_repo",
    torch_dtype=dtype,
)
pipe.enable_model_cpu_offload(device=device)
```

## Degradation Prompts


Recommended prompt templates from the benchmark:

- `blur`: `Please deblur the image and make it sharper`
- `compression`: `Please restore the image clarity and artifacts.`
- `deflare`: `Please remove the lens flare and glare from the image.`
- `demoire`: `Please remove the moire patterns from the image`
- `hazy`: `Please dehaze the image`
- `lowlight`: `Please restore this low-quality image, recovering its normal brightness and clarity.`
- `noise`: `Please remove noise from the image.`
- `rain`: `Please remove the rain from the image and restore its clarity.`
- `reflection`: `Please remove the reflection from the image.`

## CLI

CLI is optional. It is mainly for quick local testing. The recommended path is still direct Python usage with `RealRestorerPipeline`.

On CUDA, the CLI will automatically enable model CPU offload.


### Run inference from a packaged repo

```bash
realrestorer-diffuser-infer \
  --model_path /path/to/packed_realrestorer_repo \
  --image /path/to/input.png \
  --prompt "Please remove the reflection from the image." \
  --output /path/to/output.png \
  --device cuda \
  --torch_dtype bfloat16 \
  --seed 42
```



## Notes

- `model_path` is the packaged repo path for direct inference.
- When used together with `--load`, `model_path` is treated as the shared asset root.
- For text-to-image, omit `image`.
- The exported bundle is intended for direct `RealRestorerPipeline.from_pretrained(...)` usage.
- The core implementation is maintained in the patched `diffusers` repo, not in this repository.
