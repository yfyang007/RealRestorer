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
- an inference CLI that calls `RealRestorerPipeline` directly

## Installation

### 1. Install the patched `diffusers` repo

Clone and install the `diffusers` repository that contains `RealRestorerPipeline`:

```bash
git clone <YOUR_DIFFUSERS_REPO_URL>
cd diffusers
pip install -e .
```

If you already installed the upstream `diffusers`, make sure the patched repo is the one that is actually imported in your environment.

### 2. Install this repo

```bash
git clone <YOUR_REALRESTORER_DIFFUSER_REPO_URL>
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

## Quick Start

The intended usage style is the standard diffusers pattern:

```python
import torch
from diffusers import RealRestorerPipeline

device = "cuda"
dtype = torch.bfloat16
model_path = "/path/to/exported_realrestorer"

pipe = RealRestorerPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe.to(device)

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image.save("t2i_output.png")
```

More examples:

- [examples/basic_pretrained.py](examples/basic_pretrained.py)
- [examples/basic_source.py](examples/basic_source.py)

## Loading Modes

### 1. Load from a diffusers bundle

Use this when you already exported a self-contained bundle:

```python
import torch
from diffusers import RealRestorerPipeline

pipe = RealRestorerPipeline.from_pretrained(
    "/path/to/exported_realrestorer",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

### 2. Load directly from original RealRestorer weights

Use this when you want to load from the original RealRestorer checkpoint layout:

```python
import torch
from diffusers import RealRestorerPipeline

pipe = RealRestorerPipeline.from_realrestorer_sources(
    realrestorer_load="/path/to/realrestorer_ckpt_dir",
    model_path="/path/to/shared_models",
    device="cuda",
    dtype=torch.bfloat16,
)
pipe.to("cuda")
```

## CLI

CLI is optional. The main usage path is still:

```python
from diffusers import RealRestorerPipeline
```

### Export a bundle

```bash
realrestorer-diffuser-export \
  --load /path/to/realrestorer_ckpt_dir \
  --model_path /path/to/shared_models \
  --save_dir /path/to/exported_realrestorer \
  --device cuda \
  --torch_dtype bfloat16
```

### Run inference from original weights

```bash
realrestorer-diffuser-infer \
  --load /path/to/realrestorer_ckpt_dir \
  --model_path /path/to/shared_models \
  --image /path/to/input.png \
  --prompt "Restore the details and keep the original composition." \
  --output /path/to/output.png \
  --device cuda \
  --torch_dtype bfloat16
```

### Run inference from an exported bundle

```bash
realrestorer-diffuser-infer \
  --pretrained_model_name_or_path /path/to/exported_realrestorer \
  --image /path/to/input.png \
  --prompt "Restore the details and keep the original composition." \
  --output /path/to/output.png \
  --device cuda \
  --torch_dtype bfloat16
```

## Notes

- `model_path` can point to a directory that contains both the RealRestorer VAE asset and the local Qwen2.5-VL model.
- For text-to-image, omit `image`.
- The exported bundle is intended for direct `RealRestorerPipeline.from_pretrained(...)` usage.
- The core implementation is maintained in the patched `diffusers` repo, not in this repository.
