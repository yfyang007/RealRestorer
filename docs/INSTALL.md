# Installation Guide

This project depends on a **custom `diffusers` repository** that already includes `RealRestorerPipeline`.

## Required Setup

You need two repositories:

1. a patched `diffusers` repo that contains the RealRestorer integration
2. this `RealRestorer-diffuser` repo

The standard PyPI package below is **not enough**:

```bash
pip install diffusers
```

If you only install the upstream package, imports such as:

```python
from diffusers import RealRestorerPipeline
```

will fail because upstream `diffusers` does not ship this pipeline.

## Recommended Installation Flow

### Step 1. Install the patched `diffusers`

```bash
git clone <YOUR_DIFFUSERS_REPO_URL>
cd diffusers
pip install -e .
```

If you are working from local paths instead of GitHub:

```bash
cd /path/to/your/diffusers
pip install -e .
```

### Step 2. Install `RealRestorer-diffuser`

```bash
git clone <YOUR_REALRESTORER_DIFFUSER_REPO_URL>
cd RealRestorer-diffuser
pip install -r requirements.txt
pip install -e .
```

If you are working from a local checkout:

```bash
cd /path/to/your/RealRestorer-diffuser
pip install -r requirements.txt
pip install -e .
```

## Verify the Environment

Run:

```bash
python -c "from diffusers import RealRestorerPipeline; print(RealRestorerPipeline.__name__)"
```

Expected result:

```bash
RealRestorerPipeline
```

If this command fails, your Python environment is importing the wrong `diffusers`.

## Common Problems

### 1. Installed the wrong `diffusers`

Symptom:

```text
ImportError: cannot import name 'RealRestorerPipeline' from 'diffusers'
```

Cause:

- the environment is still using the official PyPI `diffusers`
- the patched `diffusers` repo was not installed into the current environment

Fix:

- install the patched `diffusers` repo with `pip install -e .`
- verify again with `python -c "from diffusers import RealRestorerPipeline"`

### 2. Mixed environments

Symptom:

- install succeeds, but `python` still cannot import `RealRestorerPipeline`

Cause:

- `pip` and `python` belong to different virtual environments

Fix:

- use the same environment for both commands
- prefer `python -m pip install -e .`

Example:

```bash
python -m pip install -e /path/to/diffusers
python -m pip install -r /path/to/RealRestorer-diffuser/requirements.txt
python -m pip install -e /path/to/RealRestorer-diffuser
python -c "from diffusers import RealRestorerPipeline"
```

### 3. Missing runtime dependencies

Symptom:

- import reaches `diffusers`, but fails on a dependency such as `huggingface_hub`, `transformers`, or `accelerate`

Fix:

- install the dependencies required by the patched `diffusers` repo
- then reinstall or refresh the environment if needed

## Minimal Usage After Installation

```python
import torch
from diffusers import RealRestorerPipeline

pipe = RealRestorerPipeline.from_pretrained(
    "/path/to/exported_realrestorer",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = pipe(
    prompt="Restore and enhance the image",
    image="/path/to/input.png",
    num_inference_steps=28,
).images[0]
image.save("output.png")
```

## Summary

The key requirement is simple:

- install the RealRestorer-enabled `diffusers` repo first
- install this repo second
- verify `from diffusers import RealRestorerPipeline` before running inference
