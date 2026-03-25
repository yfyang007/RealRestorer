# Bundled Reflection Runtime

This directory vendors the minimum reflection synthesis inference runtime used by `degradation_pipeline`.

## Layout

```text
reflection_runtime/
├── __init__.py
├── model.py
├── networks.py
└── assets/
    ├── checkpoints/
    └── reflections/
```

## What Is Bundled

- The inference-only `ReflectionSynthesisInferenceModel` wrapper.
- The generator network definition required by the reflection synthesis checkpoint.

Training code, dataset code, discriminator code, and the rest of the original repository are intentionally not copied here.

## External Resources Still Required

The repo does not vendor the large runtime assets. Place them in the bundled `assets/` directories or keep passing explicit CLI paths.

Expected assets:

- `assets/checkpoints/`: a compatible generator checkpoint such as `130_net_G.pth`.
- `assets/reflections/`: reflection layer images used as the synthetic overlay source.

## Copy From Your Current Setup

If you want the reflection CLI to work without extra path flags, copy your existing assets into this repo:

```bash
cp /data/yfyang/project/Single-Image-Reflection-Removal-Beyond-Linearity/Synthesis/checkpoints_synthesis/130_net_G.pth \
  /data/yfyang/project/RealRestorer-diffuser/degradation_pipeline/reflection_runtime/assets/checkpoints/

cp /data/yfyang/project/Single-Image-Reflection-Removal-Beyond-Linearity/Synthesis/img/testA/* \
  /data/yfyang/project/RealRestorer-diffuser/degradation_pipeline/reflection_runtime/assets/reflections/
```

After that, a minimal Python example is:

```python
from degradation_pipeline import DegradationPipeline

pipe = DegradationPipeline(device="cuda:0")
result = pipe(
    "/path/to/background.png",
    "reflection",
    reflection_ckpt_path="/path/to/130_net_G.pth",
    seed=42,
)
result.images[0].save("/path/to/output.png")
```

To force a specific reflection layer:

```python
from degradation_pipeline import DegradationPipeline

pipe = DegradationPipeline(device="cuda:0")
result = pipe(
    "/path/to/background.png",
    "reflection",
    reflection_ckpt_path="/path/to/130_net_G.pth",
    reflection_image="/path/to/reflection.png",
)
result.images[0].save("/path/to/output.png")
```

To sample a reflection from a directory:

```python
from degradation_pipeline import DegradationPipeline

pipe = DegradationPipeline(device="cuda:0")
result = pipe(
    "/path/to/background.png",
    "reflection",
    reflection_ckpt_path="/path/to/130_net_G.pth",
    reflection_dir="/path/to/reflection_images",
)
result.images[0].save("/path/to/output.png")
```
