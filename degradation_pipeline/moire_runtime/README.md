# Bundled Moire Runtime

This directory vendors the minimum UniDemoire inference runtime used by `degradation_pipeline`.

## Layout

```text
moire_runtime/
├── __init__.py
├── blending.py
├── model.py
├── uformer.py
├── configs/
│   └── moire-blending/
│       ├── fhdmi/blending_fhdmi.yaml
│       ├── tip/blending_tip.yaml
│       └── uhdm/blending_uhdm.yaml
└── assets/
    ├── checkpoints/
    ├── moire_patterns/
    └── real_moire/
```

## What Is Bundled

- The inference-only `MoireBlendingInferenceModel` wrapper.
- The MIB blending module and the Uformer network definition used by UniDemoire blending checkpoints.
- The three blending configs used by the existing checkpoints.

Training-only code, dataset code, diffusion code, and the rest of the original UniDemoire repository are intentionally not copied here.

## External Resources Still Required

The repo does not vendor the large runtime assets. Place them in the bundled `assets/` directories or keep passing explicit CLI paths.

Expected assets:

- `assets/checkpoints/`: `bl_uhdm.ckpt`, `bl_tip.ckpt`, `bl_fhdmi.ckpt`, or any compatible blending checkpoint.
- `assets/moire_patterns/`: synthetic or sampled moire pattern images.
- `assets/real_moire/`: real moire reference images used by the refine network.

## Copy From Your Current Setup

If you want the moire CLI to work without extra path flags, copy your existing assets into this repo:

```bash
cp /data/yfyang/project/UniDemoire/ckp_infer/*.ckpt \
  /data/yfyang/project/RealRestorer-diffuser/degradation_pipeline/moire_runtime/assets/checkpoints/

cp /data/yfyang/project/UniDemoire/data/generated/diffusion/2025-08-14-15-40-18/moire_patterns/* \
  /data/yfyang/project/RealRestorer-diffuser/degradation_pipeline/moire_runtime/assets/moire_patterns/

cp /data/yfyang/test_nonmoire/* \
  /data/yfyang/project/RealRestorer-diffuser/degradation_pipeline/moire_runtime/assets/real_moire/
```

After that, a minimal Python example is:

```python
from degradation_pipeline import DegradationPipeline

pipe = DegradationPipeline(device="cuda:0")
result = pipe("/path/to/input.png", "moire", seed=42)
result.images[0].save("/path/to/output.png")
```
