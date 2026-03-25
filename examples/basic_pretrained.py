import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.is_dir() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

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
