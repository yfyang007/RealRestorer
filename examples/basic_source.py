import torch
from diffusers import RealRestorerPipeline


device = "cuda"
dtype = torch.bfloat16
seed = 42

pipe = RealRestorerPipeline.from_realrestorer_sources(
    realrestorer_load="/path/to/realrestorer_ckpt_dir",
    model_path="/path/to/shared_models",
    device="cpu",
    dtype=dtype,
)
pipe.enable_model_cpu_offload(device=device)

image = pipe(
    image="/path/to/input.png",
    prompt="Restore the details and keep the original composition.",
    negative_prompt="oversmoothed, blurry, low quality",
    num_inference_steps=28,
    guidance_scale=3.0,
    seed=seed,
).images[0]
image.save("output.png")
