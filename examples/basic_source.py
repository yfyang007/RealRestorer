import torch
from diffusers import RealRestorerPipeline


device = "cuda"
dtype = torch.bfloat16

pipe = RealRestorerPipeline.from_realrestorer_sources(
    realrestorer_load="/path/to/realrestorer_ckpt_dir",
    model_path="/path/to/shared_models",
    device=device,
    dtype=dtype,
)
pipe.to(device)

image = pipe(
    image="/path/to/input.png",
    prompt="Restore the details and keep the original composition.",
    negative_prompt="oversmoothed, blurry, low quality",
    num_inference_steps=28,
    guidance_scale=3.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]
image.save("output.png")
