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
