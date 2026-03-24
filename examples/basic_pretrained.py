import torch
from diffusers import RealRestorerPipeline


device = "cuda"
dtype = torch.bfloat16
model_path = "/path/to/exported_realrestorer"

pipe = RealRestorerPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe.to(device)

print("Generating text-to-image...")
image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image.save("t2i_output.png")
print("Saved t2i_output.png")

print("Generating image-to-image...")
image_edit = pipe(
    prompt="A cat dressed like a wizard",
    image=image,
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image_edit.save("edit_output.png")
print("Saved edit_output.png")
