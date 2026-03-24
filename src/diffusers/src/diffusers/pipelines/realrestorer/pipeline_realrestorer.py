import contextlib
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ...models import RealRestorerAutoencoderKL, RealRestorerTransformer2DModel
from ...models.realrestorer.state_dict_utils import inspect_realrestorer_checkpoint, resolve_model_assets
from ...schedulers import RealRestorerFlowMatchScheduler
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import RealRestorerPipelineOutput


QWEN25VL_PREFIX = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:"""


class RealRestorerPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        transformer: RealRestorerTransformer2DModel,
        vae: RealRestorerAutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        processor: object,
        scheduler: Optional[RealRestorerFlowMatchScheduler] = None,
        version: str = "v1.1",
        model_guidance: float = 3.5,
        max_length: int = 640,
    ) -> None:
        if isinstance(processor, (list, tuple)) or processor is None:
            processor_source = getattr(text_encoder, "name_or_path", None)
            if not processor_source:
                text_encoder_config = getattr(text_encoder, "config", None)
                processor_source = getattr(text_encoder_config, "_name_or_path", None)
            if not processor_source:
                raise ValueError("Could not infer processor path from text_encoder while initializing RealRestorerPipeline.")
            processor = AutoProcessor.from_pretrained(
                processor_source,
                min_pixels=256 * 28 * 28,
                max_pixels=324 * 28 * 28,
            )
        if isinstance(scheduler, (list, tuple)) or scheduler is None:
            scheduler = RealRestorerFlowMatchScheduler()

        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            scheduler=scheduler,
        )
        self.version = version
        self.model_guidance = float(model_guidance)
        self.max_token_length = int(max_length)
        self.latent_channels = int(getattr(self.vae, "latent_channels", 16))
        self.vae_scale_factor = 8

        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    @classmethod
    def from_realrestorer_sources(
        cls,
        realrestorer_load: str,
        model_path: Optional[str] = None,
        ae_path: Optional[str] = None,
        qwen2vl_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        max_length: int = 640,
        dtype: torch.dtype = torch.bfloat16,
        mode: str = "flash",
        version: str = "auto",
        model_guidance: float = 3.5,
    ) -> "RealRestorerPipeline":
        ae_path, qwen2vl_path = resolve_model_assets(model_path, ae_path, qwen2vl_path)
        ckpt_info = inspect_realrestorer_checkpoint(realrestorer_load)
        inferred_version = "v1.0" if ckpt_info["has_scale_factor"] else "v1.1"
        final_version = inferred_version if version == "auto" else version
        text_encoder_path = qwen2vl_path
        processor_path = qwen2vl_path
        if (
            os.path.isdir(qwen2vl_path)
            and os.path.isdir(os.path.join(qwen2vl_path, "text_encoder"))
            and os.path.isdir(os.path.join(qwen2vl_path, "processor"))
        ):
            text_encoder_path = os.path.join(qwen2vl_path, "text_encoder")
            processor_path = os.path.join(qwen2vl_path, "processor")

        with torch.device("meta"):
            vae = RealRestorerAutoencoderKL()
            transformer = RealRestorerTransformer2DModel(
                mode=mode,
                version=final_version,
                guidance_embeds=bool(ckpt_info["has_guidance_in"]),
                use_mask_token=bool(ckpt_info["has_mask_token"]),
            )

        vae.load_weights(ae_path, strict=True)
        transformer.load_realrestorer_weights(realrestorer_load, strict=False)

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(
            processor_path,
            min_pixels=256 * 28 * 28,
            max_pixels=324 * 28 * 28,
        )

        pipe = cls(
            transformer=transformer.to(device=device, dtype=dtype),
            vae=vae.to(device=device, dtype=torch.float32),
            text_encoder=text_encoder.to(device=device, dtype=dtype),
            processor=processor,
            scheduler=RealRestorerFlowMatchScheduler(),
            version=final_version,
            model_guidance=model_guidance,
            max_length=max_length,
        )
        pipe.to(device)
        return pipe

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        image = image.detach().cpu().clamp(0, 1)
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def load_image(image):
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]
            if image.ndim != 3:
                raise ValueError(f"Unsupported ndarray shape: {image.shape}")
            if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            tensor = torch.from_numpy(image).float()
            if tensor.max() > 1:
                tensor = tensor / 255.0
            return tensor.permute(2, 0, 1).unsqueeze(0)
        if isinstance(image, Image.Image):
            return RealRestorerPipeline._pil_to_tensor(image).unsqueeze(0)
        if isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1:
                tensor = tensor / 255.0
            return tensor
        if isinstance(image, str):
            return RealRestorerPipeline._pil_to_tensor(Image.open(image).convert("RGB")).unsqueeze(0)
        raise ValueError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _split_string(s: str):
        s = s.replace("'", '"').replace("“", '"').replace("”", '"')
        result = []
        in_quotes = False
        temp = ""

        for idx, char in enumerate(s):
            if char == '"' and idx > 155:
                temp += char
                if not in_quotes:
                    result.append(temp)
                    temp = ""

                in_quotes = not in_quotes
                continue
            if in_quotes:
                result.append("“" + char + "”")
            else:
                temp += char

        if temp:
            result.append(temp)

        return result

    @staticmethod
    def _prepare_qwen_image(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        min_pixels = 4 * 28 * 28
        max_pixels = 16384 * 28 * 28
        width, height = image.size
        h_bar = max(28, round(height / 28) * 28)
        w_bar = max(28, round(width / 28) * 28)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(28, math.floor(height / beta / 28) * 28)
            w_bar = max(28, math.floor(width / beta / 28) * 28)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / max(height * width, 1))
            h_bar = max(28, math.ceil(height * beta / 28) * 28)
            w_bar = max(28, math.ceil(width * beta / 28) * 28)
        return image.resize((w_bar, h_bar), Image.BICUBIC)

    def _get_qwenvl_embeds(
        self,
        prompts: List[str],
        ref_images: Optional[List[Optional[Image.Image]]],
        edit_types: List[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_dtype = next(self.text_encoder.parameters()).dtype
        model_device = next(self.text_encoder.parameters()).device
        text_encoder_device = getattr(self, "_offload_device", None) or model_device
        if text_encoder_device != model_device:
            self.text_encoder.to(device=text_encoder_device, dtype=model_dtype)
        batch_size = len(prompts)

        if ref_images is None:
            ref_images = [None] * batch_size

        embs = torch.zeros(
            batch_size,
            self.max_token_length,
            self.text_encoder.config.hidden_size,
            dtype=dtype,
            device=device,
        )
        masks = torch.zeros(batch_size, self.max_token_length, dtype=torch.long, device=device)

        for idx, (prompt, ref_image, edit_type) in enumerate(zip(prompts, ref_images, edit_types)):
            messages = [{"role": "user", "content": []}]
            messages[0]["content"].append({"type": "text", "text": QWEN25VL_PREFIX})
            if edit_type != 0 and ref_image is not None:
                messages[0]["content"].append({"type": "image", "image": ref_image})
            messages[0]["content"].append({"type": "text", "text": prompt})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            token_list = []
            for text_each in self._split_string(text):
                txt_inputs = self.processor(
                    text=text_each,
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                token_each = txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:, 1:-1]
                token_list.append(token_each)

            new_txt_ids = torch.cat(token_list, dim=1).to(text_encoder_device)

            if edit_type != 0 and ref_image is not None:
                image_inputs = [self._prepare_qwen_image(ref_image)]
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                old_inputs_ids = inputs.input_ids.to(text_encoder_device)
                idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
                idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
                input_ids = torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0).unsqueeze(0)
                attention_mask = (input_ids > 0).long().to(text_encoder_device)
                outputs = self.text_encoder(
                    input_ids=input_ids.to(text_encoder_device),
                    attention_mask=attention_mask,
                    pixel_values=inputs.pixel_values.to(text_encoder_device, dtype=model_dtype),
                    image_grid_thw=inputs.image_grid_thw.to(text_encoder_device),
                    output_hidden_states=True,
                )
            else:
                input_ids = new_txt_ids
                attention_mask = (input_ids > 0).long().to(text_encoder_device)
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            emb = outputs.hidden_states[-1]
            valid_length = min(self.max_token_length, max(0, emb.shape[1] - 217))
            if valid_length > 0:
                embs[idx, :valid_length] = emb[0, 217 : 217 + valid_length].to(device=device, dtype=dtype)
                masks[idx, :valid_length] = 1

        if getattr(self, "_offload_device", None) is not None:
            self.text_encoder.to("cpu")

        return embs, masks

    @staticmethod
    def _pack_latents(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def _unpack_latents(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def _prepare_img_ids(
        batch_size: int,
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
        axis0: float = 0.0,
    ) -> torch.Tensor:
        img_ids = torch.zeros(packed_height, packed_width, 3, dtype=dtype)
        img_ids[..., 0] = axis0
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_height, dtype=dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_width, dtype=dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        return img_ids.to(device=device, dtype=dtype)

    @staticmethod
    def process_diff_norm(diff_norm: torch.Tensor, k: float) -> torch.Tensor:
        pow_result = torch.pow(diff_norm, k)
        return torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )

    @staticmethod
    def _resize_image(img: Image.Image, img_size: int = 1024) -> tuple[Image.Image, tuple[int, int]]:
        width, height = img.size
        ratio = width / height
        if width > height:
            width_new = math.ceil(math.sqrt(img_size * img_size * ratio))
            height_new = math.ceil(width_new / ratio)
        else:
            height_new = math.ceil(math.sqrt(img_size * img_size / ratio))
            width_new = math.ceil(height_new * ratio)
        height_new = max(16, height_new // 16 * 16)
        width_new = max(16, width_new // 16 * 16)
        return img.resize((width_new, height_new), Image.LANCZOS), img.size

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        ref_images_raw: Optional[torch.Tensor],
        edit_type: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_batch = [prompt, negative_prompt]

        if ref_images_raw is None or edit_type == 0:
            ref_images = [None, None]
            edit_types = [0, 0]
        else:
            pil_image = self._tensor_to_pil(ref_images_raw[0])
            ref_images = [pil_image, pil_image]
            edit_types = [edit_type, edit_type]

        txt, mask = self._get_qwenvl_embeds(
            prompt_batch,
            ref_images,
            edit_types,
            device=self._execution_device if isinstance(self._execution_device, torch.device) else torch.device(self._execution_device),
            dtype=self.text_encoder.dtype,
        )
        return txt, mask

    def _encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        vae_device = self.vae.device
        vae_dtype = self.vae.dtype
        image = image.to(device=vae_device, dtype=vae_dtype)
        autocast_context = (
            torch.autocast(device_type=vae_device.type, enabled=False)
            if vae_device.type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )
        with autocast_context:
            encoded = self.vae.encode(image * 2 - 1)
        if hasattr(encoded, "latents"):
            return encoded.latents
        if hasattr(encoded, "latent_dist"):
            latent_dist = encoded.latent_dist
            if torch.is_tensor(latent_dist):
                return latent_dist
            if hasattr(latent_dist, "sample"):
                return latent_dist.sample()
        if isinstance(encoded, tuple):
            return encoded[0]
        raise ValueError("Unsupported VAE encoder output for RealRestorer.")

    def _decode_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_device = self.vae.device
        vae_dtype = self.vae.dtype
        latents = latents.to(device=vae_device, dtype=vae_dtype)
        autocast_context = (
            torch.autocast(device_type=vae_device.type, enabled=False)
            if vae_device.type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )
        with autocast_context:
            decoded = self.vae.decode(latents)
        if hasattr(decoded, "sample"):
            return decoded.sample
        if isinstance(decoded, tuple):
            return decoded[0]
        raise ValueError("Unsupported VAE decoder output for RealRestorer.")

    @staticmethod
    def _autocast_context(device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _denoise_edit(
        self,
        latents: torch.Tensor,
        ref_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: Sequence[float],
        guidance_scale: float,
        timesteps_truncate: float,
        process_norm_power: float,
    ) -> torch.Tensor:
        for t in timesteps[:-1]:
            latent_model_input = latents.repeat(2, 1, 1) if guidance_scale != -1 else latents
            ref_model_input = ref_latents.repeat(latent_model_input.shape[0], 1, 1)
            model_input = torch.cat([latent_model_input, ref_model_input], dim=1)
            t_vec = torch.full((model_input.shape[0],), float(t), dtype=model_input.dtype, device=model_input.device)
            guidance_vec = torch.full(
                (model_input.shape[0],),
                self.model_guidance,
                dtype=model_input.dtype,
                device=model_input.device,
            )
            pred_full = self.transformer(
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                prompt_embeds_mask=prompt_mask,
                timestep=t_vec,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=guidance_vec,
                return_dict=False,
            )[0]
            pred = pred_full[:, : latents.shape[1]]
            if guidance_scale != -1:
                cond, uncond = pred.chunk(2, dim=0)
                if float(t) > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=2, keepdim=True)
                    pred = uncond + guidance_scale * (cond - uncond) / self.process_diff_norm(
                        diff_norm, k=process_norm_power
                    )
                else:
                    pred = uncond + guidance_scale * (cond - uncond)
            latents = self.scheduler.step(pred, t, latents, return_dict=False)[0]
        return latents

    def _denoise_t2i(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: Sequence[float],
        guidance_scale: float,
        timesteps_truncate: float,
        process_norm_power: float,
    ) -> torch.Tensor:
        for t in timesteps[:-1]:
            latent_model_input = latents.repeat(2, 1, 1) if guidance_scale != -1 else latents
            t_vec = torch.full(
                (latent_model_input.shape[0],),
                float(t),
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            guidance_vec = torch.full(
                (latent_model_input.shape[0],),
                self.model_guidance,
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                prompt_embeds_mask=prompt_mask,
                timestep=t_vec,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=guidance_vec,
                return_dict=False,
            )[0]
            if guidance_scale != -1:
                cond, uncond = pred.chunk(2, dim=0)
                if float(t) > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=2, keepdim=True)
                    pred = uncond + guidance_scale * (cond - uncond) / self.process_diff_norm(
                        diff_norm, k=process_norm_power
                    )
                else:
                    pred = uncond + guidance_scale * (cond - uncond)
            latents = self.scheduler.step(pred, t, latents, return_dict=False)[0]
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        image: Optional[Union[Image.Image, torch.Tensor, np.ndarray, str]] = None,
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        size_level: int = 1024,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pil",
        timesteps_truncate: float = 0.93,
        process_norm_power: float = 0.4,
        return_dict: bool = True,
    ):
        device = self._execution_device if isinstance(self._execution_device, torch.device) else torch.device(self._execution_device)

        if generator is None and seed is not None:
            generator_device = device if device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        if image is None:
            task_type = "t2i"
            if width is None:
                width = size_level
            if height is None:
                height = size_level
            original_size = (width, height)
            ref_images_raw = None
            ref_latents = None
        else:
            task_type = "edit"
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            elif isinstance(image, str):
                pil_image = Image.open(image).convert("RGB")
            else:
                tensor_image = self.load_image(image).to(device=device, dtype=torch.float32)
                pil_image = self._tensor_to_pil(tensor_image[0])
            resized_image, original_size = self._resize_image(pil_image, img_size=size_level)
            ref_images_raw = self.load_image(resized_image).to(device=device, dtype=torch.float32)
            height, width = ref_images_raw.shape[-2:]
            ref_latents_tensor = self._encode_vae_image(ref_images_raw)
            ref_latents = self._pack_latents(ref_latents_tensor.to(device=device, dtype=torch.bfloat16))

        if height is None or width is None:
            raise ValueError("Both height and width must be resolved before sampling.")

        noise = randn_tensor(
            (
                1,
                self.latent_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        latents = self._pack_latents(noise)

        prompt_embeds, prompt_mask = self._encode_prompt(
            prompt,
            negative_prompt,
            ref_images_raw,
            edit_type=1 if task_type == "edit" else 0,
        )
        txt_ids = torch.zeros(
            prompt_embeds.shape[0],
            prompt_embeds.shape[1],
            3,
            dtype=prompt_embeds.dtype,
            device=device,
        )

        packed_h = math.ceil(height / 16)
        packed_w = math.ceil(width / 16)
        img_ids = self._prepare_img_ids(
            batch_size=prompt_embeds.shape[0],
            packed_height=packed_h,
            packed_width=packed_w,
            dtype=prompt_embeds.dtype,
            device=device,
            axis0=0.0,
        )

        if task_type == "edit":
            ref_axis = 0.0 if self.version == "v1.0" else 1.0
            ref_img_ids = self._prepare_img_ids(
                batch_size=prompt_embeds.shape[0],
                packed_height=packed_h,
                packed_width=packed_w,
                dtype=prompt_embeds.dtype,
                device=device,
                axis0=ref_axis,
            )
            combined_img_ids = torch.cat([img_ids, ref_img_ids], dim=1)
        else:
            combined_img_ids = img_ids

        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            image_seq_len=latents.shape[1],
        )
        timesteps = self.scheduler.timesteps.tolist()

        with self._autocast_context(device):
            if task_type == "edit":
                latents = self._denoise_edit(
                    latents=latents,
                    ref_latents=ref_latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    img_ids=combined_img_ids,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    timesteps_truncate=timesteps_truncate,
                    process_norm_power=process_norm_power,
                )
            else:
                latents = self._denoise_t2i(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    img_ids=combined_img_ids,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    timesteps_truncate=timesteps_truncate,
                    process_norm_power=process_norm_power,
                )

        decoded = self._decode_vae_latents(self._unpack_latents(latents.float(), height, width))
        decoded = decoded.clamp(-1, 1).mul(0.5).add(0.5)
        images = [self._tensor_to_pil(img.float()).resize(original_size) for img in decoded]

        if output_type == "np":
            images = [np.asarray(img) for img in images]

        if not return_dict:
            return (images,)
        return RealRestorerPipelineOutput(images=images)


RealRestorerDiffusionPipeline = RealRestorerPipeline
