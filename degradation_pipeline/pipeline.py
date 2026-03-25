from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from .pipeline_output import DegradationPipelineOutput


SUPPORTED_DEGRADATIONS = ("blur", "haze", "noise", "rain", "sr", "moire", "reflection")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_MIDAS_TRANSFORM_BY_MODEL = {
    "DPT_BEiT_L_512": "beit512_transform",
    "DPT_BEiT_L_384": "dpt_transform",
    "DPT_BEiT_B_384": "dpt_transform",
    "DPT_SwinV2_L_384": "swin384_transform",
    "DPT_SwinV2_B_384": "swin384_transform",
    "DPT_SwinV2_T_256": "swin256_transform",
    "DPT_Swin_L_384": "swin384_transform",
    "DPT_Next_ViT_L_384": "dpt_transform",
    "DPT_LeViT_224": "levit_transform",
    "DPT_Large": "dpt_transform",
    "DPT_Hybrid": "dpt_transform",
    "MiDaS": "default_transform",
    "MiDaS_small": "small_transform",
}


def _load_rgb_image(image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {image.shape}")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported, got {image.shape[0]}")
            image = image.squeeze(0)
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}")
        image = image.detach().cpu()
        if image.dtype.is_floating_point:
            image = image.clamp(0, 1)
        else:
            image = image.float().clamp(0, 255) / 255.0
        return TF.to_pil_image(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(tensor.squeeze(0).cpu())


def _iter_images_from_dir(path: Path) -> List[Path]:
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _default_moire_runtime_root() -> Path:
    return Path(__file__).resolve().parent / "moire_runtime"


def _default_moire_assets_root() -> Path:
    return _default_moire_runtime_root() / "assets"


def _default_reflection_runtime_root() -> Path:
    return Path(__file__).resolve().parent / "reflection_runtime"


def _default_reflection_assets_root() -> Path:
    return _default_reflection_runtime_root() / "assets"


def _default_moire_config_path(legacy_root: Optional[Path] = None) -> Path:
    config_root = legacy_root if legacy_root is not None else _default_moire_runtime_root()
    return config_root / "configs/moire-blending/uhdm/blending_uhdm.yaml"


def _default_moire_ckpt_dir(legacy_root: Optional[Path] = None) -> Path:
    if legacy_root is not None:
        return legacy_root / "ckp_infer"
    return _default_moire_assets_root() / "checkpoints"


def _default_moire_pattern_dir() -> Path:
    return _default_moire_assets_root() / "moire_patterns"


def _default_real_moire_dir() -> Path:
    return _default_moire_assets_root() / "real_moire"


def _default_reflection_ckpt_path(legacy_root: Optional[Path] = None) -> Path:
    if legacy_root is not None:
        return legacy_root / "checkpoints_synthesis/130_net_G.pth"
    return _default_reflection_assets_root() / "checkpoints/130_net_G.pth"


def _default_reflection_dir(legacy_root: Optional[Path] = None) -> Path:
    if legacy_root is not None:
        return legacy_root / "img/testA"
    return _default_reflection_assets_root() / "reflections"


def _resolve_local_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _normalize_reflection_root(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if (path / "checkpoints_synthesis").is_dir() and (path / "img").is_dir():
        return path
    synthesis_root = path / "Synthesis"
    if (synthesis_root / "checkpoints_synthesis").is_dir() and (synthesis_root / "img").is_dir():
        return synthesis_root
    return path


def _try_import_j1():
    try:
        from scipy.special import j1

        return j1
    except Exception:
        return None


def _require_random_noise():
    try:
        from skimage.util import random_noise

        return random_noise
    except Exception as exc:
        raise ImportError(
            "Noise degradation requires scikit-image. Install it from requirements.degradation.txt."
        ) from exc


class _MidasDepthEstimator:
    def __init__(
        self,
        device: torch.device,
        model_type: str = "DPT_Large",
        repo_or_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = device
        self.model_type = model_type
        self.repo_or_dir = str(repo_or_dir) if repo_or_dir is not None else "isl-org/MiDaS"
        self._midas = None
        self._transform = None
        self._resolved_transform_name = _MIDAS_TRANSFORM_BY_MODEL.get(self.model_type, "dpt_transform")

    def _dependency_available(self, module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def _hub_source(self) -> str:
        return "local" if Path(self.repo_or_dir).exists() else "github"

    def _ensure_loaded(self) -> None:
        if self._midas is not None:
            return

        if not self._dependency_available("timm"):
            raise RuntimeError(
                "MiDaS dependency is missing. Install `timm` from requirements.degradation.txt "
                "to enable haze/rain depth estimation."
            )

        hub_kwargs = {"source": self._hub_source()}
        if hub_kwargs["source"] == "github":
            hub_kwargs["trust_repo"] = True

        try:
            self._midas = torch.hub.load(self.repo_or_dir, self.model_type, **hub_kwargs).to(self.device)
            self._midas.eval()
            transforms_bundle = torch.hub.load(self.repo_or_dir, "transforms", **hub_kwargs)
            self._transform = getattr(transforms_bundle, self._resolved_transform_name)
        except AttributeError as exc:
            raise RuntimeError(
                "MiDaS transform resolution failed for model type "
                f"`{self.model_type}` using transform `{self._resolved_transform_name}`."
            ) from exc
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load MiDaS because dependency `{exc.name}` is missing. "
                "Install `timm` and the rest of requirements.degradation.txt."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Failed to load MiDaS from "
                f"`{self.repo_or_dir}` with model `{self.model_type}`."
            ) from exc

    def get_config(self) -> Dict[str, Any]:
        return {
            "repo_or_dir": self.repo_or_dir,
            "hub_source": self._hub_source(),
            "model_type": self.model_type,
            "transform": self._resolved_transform_name,
        }

    def get_depth_map(self, img_rgb: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        assert self._transform is not None
        assert self._midas is not None
        input_tensor = self._transform(img_rgb).to(self.device)
        with torch.no_grad():
            depth_map = self._midas(input_tensor)
        depth_map = depth_map.squeeze().detach().cpu().numpy()
        depth_map = cv2.resize(depth_map, (img_rgb.shape[1], img_rgb.shape[0]))
        min_val = float(depth_map.min())
        max_val = float(depth_map.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(depth_map, dtype=np.float32)
        return ((depth_map - min_val) / (max_val - min_val)).astype(np.float32)


class BlurDegrader:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._j1 = _try_import_j1()

    def circular_lowpass_kernel(self, cutoff: float, kernel_size: int) -> torch.Tensor:
        if self._j1 is None:
            raise RuntimeError("scipy is required for sinc blur kernels.")
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        center = (kernel_size - 1) / 2
        x, y = np.indices((kernel_size, kernel_size))
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        r[r == 0] = 1e-9
        kernel_np = (cutoff * self._j1(cutoff * r)) / (2 * np.pi * r)
        kernel_np[int(center), int(center)] = cutoff**2 / (4 * np.pi)
        kernel_np /= np.sum(kernel_np)
        return torch.from_numpy(kernel_np).float()

    def apply_blur(self, img_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad_size = (kernel.shape[0] - 1) // 2
        padded = F.pad(img_tensor, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        kernel = kernel.view(1, 1, *kernel.shape).repeat(padded.shape[1], 1, 1, 1)
        return F.conv2d(padded, kernel, padding=0, groups=padded.shape[1])

    def add_gaussian_noise(self, img_tensor: torch.Tensor, noise_range: Tuple[float, float] = (1, 20)) -> torch.Tensor:
        sigma = random.uniform(*noise_range) / 255.0
        return torch.clamp(img_tensor + torch.randn_like(img_tensor) * sigma, 0, 1)

    def add_poisson_noise(
        self, img_tensor: torch.Tensor, scale_range: Tuple[float, float] = (0.05, 2.0)
    ) -> torch.Tensor:
        scale = random.uniform(*scale_range)
        return torch.clamp(torch.poisson(img_tensor * scale) / scale, 0, 1)

    def apply_jpeg_compression(
        self, img_tensor: torch.Tensor, quality_range: Tuple[int, int] = (30, 95)
    ) -> Tuple[torch.Tensor, int]:
        quality = random.randint(*quality_range)
        buffer = BytesIO()
        _tensor_to_pil(img_tensor).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg_pil = Image.open(buffer).convert("RGB")
        return _pil_to_tensor(img_jpeg_pil).to(img_tensor.device), quality

    def apply(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        stage1_log: Dict[str, Any] = {}
        stage2_log: Dict[str, Any] = {}
        second_stage_applied = False

        lq_tensor = _pil_to_tensor(image).to(self.device)
        original_size = (lq_tensor.shape[2], lq_tensor.shape[3])

        kernel_size1 = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
        if self._j1 is not None and random.random() < 0.5:
            cutoff = random.uniform(np.pi / 4, np.pi)
            kernel1 = self.circular_lowpass_kernel(cutoff, kernel_size1).to(self.device)
            stage1_log["blur_type"] = f"sinc(cutoff={cutoff:.2f})"
        else:
            sigma = random.uniform(0.5, 6.0)
            kernel_1d = cv2.getGaussianKernel(kernel_size1, sigma)
            kernel1 = torch.from_numpy(np.outer(kernel_1d, kernel_1d)).float().to(self.device)
            stage1_log["blur_type"] = f"gaussian(sigma={sigma:.2f})"
        lq_tensor = self.apply_blur(lq_tensor, kernel1)
        stage1_log["blur_kernel_size"] = kernel_size1

        scale1 = random.uniform(2, 4)
        mode1 = random.choice(["area", "bilinear", "bicubic"])
        lq_tensor = F.interpolate(lq_tensor, scale_factor=1 / scale1, mode=mode1, antialias=(mode1 != "area"))
        stage1_log["resize"] = f"factor={1 / scale1:.2f}, mode={mode1}"

        if random.random() < 0.75:
            noise_range = (1, 25)
            lq_tensor = self.add_gaussian_noise(lq_tensor, noise_range=noise_range)
            stage1_log["noise"] = f"gaussian(range={noise_range})"
        else:
            scale_range = (0.05, 2.5)
            lq_tensor = self.add_poisson_noise(lq_tensor, scale_range=scale_range)
            stage1_log["noise"] = f"poisson(range={scale_range})"

        lq_tensor, jpeg_q1 = self.apply_jpeg_compression(lq_tensor, quality_range=(20, 80))
        stage1_log["jpeg_quality"] = jpeg_q1

        if random.random() < 0.7:
            second_stage_applied = True
            kernel_size2 = random.choice([7, 9, 11, 13])
            if self._j1 is not None and random.random() < 0.5:
                cutoff = random.uniform(np.pi / 4, np.pi)
                kernel2 = self.circular_lowpass_kernel(cutoff, kernel_size2).to(self.device)
                stage2_log["blur_type"] = f"sinc(cutoff={cutoff:.2f})"
            else:
                sigma = random.uniform(0.5, 5.0)
                kernel_1d = cv2.getGaussianKernel(kernel_size2, sigma)
                kernel2 = torch.from_numpy(np.outer(kernel_1d, kernel_1d)).float().to(self.device)
                stage2_log["blur_type"] = f"gaussian(sigma={sigma:.2f})"
            lq_tensor = self.apply_blur(lq_tensor, kernel2)
            stage2_log["blur_kernel_size"] = kernel_size2
            scale2 = random.uniform(3, 5)
            mode2 = random.choice(["area", "bilinear", "bicubic"])
            lq_tensor = F.interpolate(lq_tensor, scale_factor=1 / scale2, mode=mode2, antialias=(mode2 != "area"))
            stage2_log["resize"] = f"factor={1 / scale2:.2f}, mode={mode2}"

        lq_tensor = F.interpolate(lq_tensor, size=original_size, mode="bicubic", antialias=True)
        metadata: Dict[str, Any] = {
            "source": "pipeline/blur_degradation.py",
            "type": "blur",
            "method": "realesrgan_style",
            "stage1": stage1_log,
            "score": 1.0 if second_stage_applied else 0.5,
        }
        if stage2_log:
            metadata["stage2"] = stage2_log
        return _tensor_to_pil(lq_tensor), metadata


class HazeDegrader:
    def __init__(
        self,
        depth_estimator: _MidasDepthEstimator,
        fog_texture_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.depth_estimator = depth_estimator
        self.fog_textures: List[Tuple[str, np.ndarray]] = []
        if fog_texture_dir is not None:
            fog_dir = Path(fog_texture_dir)
            if fog_dir.is_dir():
                for path in _iter_images_from_dir(fog_dir):
                    texture = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                    self.fog_textures.append((path.name, texture))

    def synthesize_depth_haze(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clear = img_rgb.astype(np.float32) / 255.0
        depth_inverse = self.depth_estimator.get_depth_map(img_rgb)
        depth_proxy = 1.0 - depth_inverse
        gamma = random.uniform(1.0, 2.0)
        depth_term = np.clip(depth_proxy, 0, 1) ** gamma
        beta = random.uniform(0.6, 1.8)
        transmission = np.exp(-beta * depth_term)
        a_base = np.interp(beta, [0.3, 3.0], [0.78, 0.95])
        jitter = np.random.normal(0, 0.01, size=3)
        bias = np.array([-0.005, 0.0, 0.005]) * random.uniform(0, 1)
        atmosphere = np.clip(a_base + jitter + bias, 0.7, 1.0)
        transmission = cv2.GaussianBlur(transmission, (0, 0), 1.0)
        transmission = np.clip(transmission, 0.05, 1.0)
        hazy = clear * transmission[..., None] + atmosphere * (1 - transmission[..., None])
        output = (np.clip(hazy, 0, 1) * 255).astype(np.uint8)
        metadata = {
            "gamma": round(gamma, 4),
            "beta": round(beta, 4),
            "atmosphere_light": [round(float(value), 4) for value in atmosphere.tolist()],
        }
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR), metadata

    def overlay_fog_texture(self, base_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.fog_textures:
            return base_img, {"textures": [], "alpha": 0.0}
        h, w, _ = base_img.shape
        sample_count = random.randint(1, min(10, len(self.fog_textures)))
        selected = random.sample(self.fog_textures, k=sample_count)
        composite_fog = np.zeros((h, w, 3), dtype=np.float32)
        texture_names: List[str] = []
        for name, texture_np in selected:
            texture_names.append(name)
            resized = cv2.resize(texture_np, (w, h), interpolation=cv2.INTER_LINEAR)
            composite_fog = 1 - (1 - composite_fog) * (1 - resized[..., :3])
        alpha = random.uniform(0.25, 0.6)
        base = base_img.astype(np.float32) / 255.0
        blended = 1 - (1 - base) * (1 - composite_fog * alpha)
        return (np.clip(blended, 0, 1) * 255).astype(np.uint8), {"textures": texture_names, "alpha": round(alpha, 4)}

    def apply(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        input_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        use_depth_haze = random.random() > 0.2
        use_texture_haze = bool(self.fog_textures) and random.random() > 0.3
        if not use_depth_haze and not use_texture_haze:
            use_depth_haze = True

        degraded = input_bgr
        metadata: Dict[str, Any] = {
            "source": "pipeline/haze_degradation.py",
            "type": "haze",
            "method": "depth_plus_texture",
            "midas": self.depth_estimator.get_config(),
            "used_depth_haze": use_depth_haze,
            "used_texture_haze": use_texture_haze,
        }

        if use_depth_haze:
            degraded, depth_meta = self.synthesize_depth_haze(degraded)
            metadata["depth_haze"] = depth_meta
        if use_texture_haze:
            degraded, texture_meta = self.overlay_fog_texture(degraded)
            metadata["texture_haze"] = texture_meta

        return Image.fromarray(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)), metadata


class NoiseDegrader:
    gaussian_var_range = (0.005, 0.1)
    salt_pepper_amount_range = (0.005, 0.05)
    granular_strength_range = (0.6, 2.4)
    density_avg_kernel_range = (3, 7)
    density_avg_threshold_range = (0.4, 0.8)

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._random_noise = _require_random_noise()
        self._j1 = _try_import_j1()

    def add_gaussian_noise(self, image_np: np.ndarray, log_list: List[str]) -> np.ndarray:
        variance = random.uniform(*self.gaussian_var_range)
        log_list.append(f"gaussian(var={variance:.4f})")
        return self._random_noise(image_np, mode="gaussian", var=variance)

    def add_salt_pepper_noise(self, image_np: np.ndarray, log_list: List[str]) -> np.ndarray:
        amount = random.uniform(*self.salt_pepper_amount_range)
        log_list.append(f"s&p(amount={amount:.4f})")
        return self._random_noise(image_np, mode="s&p", amount=amount)

    def add_poisson_noise(self, image_np: np.ndarray, log_list: List[str]) -> np.ndarray:
        log_list.append("poisson")
        return self._random_noise(image_np, mode="poisson")

    def add_granular_noise(self, arr_uint8: np.ndarray, log_list: List[str]) -> np.ndarray:
        strength = random.uniform(*self.granular_strength_range)
        arr_float = arr_uint8.astype(np.float32)
        h, w, c = arr_uint8.shape
        rng = np.random.default_rng()
        sigma = 8.0 * strength
        luma_noise = rng.normal(0, sigma, size=(h, w, 1))
        chroma_noise = rng.normal(0, sigma * 0.35, size=(h, w, c))
        log_list.append(f"granular(strength={strength:.3f})")
        return arr_float + luma_noise + chroma_noise

    def apply_density_based_averaging(self, arr_uint8: np.ndarray, log_list: List[str]) -> np.ndarray:
        kernel_size = random.choice(
            range(self.density_avg_kernel_range[0], self.density_avg_kernel_range[1] + 1, 2)
        )
        threshold = random.uniform(*self.density_avg_threshold_range)
        gray_img = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        local_mean = cv2.boxFilter(gray_img, -1, (kernel_size, kernel_size))
        variance_map = cv2.boxFilter((gray_img - local_mean) ** 2, -1, (kernel_size, kernel_size))
        high_density_mask = variance_map > variance_map.mean() * threshold
        if not np.any(high_density_mask):
            log_list.append("density_avg_skipped")
            return arr_uint8
        blurred_arr = cv2.blur(arr_uint8, (kernel_size, kernel_size))
        result_arr = arr_uint8.copy()
        result_arr[high_density_mask] = blurred_arr[high_density_mask]
        log_list.append(f"density_avg(k={kernel_size}, th={threshold:.2f})")
        return result_arr

    def circular_lowpass_kernel(self, cutoff: float, kernel_size: int) -> torch.Tensor:
        if self._j1 is None:
            raise RuntimeError("scipy is required for sinc blur kernels.")
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        center = (kernel_size - 1) / 2
        x, y = np.indices((kernel_size, kernel_size))
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        r_eps = r + 1e-9
        kernel_np = (cutoff * self._j1(cutoff * r_eps)) / (2 * np.pi * r_eps)
        kernel_np[r == 0] = cutoff**2 / (4 * np.pi)
        kernel_np /= np.sum(kernel_np)
        return torch.from_numpy(kernel_np).float()

    def apply_blur(self, img_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kernel = kernel.view(1, 1, *kernel.shape).repeat(img_tensor.shape[1], 1, 1, 1)
        pad_size = (kernel.shape[-1] - 1) // 2
        padded = F.pad(img_tensor, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        return F.conv2d(padded, kernel, padding=0, groups=img_tensor.shape[1])

    def apply_jpeg_compression(
        self, img_tensor: torch.Tensor, quality_range: Tuple[int, int] = (30, 95)
    ) -> Tuple[torch.Tensor, int]:
        quality = random.randint(*quality_range)
        buffer = BytesIO()
        _tensor_to_pil(img_tensor).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg_pil = Image.open(buffer).convert("RGB")
        return _pil_to_tensor(img_jpeg_pil).to(img_tensor.device), quality

    def apply_realesrgan_style_degradation(self, image: Image.Image) -> Tuple[Image.Image, str]:
        degradation_log: List[str] = []
        lq_tensor = _pil_to_tensor(image).to(self.device)
        original_size = (lq_tensor.shape[2], lq_tensor.shape[3])

        kernel_size1 = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
        if self._j1 is not None and random.random() < 0.5:
            cutoff = np.random.uniform(np.pi / 4, np.pi)
            kernel1 = self.circular_lowpass_kernel(cutoff, kernel_size1).to(self.device)
            blur_type = f"sinc(cutoff={cutoff:.2f})"
        else:
            sigma = np.random.uniform(0.5, 6.0)
            kernel_1d = cv2.getGaussianKernel(kernel_size1, sigma)
            kernel1 = torch.from_numpy(np.outer(kernel_1d, kernel_1d)).float().to(self.device)
            blur_type = f"gaussian(sigma={sigma:.2f})"
        lq_tensor = self.apply_blur(lq_tensor, kernel1)
        degradation_log.append(f"blur_1(type={blur_type}, size={kernel_size1})")

        scale1 = random.uniform(2, 4)
        mode1 = random.choice(["area", "bilinear", "bicubic"])
        lq_tensor = F.interpolate(lq_tensor, scale_factor=1 / scale1, mode=mode1, antialias=(mode1 != "area"))
        degradation_log.append(f"resize_1(factor={1 / scale1:.2f}, mode={mode1})")

        lq_tensor, jpeg_q1 = self.apply_jpeg_compression(lq_tensor, quality_range=(20, 80))
        degradation_log.append(f"jpeg_1(q={jpeg_q1})")

        if random.random() < 0.7:
            kernel_size2 = random.choice([7, 9, 11, 13])
            if self._j1 is not None and random.random() < 0.5:
                cutoff = np.random.uniform(np.pi / 4, np.pi)
                kernel2 = self.circular_lowpass_kernel(cutoff, kernel_size2).to(self.device)
                blur_type = f"sinc(cutoff={cutoff:.2f})"
            else:
                sigma = np.random.uniform(0.5, 5.0)
                kernel_1d = cv2.getGaussianKernel(kernel_size2, sigma)
                kernel2 = torch.from_numpy(np.outer(kernel_1d, kernel_1d)).float().to(self.device)
                blur_type = f"gaussian(sigma={sigma:.2f})"
            lq_tensor = self.apply_blur(lq_tensor, kernel2)
            degradation_log.append(f"blur_2(type={blur_type}, size={kernel_size2})")
            scale2 = random.uniform(3, 5)
            mode2 = random.choice(["area", "bilinear", "bicubic"])
            lq_tensor = F.interpolate(lq_tensor, scale_factor=1 / scale2, mode=mode2, antialias=(mode2 != "area"))
            degradation_log.append(f"resize_2(factor={1 / scale2:.2f}, mode={mode2})")

        if lq_tensor.shape[2] != original_size[0] or lq_tensor.shape[3] != original_size[1]:
            lq_tensor = F.interpolate(lq_tensor, size=original_size, mode="bicubic", antialias=True)
            degradation_log.append("final_resize_to_original")

        return _tensor_to_pil(torch.clamp(lq_tensor, 0, 1)), "realesrgan_style(" + " -> ".join(degradation_log) + ")"

    def apply(
        self,
        image: Image.Image,
        enable_density_averaging: bool = True,
        enable_realesrgan_degradation: bool = True,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        image_np_uint8 = np.array(image.convert("RGB"))
        degradation_log: List[str] = []

        def granular_wrapper(img_np: np.ndarray, log_list: List[str]) -> np.ndarray:
            noisy = self.add_granular_noise((img_np * 255).astype(np.uint8), log_list)
            return np.clip(noisy, 0, 255).astype(np.uint8) / 255.0

        noise_functions = [
            self.add_gaussian_noise,
            self.add_salt_pepper_noise,
            self.add_poisson_noise,
            granular_wrapper,
        ]
        num_noises_to_apply = random.randint(2, len(noise_functions))
        chosen_functions = random.sample(noise_functions, k=num_noises_to_apply)

        image_np_float = image_np_uint8 / 255.0
        for noise_func in chosen_functions:
            image_np_float = noise_func(image_np_float, degradation_log)

        noisy_image_uint8 = (np.clip(image_np_float, 0, 1) * 255).astype(np.uint8)
        if enable_density_averaging and random.random() < 0.4:
            noisy_image_uint8 = self.apply_density_based_averaging(noisy_image_uint8, degradation_log)

        lq_image = Image.fromarray(noisy_image_uint8)
        if enable_realesrgan_degradation and random.random() < 0.5:
            lq_image, realesrgan_log = self.apply_realesrgan_style_degradation(lq_image)
            degradation_log.append(realesrgan_log)

        metadata = {
            "source": "pipeline/noise_degradation.py",
            "type": "noise",
            "method": "composite_noise",
            "enable_density_averaging": enable_density_averaging,
            "enable_realesrgan_degradation": enable_realesrgan_degradation,
            "degradation_chain": degradation_log,
        }
        return lq_image, metadata


class RainDegrader:
    def __init__(
        self,
        depth_estimator: _MidasDepthEstimator,
        rain_texture_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.depth_estimator = depth_estimator
        self.rain_textures: List[Tuple[str, np.ndarray]] = []
        if rain_texture_dir is not None:
            rain_dir = Path(rain_texture_dir)
            if rain_dir.is_dir():
                for path in _iter_images_from_dir(rain_dir):
                    self.rain_textures.append((path.name, np.array(Image.open(path).convert("RGB"))))

    def procedural_rain(
        self,
        img_shape: Sequence[int],
        depth_map: np.ndarray,
        num_events: int,
        length_range: Tuple[int, int],
        global_angle: float,
        splash_chance: float,
        motion_blur_chance: float,
    ) -> np.ndarray:
        height, width = img_shape[:2]
        rain_canvas = np.zeros(img_shape, dtype=np.float32)
        for _ in range(num_events):
            y1 = np.random.randint(0, height)
            x1 = np.random.randint(0, width)
            current_depth = float(depth_map[y1, x1])
            event_type = random.random()
            if event_type < splash_chance and random.random() < current_depth**1.5:
                num_splats = random.randint(3, 6)
                splash_radius = int(3 + 8 * (1 - current_depth))
                alpha = np.random.uniform(0.6, 1.0) * current_depth
                for _ in range(num_splats):
                    offset_x = random.randint(-splash_radius, splash_radius)
                    offset_y = random.randint(-splash_radius, splash_radius)
                    px = min(width - 1, max(0, x1 + offset_x))
                    py = min(height - 1, max(0, y1 + offset_y))
                    rain_canvas[py, px] = np.minimum(1.0, rain_canvas[py, px] + alpha)
            else:
                length = int(length_range[0] + (length_range[1] - length_range[0]) * current_depth)
                rad_angle = np.deg2rad(global_angle)
                thickness = random.choice([1, 1, 1, 2])
                x2 = int(x1 + length * np.cos(rad_angle))
                y2 = int(y1 + length * np.sin(rad_angle))
                color_val = np.random.uniform(0.7, 0.95)
                color = (color_val, color_val, color_val)
                cv2.line(rain_canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                if random.random() < motion_blur_chance:
                    for _ in range(3):
                        ox = random.randint(-thickness, thickness)
                        oy = random.randint(-thickness, thickness)
                        cv2.line(
                            rain_canvas,
                            (x1 + ox, y1 + oy),
                            (x2 + ox, y2 + oy),
                            tuple(channel * 0.2 for channel in color),
                            thickness,
                            cv2.LINE_AA,
                        )
        return rain_canvas

    def overlay_texture(self, base_img: np.ndarray, texture_np: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w, _ = base_img.shape
        texture_resized = cv2.resize(texture_np, (w, h), interpolation=cv2.INTER_LINEAR)
        texture_normalized = texture_resized[..., :3]
        if float(np.mean(texture_normalized)) > 127:
            texture_normalized = 255 - texture_normalized
        alpha = random.uniform(0.25, 0.6)
        base = base_img.astype(np.float32) / 255.0
        texture_float = texture_normalized.astype(np.float32) / 255.0
        blended = 1 - (1 - base) * (1 - texture_float * alpha)
        return np.clip(blended, 0, 1), alpha

    def apply(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        input_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        img_float = input_rgb.astype(np.float32) / 255.0

        num_events = random.randint(8000, 15000)
        global_angle = random.uniform(75, 105)
        surface_splash_chance = random.uniform(0.02, 0.07)
        use_texture = bool(self.rain_textures) and random.random() < 0.5

        depth_map = self.depth_estimator.get_depth_map(input_rgb)
        rain_layer = self.procedural_rain(
            img_float.shape,
            depth_map=depth_map,
            num_events=num_events,
            length_range=(5, 50),
            global_angle=global_angle,
            splash_chance=surface_splash_chance,
            motion_blur_chance=0.4,
        )

        final_img_float = np.clip(img_float + rain_layer, 0, 1)
        texture_name = None
        texture_alpha = None
        if use_texture:
            texture_name, texture_to_use = random.choice(self.rain_textures)
            final_img_float, texture_alpha = self.overlay_texture((final_img_float * 255).astype(np.uint8), texture_to_use)

        final_img_uint8 = (final_img_float * 255).astype(np.uint8)
        metadata = {
            "source": "pipeline/rain_degradation.py",
            "type": "rain",
            "method": "depth_aware_rain",
            "midas": self.depth_estimator.get_config(),
            "num_events": num_events,
            "global_angle": round(global_angle, 4),
            "surface_splash_chance": round(surface_splash_chance, 4),
            "used_texture": use_texture,
        }
        if texture_name is not None:
            metadata["texture_name"] = texture_name
            metadata["texture_alpha"] = round(float(texture_alpha), 4)
        return Image.fromarray(final_img_uint8), metadata


class SRDegrader:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def add_gaussian_noise(self, img_tensor: torch.Tensor, noise_range: Tuple[int, int] = (1, 15)) -> torch.Tensor:
        sigma = random.uniform(noise_range[0], noise_range[1]) / 255.0
        return torch.clamp(img_tensor + torch.randn_like(img_tensor) * sigma, 0, 1)

    def apply_jpeg_compression(
        self, img_tensor: torch.Tensor, quality_range: Tuple[int, int] = (40, 95)
    ) -> Tuple[torch.Tensor, int]:
        quality = random.randint(quality_range[0], quality_range[1])
        buffer = BytesIO()
        _tensor_to_pil(img_tensor).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg_pil = Image.open(buffer).convert("RGB")
        return _pil_to_tensor(img_jpeg_pil).to(img_tensor.device), quality

    def apply(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        degradation_log: List[str] = []
        gt_tensor = _pil_to_tensor(image).to(self.device)
        lq_tensor = gt_tensor.clone()
        original_h, original_w = gt_tensor.shape[2], gt_tensor.shape[3]

        scale = random.uniform(0.5, 0.9)
        if min(original_h, original_w) < 512:
            scale = random.uniform(0.8, 1.0)
        if scale < 1.0:
            mode_down = random.choice(["area", "bilinear", "bicubic"])
            use_antialias = mode_down in ["bilinear", "bicubic"]
            lq_tensor = F.interpolate(lq_tensor, scale_factor=scale, mode=mode_down, antialias=use_antialias)
            degradation_log.append(f"resize_down(scale={scale:.2f}, mode={mode_down})")

        if random.random() < 0.6:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 1.2)
            lq_tensor = TF.gaussian_blur(lq_tensor, kernel_size, sigma)
            degradation_log.append(f"blur(sigma={sigma:.2f})")

        if random.random() < 0.5:
            lq_tensor = self.add_gaussian_noise(lq_tensor, noise_range=(1, 10))
            degradation_log.append("noise(gaussian_light)")

        q_range = (40, 60) if random.random() < 0.2 else (60, 95)
        lq_tensor, jpeg_q = self.apply_jpeg_compression(lq_tensor, quality_range=q_range)
        degradation_log.append(f"jpeg(q={jpeg_q})")

        if lq_tensor.shape[2] != original_h or lq_tensor.shape[3] != original_w:
            mode_up = random.choice(["bilinear", "bicubic"])
            lq_tensor = F.interpolate(lq_tensor, size=(original_h, original_w), mode=mode_up, antialias=True)
            degradation_log.append(f"resize_back(mode={mode_up})")

        metadata = {
            "source": "pipeline/sr_degradation.py",
            "type": "sr",
            "method": "web_image_simulation",
            "degradation_chain": degradation_log,
        }
        return _tensor_to_pil(torch.clamp(lq_tensor, 0, 1)), metadata


class MoireDegrader:
    def __init__(
        self,
        device: torch.device,
        unidemoire_root: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
        moire_pattern_dir: Optional[Union[str, Path]] = None,
        real_moire_dir: Optional[Union[str, Path]] = None,
        model_input_size: int = 512,
    ) -> None:
        self.device = device
        self.model_input_size = int(model_input_size)
        self.runtime_root = _default_moire_runtime_root().resolve()
        self.legacy_unidemoire_root = _resolve_local_path(unidemoire_root)
        self.config_path = _resolve_local_path(config_path) or _default_moire_config_path(self.legacy_unidemoire_root)
        self.ckpt_dir = _resolve_local_path(ckpt_dir) or _default_moire_ckpt_dir(self.legacy_unidemoire_root)
        self.moire_pattern_dir = _resolve_local_path(moire_pattern_dir) or _default_moire_pattern_dir()
        self.real_moire_dir = _resolve_local_path(real_moire_dir) or _default_real_moire_dir()
        self._imports: Optional[Dict[str, Any]] = None
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.runtime_root.is_dir():
            raise FileNotFoundError(f"Bundled moire runtime not found: {self.runtime_root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Moire config not found: {self.config_path}")
        if not self.ckpt_dir.is_dir():
            raise FileNotFoundError(f"Moire checkpoint dir not found: {self.ckpt_dir}")
        if not self.moire_pattern_dir.is_dir():
            raise FileNotFoundError(f"Moire pattern dir not found: {self.moire_pattern_dir}")
        if not self.real_moire_dir.is_dir():
            raise FileNotFoundError(f"Real moire reference dir not found: {self.real_moire_dir}")

    def _load_imports(self) -> Dict[str, Any]:
        if self._imports is not None:
            return self._imports
        try:
            from .moire_runtime import MoireBlendingInferenceModel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Moire degradation requires the bundled moire runtime dependencies. "
                "Install requirements.degradation.txt first."
            ) from exc
        self._imports = {
            "MoireBlendingInferenceModel": MoireBlendingInferenceModel,
            "transforms": transforms,
        }
        return self._imports

    def _load_model(self, ckpt_path: Path):
        imports = self._load_imports()
        MoireBlendingInferenceModel = imports["MoireBlendingInferenceModel"]
        model = MoireBlendingInferenceModel.from_config(self.config_path, ckpt_path=ckpt_path).eval().to(self.device)
        return model

    def apply(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        imports = self._load_imports()
        transforms_module = imports["transforms"]
        ckpt_files = sorted(path for path in self.ckpt_dir.iterdir() if path.suffix == ".ckpt")
        moire_pattern_paths = _iter_images_from_dir(self.moire_pattern_dir)
        real_moire_paths = _iter_images_from_dir(self.real_moire_dir)
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in {self.ckpt_dir}")
        if not moire_pattern_paths:
            raise FileNotFoundError(f"No moire pattern images found in {self.moire_pattern_dir}")
        if not real_moire_paths:
            raise FileNotFoundError(f"No real moire reference images found in {self.real_moire_dir}")

        selected_ckpt_path = random.choice(ckpt_files)
        moire_pattern_path = random.choice(moire_pattern_paths)
        real_moire_path = random.choice(real_moire_paths)
        current_model = self._load_model(selected_ckpt_path)

        transform_resize_512 = transforms_module.Compose(
            [transforms_module.Resize((self.model_input_size, self.model_input_size), antialias=True), transforms_module.ToTensor()]
        )
        transform_resize_256 = transforms_module.Compose(
            [transforms_module.Resize((256, 256), antialias=True), transforms_module.ToTensor()]
        )

        original_size = image.size
        natural_image_tensor = transform_resize_512(image).unsqueeze(0).to(self.device)
        moire_pattern_tensor = transform_resize_512(Image.open(moire_pattern_path).convert("RGB")).unsqueeze(0).to(self.device)
        real_moire_tensor = transform_resize_256(Image.open(real_moire_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, result_tensor = current_model(moire_pattern_tensor, natural_image_tensor, real_moire_tensor)

        degraded_image = transforms_module.ToPILImage()(result_tensor.squeeze(0).detach().cpu()).convert("RGB")
        if degraded_image.size != original_size:
            degraded_image = degraded_image.resize(original_size, Image.Resampling.LANCZOS)

        del current_model, natural_image_tensor, moire_pattern_tensor, real_moire_tensor, result_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata = {
            "source": "degradation_pipeline/moire_runtime",
            "type": "moire",
            "method": "unidemoire_blending",
            "ckpt": selected_ckpt_path.name,
            "moire_pattern": moire_pattern_path.name,
            "real_moire_reference": real_moire_path.name,
            "config_path": str(self.config_path),
            "runtime_root": str(self.runtime_root),
            "ckpt_dir": str(self.ckpt_dir),
            "moire_pattern_dir": str(self.moire_pattern_dir),
            "real_moire_dir": str(self.real_moire_dir),
        }
        if self.legacy_unidemoire_root is not None:
            metadata["legacy_unidemoire_root"] = str(self.legacy_unidemoire_root)
        return degraded_image, metadata


class ReflectionDegrader:
    _REFLECTION_TYPES = ("focused", "defocused", "ghosting")

    def __init__(
        self,
        device: torch.device,
        reflection_root: Optional[Union[str, Path]] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
        reflection_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = device
        self.runtime_root = _default_reflection_runtime_root().resolve()
        self.legacy_reflection_root = _normalize_reflection_root(_resolve_local_path(reflection_root))
        self.ckpt_path = _resolve_local_path(ckpt_path) or _default_reflection_ckpt_path(self.legacy_reflection_root)
        self.reflection_dir = _resolve_local_path(reflection_dir) or _default_reflection_dir(self.legacy_reflection_root)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._model = None
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.runtime_root.is_dir():
            raise FileNotFoundError(f"Bundled reflection runtime not found: {self.runtime_root}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Reflection checkpoint not found: {self.ckpt_path}")
        if not self.reflection_dir.is_dir():
            raise FileNotFoundError(f"Reflection image dir not found: {self.reflection_dir}")

    def _get_model(self):
        if self._model is None:
            try:
                from .reflection_runtime import ReflectionSynthesisInferenceModel
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "Reflection degradation requires the bundled reflection runtime dependencies. "
                    "Install requirements.degradation.txt first."
                ) from exc
            self._model = ReflectionSynthesisInferenceModel(ckpt_path=self.ckpt_path).eval().to(self.device)
        return self._model

    def _resolve_reflection_type(self, reflection_type: Optional[str]) -> str:
        if reflection_type is None or reflection_type.lower() == "random":
            return random.choice(self._REFLECTION_TYPES)
        resolved = reflection_type.lower()
        if resolved not in self._REFLECTION_TYPES:
            raise ValueError(
                f"Unsupported reflection type: {reflection_type}. "
                f"Choose from {self._REFLECTION_TYPES} or `random`."
            )
        return resolved

    def _process_reflection(self, image: Image.Image, reflection_type: str) -> Image.Image:
        if reflection_type == "focused":
            return image
        if reflection_type == "defocused":
            image_np = np.array(image)
            k_sz = np.linspace(5, 10, 80)
            sigma = float(k_sz[np.random.randint(0, len(k_sz))])
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma, sigma, 0)
            return Image.fromarray(blurred.astype(np.uint8))
        if reflection_type == "ghosting":
            image_np = np.array(image)
            rows, cols, _ = image_np.shape
            shift_x = np.random.randint(max(1, int(cols * 0.05)), max(2, int(cols * 0.1)) + 1)
            shift_y = np.random.randint(max(1, int(rows * 0.05)), max(2, int(rows * 0.1)) + 1)
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(image_np, matrix, (cols, rows))
            attenuation = np.random.uniform(0.5, 1.0)
            ghosted = cv2.addWeighted(image_np.astype(float), attenuation, shifted.astype(float), 1 - attenuation, 0)
            ghosted = np.clip(ghosted, 0, 255)
            ghosted_image = Image.fromarray(ghosted.astype(np.uint8))
            cropped = ghosted_image.crop((shift_x, shift_y, cols, rows))
            return cropped.resize((cols, rows), Image.Resampling.LANCZOS)
        raise ValueError(f"Unsupported reflection type: {reflection_type}")

    def apply(
        self,
        image: Image.Image,
        *,
        reflection_image: Optional[Union[str, Path, Image.Image, np.ndarray, torch.Tensor]] = None,
        reflection_type: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        if reflection_image is None:
            reflection_candidates = _iter_images_from_dir(self.reflection_dir)
            if not reflection_candidates:
                raise FileNotFoundError(f"No reflection images found in {self.reflection_dir}")
            selected_reflection_path = random.choice(reflection_candidates)
            reflection_source = _load_rgb_image(selected_reflection_path)
        else:
            selected_reflection_path = _resolve_local_path(reflection_image) if isinstance(reflection_image, (str, Path)) else None
            reflection_source = _load_rgb_image(reflection_image)

        if reflection_source.size != image.size:
            reflection_source = reflection_source.resize(image.size, Image.Resampling.LANCZOS)

        selected_reflection_type = self._resolve_reflection_type(reflection_type)
        processed_reflection = self._process_reflection(reflection_source, selected_reflection_type)

        transmission_tensor = self.transform(image).unsqueeze(0).to(self.device)
        reflection_tensor = self.transform(processed_reflection).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self._get_model()(transmission_tensor, reflection_tensor)

        mix_tensor = result["mix"].detach().cpu()
        degraded_image = TF.to_pil_image(((mix_tensor.squeeze(0).clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)).convert("RGB")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata = {
            "source": "degradation_pipeline/reflection_runtime",
            "type": "reflection",
            "method": "reflection_synthesis",
            "reflection_type": selected_reflection_type,
            "ckpt_path": str(self.ckpt_path),
            "runtime_root": str(self.runtime_root),
            "reflection_dir": str(self.reflection_dir),
        }
        if selected_reflection_path is not None:
            metadata["reflection_image"] = str(selected_reflection_path)
            metadata["reflection_image_name"] = selected_reflection_path.name
        if self.legacy_reflection_root is not None:
            metadata["legacy_reflection_root"] = str(self.legacy_reflection_root)
        return degraded_image, metadata


@contextmanager
def _temporary_seed(seed: Optional[int]):
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class _NoiseOptions:
    enable_density_averaging: bool = True
    enable_realesrgan_degradation: bool = True


class DegradationPipeline:
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        midas_model_type: str = "DPT_Large",
        midas_repo_or_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.midas_model_type = midas_model_type
        self.midas_repo_or_dir = _resolve_local_path(midas_repo_or_dir) or midas_repo_or_dir
        self._depth_estimator: Optional[_MidasDepthEstimator] = None
        self._blur_degrader: Optional[BlurDegrader] = None
        self._noise_degrader: Optional[NoiseDegrader] = None
        self._sr_degrader: Optional[SRDegrader] = None
        self._haze_degraders: Dict[Optional[str], HazeDegrader] = {}
        self._rain_degraders: Dict[Optional[str], RainDegrader] = {}
        self._moire_degraders: Dict[Tuple[str, str, str, str, str, int], MoireDegrader] = {}
        self._reflection_degraders: Dict[Tuple[str, str, str], ReflectionDegrader] = {}

    def _get_depth_estimator(self) -> _MidasDepthEstimator:
        if self._depth_estimator is None:
            self._depth_estimator = _MidasDepthEstimator(
                device=self.device,
                model_type=self.midas_model_type,
                repo_or_dir=self.midas_repo_or_dir,
            )
        return self._depth_estimator

    def _get_blur_degrader(self) -> BlurDegrader:
        if self._blur_degrader is None:
            self._blur_degrader = BlurDegrader(self.device)
        return self._blur_degrader

    def _get_noise_degrader(self) -> NoiseDegrader:
        if self._noise_degrader is None:
            self._noise_degrader = NoiseDegrader(self.device)
        return self._noise_degrader

    def _get_sr_degrader(self) -> SRDegrader:
        if self._sr_degrader is None:
            self._sr_degrader = SRDegrader(self.device)
        return self._sr_degrader

    def _get_haze_degrader(self, fog_texture_dir: Optional[Union[str, Path]]) -> HazeDegrader:
        cache_key = str(_resolve_local_path(fog_texture_dir)) if fog_texture_dir else None
        if cache_key not in self._haze_degraders:
            self._haze_degraders[cache_key] = HazeDegrader(self._get_depth_estimator(), fog_texture_dir=fog_texture_dir)
        return self._haze_degraders[cache_key]

    def _get_rain_degrader(self, rain_texture_dir: Optional[Union[str, Path]]) -> RainDegrader:
        cache_key = str(_resolve_local_path(rain_texture_dir)) if rain_texture_dir else None
        if cache_key not in self._rain_degraders:
            self._rain_degraders[cache_key] = RainDegrader(self._get_depth_estimator(), rain_texture_dir=rain_texture_dir)
        return self._rain_degraders[cache_key]

    def _get_moire_degrader(
        self,
        *,
        unidemoire_root: Optional[Union[str, Path]],
        config_path: Optional[Union[str, Path]],
        ckpt_dir: Optional[Union[str, Path]],
        moire_pattern_dir: Optional[Union[str, Path]],
        real_moire_dir: Optional[Union[str, Path]],
        model_input_size: int,
    ) -> MoireDegrader:
        resolved_root = _resolve_local_path(unidemoire_root)
        resolved = (
            str(resolved_root or ""),
            str(_resolve_local_path(config_path) or _default_moire_config_path(resolved_root)),
            str(_resolve_local_path(ckpt_dir) or _default_moire_ckpt_dir(resolved_root)),
            str(_resolve_local_path(moire_pattern_dir) or _default_moire_pattern_dir()),
            str(_resolve_local_path(real_moire_dir) or _default_real_moire_dir()),
            int(model_input_size),
        )
        if resolved not in self._moire_degraders:
            self._moire_degraders[resolved] = MoireDegrader(
                device=self.device,
                unidemoire_root=unidemoire_root,
                config_path=config_path,
                ckpt_dir=ckpt_dir,
                moire_pattern_dir=moire_pattern_dir,
                real_moire_dir=real_moire_dir,
                model_input_size=model_input_size,
            )
        return self._moire_degraders[resolved]

    def _get_reflection_degrader(
        self,
        *,
        reflection_root: Optional[Union[str, Path]],
        reflection_ckpt_path: Optional[Union[str, Path]],
        reflection_dir: Optional[Union[str, Path]],
    ) -> ReflectionDegrader:
        resolved_root = _normalize_reflection_root(_resolve_local_path(reflection_root))
        resolved = (
            str(resolved_root or ""),
            str(_resolve_local_path(reflection_ckpt_path) or _default_reflection_ckpt_path(resolved_root)),
            str(_resolve_local_path(reflection_dir) or _default_reflection_dir(resolved_root)),
        )
        if resolved not in self._reflection_degraders:
            self._reflection_degraders[resolved] = ReflectionDegrader(
                device=self.device,
                reflection_root=reflection_root,
                ckpt_path=reflection_ckpt_path,
                reflection_dir=reflection_dir,
            )
        return self._reflection_degraders[resolved]

    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        degradation_type: str,
        *,
        seed: Optional[int] = None,
        fog_texture_dir: Optional[Union[str, Path]] = None,
        rain_texture_dir: Optional[Union[str, Path]] = None,
        enable_density_averaging: bool = True,
        enable_realesrgan_degradation: bool = True,
        unidemoire_root: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
        moire_pattern_dir: Optional[Union[str, Path]] = None,
        real_moire_dir: Optional[Union[str, Path]] = None,
        model_input_size: int = 512,
        reflection_root: Optional[Union[str, Path]] = None,
        reflection_ckpt_path: Optional[Union[str, Path]] = None,
        reflection_dir: Optional[Union[str, Path]] = None,
        reflection_image: Optional[Union[str, Path, Image.Image, np.ndarray, torch.Tensor]] = None,
        reflection_type: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[DegradationPipelineOutput, Tuple[List[Image.Image], List[Dict[str, Any]]]]:
        degradation_type = degradation_type.lower()
        if degradation_type not in SUPPORTED_DEGRADATIONS:
            raise ValueError(f"Unsupported degradation type: {degradation_type}. Choose from {SUPPORTED_DEGRADATIONS}.")

        input_image = _load_rgb_image(image)
        with _temporary_seed(seed):
            if degradation_type == "blur":
                degraded_image, metadata = self._get_blur_degrader().apply(input_image)
            elif degradation_type == "haze":
                degraded_image, metadata = self._get_haze_degrader(fog_texture_dir).apply(input_image)
            elif degradation_type == "noise":
                noise_options = _NoiseOptions(
                    enable_density_averaging=enable_density_averaging,
                    enable_realesrgan_degradation=enable_realesrgan_degradation,
                )
                degraded_image, metadata = self._get_noise_degrader().apply(
                    input_image,
                    enable_density_averaging=noise_options.enable_density_averaging,
                    enable_realesrgan_degradation=noise_options.enable_realesrgan_degradation,
                )
            elif degradation_type == "rain":
                degraded_image, metadata = self._get_rain_degrader(rain_texture_dir).apply(input_image)
            elif degradation_type == "sr":
                degraded_image, metadata = self._get_sr_degrader().apply(input_image)
            elif degradation_type == "moire":
                degraded_image, metadata = self._get_moire_degrader(
                    unidemoire_root=unidemoire_root,
                    config_path=config_path,
                    ckpt_dir=ckpt_dir,
                    moire_pattern_dir=moire_pattern_dir,
                    real_moire_dir=real_moire_dir,
                    model_input_size=model_input_size,
                ).apply(input_image)
            else:
                degraded_image, metadata = self._get_reflection_degrader(
                    reflection_root=reflection_root,
                    reflection_ckpt_path=reflection_ckpt_path,
                    reflection_dir=reflection_dir,
                ).apply(
                    input_image,
                    reflection_image=reflection_image,
                    reflection_type=reflection_type,
                )

        output = DegradationPipelineOutput(images=[degraded_image], metadata=[metadata])
        if return_dict:
            return output
        return output.images, output.metadata
