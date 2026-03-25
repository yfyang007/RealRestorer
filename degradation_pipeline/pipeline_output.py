from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image


@dataclass
class DegradationPipelineOutput:
    images: List[Image.Image]
    metadata: List[Dict[str, Any]]
