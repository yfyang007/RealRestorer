from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ...utils import BaseOutput


@dataclass
class RealRestorerPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
