from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degradation_pipeline import DegradationPipeline


def main() -> None:
    image_path = Path("/path/to/background.png")
    output_path = Path("degraded_output.png")

    pipe = DegradationPipeline(device="cuda:0")
    result = pipe(
        image_path,
        "reflection",
        reflection_ckpt_path="/path/to/130_net_G.pth",
        reflection_image="/path/to/reflection.png",
        reflection_type="random",
        seed=42,
    )

    result.images[0].save(output_path)
    print(result.metadata[0])


if __name__ == "__main__":
    main()
