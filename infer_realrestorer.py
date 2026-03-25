from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    local_diffusers_src = repo_root / "diffusers" / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if local_diffusers_src.is_dir() and str(local_diffusers_src) not in sys.path:
        sys.path.insert(0, str(local_diffusers_src))


def main() -> None:
    _bootstrap()
    from RealRestorer.inference import main as inference_main

    inference_main()


if __name__ == "__main__":
    main()
