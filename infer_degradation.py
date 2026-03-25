from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _bootstrap()
    from degradation_pipeline.infer import main as degrade_main

    degrade_main()


if __name__ == "__main__":
    main()
