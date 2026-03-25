from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    benchmark_root = repo_root / "RealIR-Bench"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if benchmark_root.is_dir() and str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))


def main() -> int:
    _bootstrap()
    from evaluate_fs import main as benchmark_main

    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
