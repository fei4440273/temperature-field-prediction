#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from sic_cu.eval.development_v4 import export_development_v4_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export V4 M0/M1 diagnostics from existing checkpoints only."
    )
    parser.add_argument("--config", default="configs/optimization_v4.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    destination = export_development_v4_diagnostics(
        config_path=args.config,
        output_root=args.output_root,
        effective_cli=["python", "scripts/07_export_development_diagnostics.py", *sys.argv[1:]],
    )
    print(destination)


if __name__ == "__main__":
    main()
