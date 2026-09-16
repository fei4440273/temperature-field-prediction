#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sic_cu.eval.release import freeze_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a protocol-compatible release before any test evaluation"
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--model-config", action="append", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    result = freeze_release(
        args.release_id,
        args.checkpoint,
        args.model_config,
        args.command,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
