#!/usr/bin/env python3
from __future__ import annotations

def main() -> None:
    raise RuntimeError(
        "Post-hoc deployment constraint injection is disabled by the V2 protocol. "
        "Train constraints explicitly, freeze the selected checkpoint with "
        "scripts/06_freeze_release.py, and evaluate that unchanged artifact."
    )


if __name__ == "__main__":
    main()
