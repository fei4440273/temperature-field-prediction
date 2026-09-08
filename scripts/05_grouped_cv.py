#!/usr/bin/env python3
"""Retired entry point retained to fail clearly for old commands."""


def main() -> None:
    raise SystemExit(
        "Grouped cross-validation is disabled. Use the fixed train/validation/test protocol."
    )


if __name__ == "__main__":
    main()
