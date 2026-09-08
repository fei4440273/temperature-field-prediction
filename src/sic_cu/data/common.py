from __future__ import annotations

import hashlib
import re
from pathlib import Path


POWER_RE = re.compile(r"(?:HotData-|ColdData-)?(?P<power>\d+(?:\.\d+)?)W")
IR_RE = re.compile(r"(?P<power>\d+(?:\.\d+)?)W-(?P<time>\d+(?:\.\d+)?)s")


def parse_power(path: str | Path) -> float:
    match = POWER_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse power from {path}")
    return float(match.group("power"))


def parse_ir_power_time(path: str | Path) -> tuple[float, float]:
    match = IR_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse IR power/time from {path}")
    return float(match.group("power")), float(match.group("time"))


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_to_gib(value: int) -> float:
    return value / 1024**3

