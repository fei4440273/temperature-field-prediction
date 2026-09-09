from __future__ import annotations

from pathlib import Path

import polars as pl

from sic_cu.config import PROJECT_ROOT


def processed_ir_path(split: str | None = None) -> Path:
    """Resolve an IR file without opening test labels for non-test calls."""
    if split not in {None, "train", "validation", "test"}:
        raise ValueError("split must be train, validation, test, or None")
    filename = "test_ir_radial.parquet" if split == "test" else "experiment_ir_radial.parquet"
    return PROJECT_ROOT / "data/processed" / filename


def load_processed_ir_observations(split: str | None = None) -> pl.DataFrame:
    frame = pl.read_parquet(processed_ir_path(split))
    return frame if split is None else frame.filter(pl.col("split") == split)
