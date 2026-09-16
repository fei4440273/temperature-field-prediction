"""Review-process guard: synthetic data IO only and no CUDA initialization."""

import json
import os
import sys
from pathlib import Path

import polars as pl
import torch


REVIEW_ROOT = Path(__file__).resolve().parent
COUNTS = {"outside_data_io_denied": 0, "cuda_initialization_attempts": 0,
          "synthetic_torch_loads": 0, "synthetic_parquet_reads": 0}


def _allowed_data(value):
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).resolve()
        if not path.is_relative_to(REVIEW_ROOT):
            COUNTS["outside_data_io_denied"] += 1
            raise RuntimeError("Independent CPU review forbids non-fixture model/data IO: " + str(path))


def _audit(event, args):
    if event == "open" and isinstance(args[0], (str, os.PathLike)):
        if Path(args[0]).suffix.lower() in {".pt", ".parq", ".parquet"}:
            _allowed_data(args[0])


def _no_cuda(*args, **kwargs):
    COUNTS["cuda_initialization_attempts"] += 1
    raise RuntimeError("Independent CPU review forbids all CUDA initialization")


if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before review guard")
torch.cuda._lazy_init = _no_cuda
torch.cuda.init = _no_cuda
if hasattr(torch._C, "_cuda_init"):
    torch._C._cuda_init = _no_cuda
sys.addaudithook(_audit)
_torch_load = torch.load
_parquet_read = pl.read_parquet


def _guarded_torch_load(value, *args, **kwargs):
    _allowed_data(value)
    COUNTS["synthetic_torch_loads"] += 1
    return _torch_load(value, *args, **kwargs)


def _guarded_parquet_read(value, *args, **kwargs):
    _allowed_data(value)
    COUNTS["synthetic_parquet_reads"] += 1
    return _parquet_read(value, *args, **kwargs)


torch.load = _guarded_torch_load
pl.read_parquet = _guarded_parquet_read


def pytest_sessionfinish(session, exitstatus):
    initialized = torch.cuda.is_initialized()
    print("INDEPENDENT_CPU_SAFETY " + json.dumps({**COUNTS,
          "cuda_initialized": initialized, "pytest_exitstatus": int(exitstatus)}, sort_keys=True))
    if initialized or COUNTS["outside_data_io_denied"] or COUNTS["cuda_initialization_attempts"]:
        raise RuntimeError("Independent review safety boundary was attempted")
