"""Reproduce only the independent fixture reduction mismatch, with synthetic CSVs."""

import json
from pathlib import Path
import numpy as np
import torch
import obs_review_guard as guard
from sic_cu.data.experiment import radial_observations

root = Path(__file__).resolve().parent
paths = sorted((root / "pytest临时_独立IO").glob("纯合成原始CSV与Parquet*/原始合成CSV/*.csv"))
records = []
for path in paths:
    native = radial_observations(path)
    raw = native["reliability_weight_raw"].to_numpy()
    np_sum = np.sum(raw, dtype=np.float32)
    pl_sum = np.float32(native["reliability_weight_raw"].sum())
    np_weights = (raw / np_sum).astype(np.float32)
    pl_weights = (raw / pl_sum).astype(np.float32)
    actual = native["frame_weight"].to_numpy()
    records.append({"合成CSV": str(path), "NumPy原生Float32和": float(np_sum),
        "Polars原生Float32和": float(pl_sum), "两归一最大绝对差": float(abs(np_weights - actual).max()),
        "Polars归一逐点bitwise相同": bool(np.array_equal(pl_weights, actual)),
        "Polars同dtype独立除法最大绝对差": float(abs(pl_weights - actual).max()),
        "Float64理想归一最大绝对差": float(abs(raw.astype(float) / raw.astype(float).sum() - actual).max()),
        "原帧Float64权重和对1偏差": float(abs(actual.astype(float).sum() - 1))})
assert records and all(row["Float64理想归一最大绝对差"] < 3e-8 and row["原帧Float64权重和对1偏差"] < 1e-6 for row in records)
assert any(row["两归一最大绝对差"] > 5e-9 for row in records)
assert not torch.cuda.is_initialized()
payload = {"实际退出码": 0, "根因": "跨实现Float32归约与归一舍入使夹具NumPy对照偏离原Polars已存Float32值；同Polars的独立NumPy除法也不保证bitwise。所有原值均在Float64理想归一3e-8及帧和1e-6误差界内，不是Howard候选或原权重篡改",
           "记录": records, "CPU守卫计数": guard.COUNTS}
with (root / "Float32求和根因实际证据.json").open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
