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
        "Polars归一逐点bitwise相同": bool(np.array_equal(pl_weights, actual))})
assert records and all(row["Polars归一逐点bitwise相同"] for row in records)
assert any(row["两归一最大绝对差"] > 5e-9 for row in records)
assert not torch.cuda.is_initialized()
payload = {"实际退出码": 0, "根因": "不同Float32求和顺序导致夹具NumPy归一对照偏离原Polars归一；不是Howard候选或原权重篡改",
           "记录": records, "CPU守卫计数": guard.COUNTS}
with (root / "Float32求和根因实际证据.json").open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
    stream.write("\n")
print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
