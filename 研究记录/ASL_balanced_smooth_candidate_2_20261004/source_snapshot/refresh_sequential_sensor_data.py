"""Refresh the corrected 169 W test sensor while preserving other data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import polars as pl

from train_sequential_deeponet import ROOT, digest, write_json

sys.path.insert(0, str(ROOT / "src"))
from sic_cu.data.sensors import ring_average_raw
from sic_cu.eval.protocol_checks import canonical_json_sha256

SOURCE = Path("data/test_Data/colddata/colddata169W.csv")
PROCESSED = Path("data/processed/test_sensor_ring_raw.parquet")


def refresh(root, destination):
    source, processed = root/SOURCE, root/PROCESSED
    manifest_path = root/"data/processed/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in manifest["sensors"] if e["source"]==str(SOURCE)]
    if len(entries)!=1 or entries[0]["split"]!="test" or entries[0]["source_dataset"]!="test":
        raise ValueError("The corrected file must be a unique test-only manifest entry.")
    before = pl.read_parquet(processed)
    selected = (pl.col("power_w")==169.) & (pl.col("sensor_type")=="cold")
    old = before.filter(selected).sort("time_raw")
    updated = ring_average_raw(source,"cold").with_columns(
        pl.lit("test").alias("split"),pl.lit("test").alias("source_dataset"))
    updated = updated.cast(before.schema).select(before.columns).sort("time_raw")
    if not np.array_equal(old["time_raw"].to_numpy(),updated["time_raw"].to_numpy()):
        raise ValueError("Corrected sensor timestamps changed; revise the experiment explicitly.")
    if not np.isfinite(updated["value_mean_raw"].to_numpy()).all():
        raise ValueError("Corrected sensor temperatures must be finite.")
    after = pl.concat((before.filter(~selected),updated))
    if not before.filter(~selected).equals(after.filter(~selected)):
        raise AssertionError("Unrelated sensor observations changed.")
    destination.mkdir(parents=True,exist_ok=True)
    record_path = destination/"data_revision.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["source_sha256"]!=digest(source) or record["processed_after_sha256"]!=digest(processed):
            raise ValueError("An existing revision record no longer matches the data.")
        return record
    shutil.copy2(processed,destination/"previous_test_sensor_ring_raw.parquet")
    shutil.copy2(manifest_path,destination/"previous_manifest.json")
    unchanged = {str(p.relative_to(root)):digest(p) for p in (root/"data/processed").rglob("*.parquet") if p!=processed}
    record = dict(revision="cold169_sensor25_20261004",source=str(SOURCE),
        source_sha256=digest(source),previous_source_sha256=entries[0]["source_sha256"],
        processed_path=str(PROCESSED),processed_before_sha256=digest(processed),
        samples=updated.height,split="test",training_allowed=False,model_selection_allowed=False,
        first_time_s=float(updated["time_raw"][0]),first_measurement_c=float(updated["value_mean_raw"][0]),
        last_measurement_c=float(updated["value_mean_raw"][-1]),
        maximum_measurement_change_c=float(np.max(np.abs(old["value_mean_raw"].to_numpy()-updated["value_mean_raw"].to_numpy()))),
        unchanged_processed_sha256=unchanged,raw_measurements_shifted=False)
    temporary = processed.with_suffix(".parquet.tmp")
    after.write_parquet(temporary,compression="zstd",statistics=True)
    temporary.replace(processed)
    entries[0].update(source_sha256=digest(source),samples=updated.height,
                      angular_rows_collapsed=int(updated["n_angular_samples"][0]))
    manifest.pop("processed_manifest_sha256",None)
    manifest["processed_manifest_sha256"] = canonical_json_sha256(manifest)
    write_json(manifest_path,manifest)
    record["processed_after_sha256"] = digest(processed)
    record["manifest_after_sha256"] = digest(manifest_path)
    if any(digest(root/name)!=value for name,value in unchanged.items()):
        raise AssertionError("Unrelated processed files changed.")
    write_json(record_path,record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    args = parser.parse_args()
    record = refresh(ROOT,args.destination.resolve())
    print(json.dumps({k:v for k,v in record.items() if k!="unchanged_processed_sha256"},indent=2))


if __name__=="__main__":
    main()
