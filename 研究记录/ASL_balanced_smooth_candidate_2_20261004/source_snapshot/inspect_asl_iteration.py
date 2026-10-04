"""Compare ASL candidates on training/validation data without opening test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from joint_temperature_core import Geometry,fixed_splits
from sequential_deeponet_core import load_checkpoint
from sequential_deeponet_data import experiment_history,metrics
from train_sequential_deeponet import ROOT,table_predict,digest,write_json,validate


def inspect(output,device="cuda"):
    cfg = json.loads((output/"config.json").read_text(encoding="utf-8"))
    if cfg["experiment"]["methods"]!=["asl"]:
        raise ValueError("ASL-only candidate required.")
    geometry,splits = Geometry.from_project(ROOT),fixed_splits(ROOT)
    datasets = {split:experiment_history(ROOT,split,splits,geometry,cfg["model"])
                for split in ("train","validation")}
    result = dict(config_sha256=digest(output/"config.json"),selection_data=["train","validation"],
                  checkpoints={})
    for filename in ("best.pt","epoch_1000.pt"):
        path = output/"asl"/filename
        model,state = load_checkpoint(path,device)
        model.eval()
        measured = {}
        for split,(tables,provider) in datasets.items():
            measured[split] = {name:metrics(table.y,table_predict(model,table,provider,"high"))
                               for name,table in tables.items()}
        tables,provider = datasets["validation"]
        score,_ = validate(model,tables,provider,cfg["validation_selection"])
        result["checkpoints"][filename] = dict(epoch=state["epoch"],sha256=digest(path),
                                             validation_score=score,metrics=measured)
        print(json.dumps(dict(checkpoint=filename,epoch=state["epoch"],validation_score=score,
              rmse_k={split:{name:m["rmse_k"] for name,m in rows.items()}
                      for split,rows in measured.items()}),ensure_ascii=False),flush=True)
    write_json(output/"candidate_validation.json",result)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    inspect(args.output.resolve(),args.device)
