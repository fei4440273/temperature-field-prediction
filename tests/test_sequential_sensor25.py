"""Initial-temperature and export contracts for the corrected sensor revision."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from test_sequential_deeponet import config, physical, curves
from joint_temperature_core import Geometry, Table, physics_loss
from sequential_deeponet_core import METHODS, SequentialDeepONet
from sequential_deeponet_data import HistoryProvider
from train_sequential_deeponet import table_predict


@pytest.mark.parametrize("method", METHODS)
def test_sensor25_exact_anchor_and_fast_inference(method):
    cfg = dict(config(), experiment_copper_initial_c=25.)
    g = Geometry()
    model = SequentialDeepONet(method, g, physical(), cfg)
    provider = HistoryProvider(curves(), g, cfg)
    model.set_history_providers(low=provider, high=provider)
    x = torch.tensor([[.028, g.bottom_cu, 0., 169., 0.],
                      [.0415, g.bottom_cu, 0., 634., 0.],
                      [.0, 0., 0., 169., 1.],
                      [.028, g.bottom_cu, 3., 169., 0.]])
    for fidelity in ("low", "high"):
        prediction = model(x, fidelity)
        expected = [298.15, 298.15, 295.15] if fidelity == "high" else [295.15]*3
        torch.testing.assert_close(prediction[:3, 0], torch.tensor(expected), rtol=0, atol=0)
        table = Table(x.numpy(), np.zeros((len(x), 1), dtype=np.float32),
                      np.ones(len(x), dtype=np.float32), np.zeros(len(x), dtype=np.int32))
        np.testing.assert_allclose(table_predict(model, table, provider, fidelity),
                                   prediction.detach().numpy()[:, 0], atol=5e-5, rtol=0)
    parts = physics_loss(model, 4, torch.Generator().manual_seed(7))
    assert float(parts["初始条件"].detach()) == 0.
    legacy = SequentialDeepONet(method, g, physical(), config())
    assert model.state_dict().keys() == legacy.state_dict().keys()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in legacy.parameters())


def test_history_uses_declared_initial_temperature_and_preserves_causality():
    times = np.arange(5, dtype=float)
    values = np.column_stack((298.15+times, 298.15+.5*times))
    cfg = config()
    provider = HistoryProvider({169.: (times, values)}, Geometry(), cfg,
                               initial_temperature_k=298.15)
    long, local = provider.build([169.], [0.])
    np.testing.assert_array_equal(long[..., 2:4], 0.)
    np.testing.assert_array_equal(local[..., 2:4], 0.)
    changed = values.copy()
    changed[3:] += 999
    other = HistoryProvider({169.: (times, changed)}, Geometry(), cfg,
                            initial_temperature_k=298.15)
    for a, b in zip(provider.build([169.], [3.]), other.build([169.], [3.])):
        np.testing.assert_array_equal(a, b)
    simulation = HistoryProvider(curves(), Geometry(), cfg)
    np.testing.assert_array_equal(simulation.build([100.], [0.])[0][..., 2:4], 0.)


def test_figure_export_is_png_only(tmp_path):
    import matplotlib.pyplot as plt
    from plot_sequential_deeponet import save
    fig, ax = plt.subplots()
    ax.plot([0, 1], [25, 26])
    (tmp_path/"initial_temperature.pdf").write_bytes(b"old generated figure")
    save(fig, tmp_path, "initial_temperature")
    assert {p.suffix for p in tmp_path.iterdir()} == {".png"}


def test_refresh_replaces_only_corrected_test_sensor_and_is_repeatable(tmp_path):
    import json
    import polars as pl
    from refresh_sequential_sensor_data import refresh, SOURCE, PROCESSED
    from sic_cu.data.sensors import ring_average_raw
    from train_sequential_deeponet import digest
    source = tmp_path/SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("X,Y,t=1,t=2\n41.5,0,25.2,25.4\n0,41.5,25.4,25.6\n",encoding="utf-8")
    corrected = ring_average_raw(source,"cold").with_columns(
        pl.lit("test").alias("split"),pl.lit("test").alias("source_dataset"))
    old = corrected.with_columns((pl.col("value_mean_raw")+2).alias("value_mean_raw"))
    unrelated = old.with_columns(pl.lit("hot").alias("sensor_type"))
    processed = tmp_path/PROCESSED
    processed.parent.mkdir(parents=True)
    pl.concat((old,unrelated)).write_parquet(processed)
    training = processed.parent/"sensor_ring_raw.parquet"
    unrelated.write_parquet(training)
    manifest = dict(sensors=[dict(source=str(SOURCE),source_sha256="old",split="test",source_dataset="test")])
    (processed.parent/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    training_hash = digest(training)
    record = refresh(tmp_path,tmp_path/"revision")
    after = pl.read_parquet(processed)
    assert after.filter(pl.col("sensor_type")=="cold").equals(corrected)
    assert after.filter(pl.col("sensor_type")=="hot").equals(unrelated)
    assert digest(training)==training_hash
    assert record["maximum_measurement_change_c"]==pytest.approx(2.)
    assert record==refresh(tmp_path,tmp_path/"revision")


def test_evaluation_archives_the_corrected_source_and_rejects_stale_cache(tmp_path):
    import json
    from evaluate_sequential_deeponet import archive_sensor_revision
    from train_sequential_deeponet import digest
    source = tmp_path/"data/test_Data/colddata/colddata169W.csv"
    source.parent.mkdir(parents=True)
    source.write_text("corrected observations",encoding="utf-8")
    processed = tmp_path/"data/processed/test_sensor_ring_raw.parquet"
    processed.parent.mkdir(parents=True)
    processed.write_bytes(b"processed sensor fixture")
    output = tmp_path/"results"
    revision_dir = output/"data_revision"
    revision_dir.mkdir(parents=True)
    source_name = str(source.relative_to(tmp_path))
    record = dict(source=source_name,source_sha256=digest(source),
                  processed_path=str(processed.relative_to(tmp_path)),processed_after_sha256=digest(processed))
    (revision_dir/"data_revision.json").write_text(json.dumps(record),encoding="utf-8")
    provenance = dict(raw_source_hashes={source_name:digest(source)},
                      data_hashes={record["processed_path"]:digest(processed)})
    (output/"provenance.json").write_text(json.dumps(provenance),encoding="utf-8")
    archive_sensor_revision(output,root=tmp_path)
    assert digest(revision_dir/"corrected_colddata169W.csv")==digest(source)
    processed.write_bytes(b"stale cache")
    with pytest.raises(ValueError,match="changed"):
        archive_sensor_revision(output,root=tmp_path)
