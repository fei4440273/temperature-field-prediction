"""Raw sensor-curve diagnostics; these metrics never alter displayed predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sequential_temperature_objective import sensor_tail_metrics
from train_sequential_deeponet import digest
from evaluate_asl_iteration import as_table


def slope_waviness(times,prediction):
    times = np.asarray(times,dtype=float)
    prediction = np.asarray(prediction,dtype=float)
    step = float(np.median(np.diff(times)))
    width = max(3,int(round(5./step))|1)
    half = width//2
    rates = np.diff(prediction)/np.diff(times)
    trend = np.convolve(rates,np.ones(width)/width,mode="valid")
    central = rates[half:len(rates)-half]
    eligible = times[half:len(rates)-half]>=5.
    return float(np.sqrt(np.mean((central[eligible]-trend[eligible])**2)))


def audit(output,previous,window_s=20.):
    rows = []
    source_hashes = {}
    for tag in ("hot","cold"):
        filename = f"{tag}_predictions.npz"
        with np.load(output/"evaluation"/filename) as current,np.load(previous/"evaluation"/filename) as old:
            table = as_table(current)
            np.testing.assert_array_equal(current["x"],old["x"])
            np.testing.assert_array_equal(current["y"],old["y"])
            measured = sensor_tail_metrics(table,current["asl"],window_s)["curves"]
            old_measured = sensor_tail_metrics(table,old["asl"],window_s)["curves"]
        source_hashes[filename] = dict(current=digest(output/"evaluation"/filename),
                                     previous=digest(previous/"evaluation"/filename))
        for row,before in zip(measured,old_measured):
            power = row["power_w"]
            dense_name = f"dense_{tag}_{power:g}W_predictions.npz"
            with np.load(output/"evaluation"/dense_name) as dense,np.load(previous/"evaluation"/dense_name) as old_dense:
                np.testing.assert_array_equal(dense["x"],old_dense["x"])
                n = int(dense["grid_count"])
                times,prediction = dense["x"][:n,2],dense["asl"][:n]
                before_wave = slope_waviness(times,old_dense["asl"][:n])
                after_wave = slope_waviness(times,prediction)
                epsilon_times = dense["x"][n:,2].reshape(-1,3)[:,1]
                epsilon_values = dense["asl"][n:].reshape(-1,3)
                incoming = abs(epsilon_values[:,1]-epsilon_values[:,0])
                late = epsilon_times>=times.max()-window_s
                row.update(sensor=tag,previous_rate_error_k_per_s=before["rate_error_k_per_s"],
                    previous_predicted_rate_k_per_s=before["predicted_rate_k_per_s"],
                    previous_tail_bias_k=before["tail_bias_k"],previous_slope_waviness_k_per_s=before_wave,
                    slope_waviness_k_per_s=after_wave,late_max_incoming_jump_k=float(incoming[late].max()),
                    rate_error_improved=abs(row["rate_error_k_per_s"])<abs(before["rate_error_k_per_s"]),
                    waviness_improved=after_wave<before_wave)
            source_hashes[dense_name] = dict(current=digest(output/"evaluation"/dense_name),
                                          previous=digest(previous/"evaluation"/dense_name))
            rows.append(row)
    return dict(curves=rows,source_sha256=source_hashes,tail_window_s=window_s,
        all_six_rate_errors_improved=all(r["rate_error_improved"] for r in rows),
        all_six_waviness_metrics_improved=all(r["waviness_improved"] for r in rows),
        tail_rate_rmse_k_per_s=float(np.sqrt(np.mean([r["rate_error_k_per_s"]**2 for r in rows]))),
        previous_tail_rate_rmse_k_per_s=float(np.sqrt(np.mean([r["previous_rate_error_k_per_s"]**2 for r in rows]))),
        no_prediction_postprocessing=True,audit_script_sha256=digest(Path(__file__)))
