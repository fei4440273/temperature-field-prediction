from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from sic_cu.prediction import Prediction, metadata_json


def write_legacy_vtk_points(
    path: str | Path,
    xyz_m: np.ndarray,
    temperature_k: np.ndarray,
    material_ids: np.ndarray,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(xyz_m)
    temperature_c = np.asarray(temperature_k) - 273.15
    materials = np.asarray(material_ids)
    if points.shape != (len(temperature_c), 3) or materials.shape != temperature_c.shape:
        raise ValueError("VTK arrays have incompatible shapes")
    with destination.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("SiC-Cu axisymmetric temperature prediction\nASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n")
        np.savetxt(handle, points, fmt="%.8g %.8g %.8g")
        handle.write(f"VERTICES {len(points)} {2 * len(points)}\n")
        for index in range(len(points)):
            handle.write(f"1 {index}\n")
        handle.write(f"POINT_DATA {len(points)}\n")
        handle.write("SCALARS temperature_c float 1\nLOOKUP_TABLE default\n")
        np.savetxt(handle, temperature_c, fmt="%.7g")
        handle.write("SCALARS material_id int 1\nLOOKUP_TABLE default\n")
        np.savetxt(handle, materials, fmt="%d")


def _nearest_bottom_node(prediction: Prediction, radius_m: float) -> int:
    coordinates = prediction.coordinates_rz_m
    copper = prediction.material_ids == 0
    bottom = float(coordinates[copper, 1].min())
    distance = np.hypot(
        coordinates[:, 0] - radius_m,
        20.0 * (coordinates[:, 1] - bottom),
    )
    distance[~copper] = np.inf
    return int(np.argmin(distance))


def export_prediction(
    prediction: Prediction,
    output_directory: str | Path,
    theta_resolution: int = 72,
    vtk_times_s: tuple[float, ...] = (0.0, 50.0, 100.0, 150.0, 200.0),
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "field_rzt.npz",
        power_w=np.asarray(prediction.power_w),
        times_s=prediction.times_s,
        coordinates_rz_m=prediction.coordinates_rz_m,
        material_ids=prediction.material_ids,
        mean_temperature_k=prediction.mean_temperature_k,
        q05_temperature_k=prediction.q05_temperature_k,
        q95_temperature_k=prediction.q95_temperature_k,
        max_temperature_k=prediction.max_temperature_k,
    )
    (output / "metadata.json").write_text(metadata_json(prediction), encoding="utf-8")
    hot_index = _nearest_bottom_node(prediction, 0.028)
    cold_index = _nearest_bottom_node(prediction, 0.0415)
    with (output / "hot_cold.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "hot_temperature_c", "cold_temperature_c"))
        for index, time_s in enumerate(prediction.times_s):
            writer.writerow(
                (
                    f"{time_s:.7g}",
                    f"{prediction.mean_temperature_c[index, hot_index]:.7g}",
                    f"{prediction.mean_temperature_c[index, cold_index]:.7g}",
                )
            )
    vtk_paths: list[str] = []
    for requested in vtk_times_s:
        index = int(np.argmin(np.abs(prediction.times_s - requested)))
        if not np.isclose(prediction.times_s[index], requested, atol=1e-5):
            continue
        xyz, temperature, materials = prediction.rotate(index, theta_resolution)
        path = output / "vtk" / f"field_{prediction.times_s[index]:g}s.vtk"
        write_legacy_vtk_points(path, xyz, temperature, materials)
        vtk_paths.append(str(path))
    figure_paths = _export_figures(prediction, output / "figures")
    return {
        "field": str(output / "field_rzt.npz"),
        "metadata": str(output / "metadata.json"),
        "hot_cold": str(output / "hot_cold.csv"),
        "vtk": vtk_paths,
        "figures": figure_paths,
    }


def _export_figures(prediction: Prediction, directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(prediction.times_s, prediction.max_temperature_k - 273.15)
    axis.set(xlabel="Time (s)", ylabel="Maximum temperature (degC)")
    axis.grid(alpha=0.25)
    path = directory / "maximum_temperature.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    coordinates = prediction.coordinates_rz_m
    triangulation = mtri.Triangulation(
        coordinates[:, 0] * 1000.0,
        coordinates[:, 1] * 1000.0,
    )
    for requested in (0.0, 100.0, 200.0):
        index = int(np.argmin(np.abs(prediction.times_s - requested)))
        if not np.isclose(prediction.times_s[index], requested, atol=1e-5):
            continue
        fig, axis = plt.subplots(figsize=(8, 3.5))
        contour = axis.tricontourf(
            triangulation,
            prediction.mean_temperature_c[index],
            levels=30,
            cmap="inferno",
        )
        fig.colorbar(contour, ax=axis, label="Temperature (degC)")
        axis.set(
            xlabel="r (mm)",
            ylabel="z (mm)",
            title=f"{prediction.power_w:g} W, {requested:g} s",
        )
        axis.set_aspect("equal")
        path = directory / f"section_{requested:g}s.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths
