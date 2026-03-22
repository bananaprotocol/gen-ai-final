"""Evaluate DYffusion rollout: RMSE vs lead time, temperature drift.

Usage:
    python -m src.evaluate --predictions results/predictions/rollout_predictions.nc \
        --validation_dir data_5deg/val
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.data import PROGNOSTIC_NAMES


def area_weights(nlat: int = 32) -> np.ndarray:
    lats = np.linspace(-90 + 180 / (2 * nlat), 90 - 180 / (2 * nlat), nlat)
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.mean()


def rmse(pred, target, weights=None):
    diff_sq = (pred - target) ** 2
    if weights is not None:
        diff_sq = diff_sq * weights[:, np.newaxis]
    return float(np.sqrt(diff_sq.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--eval_vars", nargs="+",
                        default=["air_temperature_0", "air_temperature_3", "air_temperature_7",
                                 "specific_total_water_0", "specific_total_water_7",
                                 "eastward_wind_0", "eastward_wind_7",
                                 "PRESsfc", "surface_temperature"])
    parser.add_argument("--temp_var", type=str, default="surface_temperature")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    hours_per_step = args.horizon * 6  # each DYffusion step = horizon x 6-hourly data steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_ds = xr.open_dataset(args.predictions)
    n_steps = pred_ds.sizes["time"]

    val_dir = Path(args.validation_dir)
    nc_files = sorted(val_dir.rglob("*.nc"))
    val_ds = xr.open_mfdataset(nc_files, combine="nested", concat_dim="time")
    n_val = val_ds.sizes.get("time", 0)

    nlat = pred_ds.sizes["grid_yt"]
    weights = area_weights(nlat)

    eval_vars = [v for v in args.eval_vars if v in pred_ds and v in val_ds]
    print(f"Evaluating {n_steps} DYffusion steps (horizon={args.horizon}, "
          f"{hours_per_step}h per step), {n_val} validation timesteps, vars: {eval_vars}")

    rmse_results = {}
    persist_results = {}
    for var in eval_vars:
        rmse_results[var] = []
        persist_results[var] = []
        initial_field = val_ds[var].isel(time=0).values
        for lt in range(1, n_steps + 1):
            val_idx = lt * args.horizon
            if val_idx < n_val:
                target_field = val_ds[var].isel(time=val_idx).values
                pred_field = pred_ds[var].isel(time=lt - 1).values
                rmse_results[var].append(rmse(pred_field, target_field, weights))
                persist_results[var].append(rmse(initial_field, target_field, weights))
            else:
                rmse_results[var].append(float("nan"))
                persist_results[var].append(float("nan"))

    rmse_results = {var: np.array(vals) for var, vals in rmse_results.items()}
    persist_results = {var: np.array(vals) for var, vals in persist_results.items()}

    print("\nRMSE at selected lead times (model / persistence):")
    for lt in [1, 5, 10, 20, 50]:
        if lt <= n_steps:
            lead_d = lt * hours_per_step / 24
            for var in eval_vars:
                m = rmse_results[var][lt-1]
                p = persist_results[var][lt-1]
                skill = (1 - m / p) * 100 if p > 0 else float("nan")
                print(f"  {var} @ step {lt} ({lead_d:.1f}d): {m:.4f} / {p:.4f}  (skill: {skill:+.1f}%)")

    var_groups = {}
    for var in eval_vars:
        if "temperature" in var:
            var_groups.setdefault("Temperature [K]", []).append(var)
        elif "specific_total_water" in var:
            var_groups.setdefault("Specific humidity [kg/kg]", []).append(var)
        elif "PRES" in var:
            var_groups.setdefault("Surface pressure [Pa]", []).append(var)
        elif "wind" in var:
            var_groups.setdefault("Wind [m/s]", []).append(var)
        else:
            var_groups.setdefault("Other", []).append(var)

    n_panels = len(var_groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), squeeze=False)
    lead_days = np.arange(1, n_steps + 1) * (hours_per_step / 24.0)

    for ax, (group_label, group_vars) in zip(axes[:, 0], var_groups.items()):
        for var in group_vars:
            values = rmse_results[var]
            pvalues = persist_results[var]
            valid_mask = ~np.isnan(values)
            color = ax.plot(lead_days[valid_mask], values[valid_mask], label=var)[0].get_color()
            ax.plot(lead_days[valid_mask], pvalues[valid_mask],
                    color=color, linestyle="--", alpha=0.5, label=f"{var} (persist)")
        ax.set_ylabel(f"RMSE ({group_label})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Lead time (days)")
    fig.suptitle("Area-weighted RMSE vs lead time", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_vs_leadtime.png", dpi=150, bbox_inches="tight")
    plt.close()

    temp_var = args.temp_var
    if temp_var in pred_ds and temp_var in val_ds:
        w = xr.DataArray(weights, dims=["grid_yt"])
        pred_means = [float((pred_ds[temp_var].isel(time=t) * w).mean()) for t in range(n_steps)]
        target_means = []
        for t in range(n_steps):
            val_idx = (t + 1) * args.horizon
            if val_idx < n_val:
                target_means.append(float((val_ds[temp_var].isel(time=val_idx) * w).mean()))
            else:
                break

        days = np.arange(len(pred_means)) * (hours_per_step / 24.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(days, pred_means, label="Prediction", alpha=0.8)
        if target_means:
            ax1.plot(days[:len(target_means)], target_means, label="Target", alpha=0.8)
        ax1.set_xlabel("Days")
        ax1.set_ylabel(f"Global mean {temp_var}")
        ax1.legend()

        if target_means:
            n = min(len(pred_means), len(target_means))
            drift = np.array(pred_means[:n]) - np.array(target_means[:n])
            ax2.plot(days[:n], drift, color="red")
            ax2.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax2.set_xlabel("Days")
            ax2.set_ylabel("Drift")

        plt.tight_layout()
        plt.savefig(output_dir / "temperature_drift.png", dpi=150)
        plt.close()

    pred_ds.close()
    val_ds.close()
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
