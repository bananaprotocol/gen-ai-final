#!/usr/bin/env python3
"""Extended evaluation: RMSE by lead time, stability analysis, power spectra.

Usage:
    python scripts/evaluate_rollout.py \
        --predictions_dir results/predictions \
        --validation_dir data_5deg/val \
        --output_dir results/evaluation
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def compute_area_weights(nlat: int) -> np.ndarray:
    lats = np.linspace(-90 + 180 / (2 * nlat), 90 - 180 / (2 * nlat), nlat)
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.mean()


def compute_rmse(predictions, targets, var_name, area_weights=None):
    diff = predictions[var_name] - targets[var_name]
    if area_weights is not None:
        w = xr.DataArray(area_weights, dims=["grid_yt"])
        mse = (diff ** 2 * w).mean()
    else:
        mse = (diff ** 2).mean()
    return float(np.sqrt(mse))


def compute_global_mean_timeseries(ds, var_name, area_weights):
    w = xr.DataArray(area_weights, dims=["grid_yt"])
    weighted = ds[var_name] * w
    return weighted.mean(dim=["grid_yt", "grid_xt"]).values


def compute_power_spectrum(field):
    fft = np.fft.rfft(field, axis=-1)
    power = np.abs(fft) ** 2
    return power.mean(axis=-2)


def plot_stability_analysis(pred_ts, target_ts, var_name, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    days = np.arange(len(pred_ts)) * 6 / 24

    axes[0].plot(days, target_ts, label="Target", alpha=0.7)
    axes[0].plot(days, pred_ts, label="Prediction", alpha=0.7)
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel(f"Global mean {var_name}")
    axes[0].legend()

    drift = pred_ts - target_ts
    axes[1].plot(days, drift, color="red", alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel(f"Drift ({var_name})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"stability_{var_name}.png"), dpi=150)
    plt.close()


def plot_rmse_by_leadtime(rmse_dict, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for var_name, rmse_values in rmse_dict.items():
        lead_days = np.arange(1, len(rmse_values) + 1) * 6 / 24
        ax.plot(lead_days, rmse_values, label=var_name)

    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("RMSE")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_by_leadtime.png"), dpi=150)
    plt.close()


def plot_power_spectra(pred_spectrum, target_spectrum, var_name, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    wn = np.arange(len(target_spectrum))
    ax.loglog(wn[1:], target_spectrum[1:], label="Target", alpha=0.7)
    ax.loglog(wn[1:], pred_spectrum[1:], label="Prediction", alpha=0.7)

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"Power spectrum: {var_name}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"power_spectrum_{var_name}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--validation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stats_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_vars = [
        "air_temperature_7",
        "air_temperature_0",
        "PRESsfc",
        "specific_total_water_7",
    ]

    print(f"Predictions: {args.predictions_dir}")
    print(f"Validation:  {args.validation_dir}")
    print(f"Variables:   {eval_vars}")
    print("Run after inference completes.")


if __name__ == "__main__":
    main()
