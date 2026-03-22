"""Autoregressive rollout with cold sampling.

Usage:
    python -m src.inference --forecaster_ckpt checkpoints/forecaster/last.pt \
        --interpolator_ckpt checkpoints/interpolator/last.pt \
        --data_dir data_5deg --num_steps 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from src.data import PROGNOSTIC_NAMES, FORCING_NAMES, load_variables
from src.dyffusion import Interpolator, DYffusion
from src.normalization import load_normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecaster_ckpt", type=str, required=True)
    parser.add_argument("--interpolator_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/predictions")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    prog_normalizer = load_normalizer(
        data_dir / "centering.nc", data_dir / "scaling.nc", PROGNOSTIC_NAMES
    ).to(device)
    forcing_normalizer = load_normalizer(
        data_dir / "centering.nc", data_dir / "scaling.nc", FORCING_NAMES
    ).to(device)

    interpolator = Interpolator(horizon=args.horizon)
    interpolator.load_state_dict(torch.load(args.interpolator_ckpt, map_location=device))
    interpolator.to(device)
    interpolator.eval()

    model = DYffusion(interpolator=interpolator, horizon=args.horizon)
    model.load_state_dict(torch.load(args.forecaster_ckpt, map_location=device))
    model.to(device)
    model.eval()

    val_dir = data_dir / "val"
    nc_files = sorted(val_dir.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files in {val_dir}")

    ds = xr.open_dataset(nc_files[0])

    x_current = torch.from_numpy(load_variables(ds, PROGNOSTIC_NAMES)[0:1]).float().to(device)
    x_current = prog_normalizer.normalize(x_current)

    # use first-timestep forcing throughout (held constant)
    forcing = torch.from_numpy(load_variables(ds, FORCING_NAMES)[0:1]).float().to(device)
    forcing = forcing_normalizer.normalize(forcing)
    ds.close()

    predictions = []
    print(f"Running {args.num_steps} autoregressive steps...")

    with torch.inference_mode():
        for step in tqdm(range(args.num_steps)):
            x_next = model.cold_sample(x_current, forcing)
            x_next_denorm = prog_normalizer.denormalize(x_next)
            predictions.append(x_next_denorm.cpu().numpy())
            x_current = x_next

    predictions = np.concatenate(predictions, axis=0)

    coords = {
        "time": np.arange(predictions.shape[0]),
        "grid_yt": np.arange(predictions.shape[2]),
        "grid_xt": np.arange(predictions.shape[3]),
    }
    data_vars = {}
    for i, name in enumerate(PROGNOSTIC_NAMES):
        data_vars[name] = (["time", "grid_yt", "grid_xt"], predictions[:, i])

    out_ds = xr.Dataset(data_vars, coords=coords)
    out_path = output_dir / "rollout_predictions.nc"
    out_ds.to_netcdf(out_path)
    print(f"Saved {predictions.shape} to {out_path}")


if __name__ == "__main__":
    main()
