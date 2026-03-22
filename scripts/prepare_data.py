#!/usr/bin/env python3
"""Downsample FV3GFS from 1-deg to 5.625-deg, split train/val, compute stats.

Usage:
    python scripts/prepare_data.py --input_dir data --output_dir data_5deg
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
import xarray as xr


TARGET_NLAT = 32
TARGET_NLON = 64


def downsample_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Bilinear interpolation from (180, 360) to (32, 64)."""
    out_vars = {}
    for name, var in ds.data_vars.items():
        arr = var.values
        if "grid_yt" in var.dims and "grid_xt" in var.dims:
            spatial_axes = (var.dims.index("grid_yt"), var.dims.index("grid_xt"))
            factors = [1.0] * arr.ndim
            factors[spatial_axes[0]] = TARGET_NLAT / arr.shape[spatial_axes[0]]
            factors[spatial_axes[1]] = TARGET_NLON / arr.shape[spatial_axes[1]]
            arr = zoom(arr, factors, order=1)
        new_dims = list(var.dims)
        out_vars[name] = (new_dims, arr.astype(np.float32))
    coords = {}
    if "time" in ds.coords:
        coords["time"] = ds.coords["time"]
    return xr.Dataset(out_vars, coords=coords)


def compute_stats(nc_files: list[Path]) -> tuple[xr.Dataset, xr.Dataset]:
    ds = xr.open_mfdataset(nc_files, combine="nested", concat_dim="time")
    all_dims = [d for d in ds.dims if d in ("time", "grid_yt", "grid_xt")]
    centering = ds.mean(dim=all_dims)
    scaling = ds.std(dim=all_dims)
    for var in scaling.data_vars:
        if float(scaling[var].values) == 0.0:
            scaling[var] = 1.0
    ds.close()
    return centering, scaling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data_5deg")
    parser.add_argument("--val_months", nargs="+", default=["11", "12"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(input_dir.glob("*.nc"))
    nc_files = [f for f in nc_files if f.name != "ace_ckpt.tar"]
    print(f"Found {len(nc_files)} files in {input_dir}")

    val_suffixes = [m.zfill(2) for m in args.val_months]

    for nc_path in nc_files:
        month = nc_path.stem[4:6]
        dest_dir = val_dir if month in val_suffixes else train_dir
        out_path = dest_dir / nc_path.name

        if out_path.exists():
            print(f"  skip {out_path.name}")
            continue

        print(f"  {nc_path.name} -> {dest_dir.name}/")
        ds = xr.open_dataset(nc_path)
        ds_out = downsample_dataset(ds)
        ds_out.to_netcdf(out_path)
        ds.close()

    train_files = sorted(train_dir.glob("*.nc"))
    if train_files:
        print(f"Computing stats from {len(train_files)} training files...")
        centering, scaling = compute_stats(train_files)
        centering.to_netcdf(output_dir / "centering.nc")
        scaling.to_netcdf(output_dir / "scaling.nc")
        print(f"Saved centering.nc, scaling.nc to {output_dir}")


if __name__ == "__main__":
    main()
