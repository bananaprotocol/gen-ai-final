from pathlib import Path

import torch
import xarray as xr


class StandardNormalizer:
    """Per-variable (x - mean) / std normalization with (C, 1, 1) broadcasting."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def to(self, device: torch.device) -> "StandardNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def load_normalizer(
    centering_path: str | Path,
    scaling_path: str | Path,
    variable_names: list[str],
) -> StandardNormalizer:
    """Load from centering.nc / scaling.nc. Handles level-indexed variables
    like 'air_temperature_3' (stored as air_temperature with level=3)."""
    mean_ds = xr.open_dataset(centering_path)
    std_ds = xr.open_dataset(scaling_path)

    means, stds = [], []
    for name in variable_names:
        if name in mean_ds:
            means.append(float(mean_ds[name].values))
            stds.append(float(std_ds[name].values))
        else:
            parts = name.rsplit("_", 1)
            var_name, level = parts[0], int(parts[1])
            means.append(float(mean_ds[var_name].sel(level=level).values))
            stds.append(float(std_ds[var_name].sel(level=level).values))

    return StandardNormalizer(
        mean=torch.tensor(means, dtype=torch.float32),
        std=torch.tensor(stds, dtype=torch.float32),
    )
