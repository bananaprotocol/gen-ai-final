"""Climate dataset for downsampled FV3GFS with sliding-window sampling."""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from src.normalization import StandardNormalizer, load_normalizer

PROGNOSTIC_NAMES = [
    "PRESsfc",
    "surface_temperature",
    *[f"air_temperature_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"eastward_wind_{i}" for i in range(8)],
    *[f"northward_wind_{i}" for i in range(8)],
]

FORCING_NAMES = ["DSWRFtoa", "HGTsfc"]

ALL_NAMES = PROGNOSTIC_NAMES + FORCING_NAMES


def load_variables(ds: xr.Dataset, var_names: list[str]) -> np.ndarray:
    arrays = []
    n_times = None
    for name in var_names:
        arr = ds[name].values
        if arr.ndim == 3:
            n_times = arr.shape[0]
            break

    for name in var_names:
        arr = ds[name].values
        if arr.ndim == 2:
            arr = np.broadcast_to(arr[np.newaxis], (n_times or 1, *arr.shape))
        arrays.append(arr[:, np.newaxis])
    return np.concatenate(arrays, axis=1)


class ClimateDataset(Dataset):
    """Sliding-window dataset over netCDF time series.

    Each sample is (horizon + 1) consecutive timesteps, returned as
    normalized prognostic and forcing tensors.
    """

    def __init__(
        self,
        data_dir: str | Path,
        normalizer: StandardNormalizer,
        forcing_normalizer: StandardNormalizer,
        horizon: int = 6,
        max_samples: int | None = None,
    ):
        self.horizon = horizon
        self.normalizer = normalizer
        self.forcing_normalizer = forcing_normalizer

        data_dir = Path(data_dir)
        nc_files = sorted(data_dir.rglob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found in {data_dir}")

        self.data_prognostic = []
        self.data_forcing = []
        self.windows = []

        for file_idx, nc_path in enumerate(nc_files):
            ds = xr.open_dataset(nc_path)
            n_times = ds.sizes.get("time", ds.sizes.get("sample", 1))

            prog = load_variables(ds, PROGNOSTIC_NAMES)
            forcing = load_variables(ds, FORCING_NAMES)

            prog_tensor = torch.from_numpy(prog).float()
            force_tensor = torch.from_numpy(forcing).float()

            prog_tensor = self.normalizer.normalize(prog_tensor)
            force_tensor = self.forcing_normalizer.normalize(force_tensor)

            self.data_prognostic.append(prog_tensor)
            self.data_forcing.append(force_tensor)

            for t in range(n_times - horizon):
                self.windows.append((file_idx, t))

            ds.close()

        if max_samples is not None:
            self.windows = self.windows[:max_samples]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, t = self.windows[idx]
        prog = self.data_prognostic[file_idx][t : t + self.horizon + 1]
        forcing = self.data_forcing[file_idx][t : t + self.horizon + 1]
        return {"prognostic": prog, "forcing": forcing}


def make_dataloaders(
    data_dir: str,
    horizon: int = 6,
    batch_size: int = 8,
    num_workers: int = 4,
    max_val_samples: int | None = 40,
):
    data_dir = Path(data_dir)
    prog_norm = load_normalizer(data_dir / "centering.nc", data_dir / "scaling.nc", PROGNOSTIC_NAMES)
    forcing_norm = load_normalizer(data_dir / "centering.nc", data_dir / "scaling.nc", FORCING_NAMES)

    train_ds = ClimateDataset(data_dir / "train", prog_norm, forcing_norm, horizon=horizon)
    val_ds = ClimateDataset(data_dir / "val", prog_norm, forcing_norm, horizon=horizon, max_samples=max_val_samples)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
