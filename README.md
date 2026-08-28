# Spherical DYffusion at 5.625°

A reimplementation of [Spherical DYffusion](https://arxiv.org/abs/2406.14798) (Cachay et al.,
2024), a probabilistic climate emulator that combines temporal diffusion with Spherical
Fourier Neural Operators (SFNOs). Given an atmospheric state, it produces an autoregressive
rollout days to months ahead.

Two ideas carry it. The SFNO backbone applies spectral convolutions in the spherical harmonic
domain, which avoids the pole artifacts that standard 2D convolutions produce on a
latitude-longitude grid. DYffusion replaces the noise schedule of a standard diffusion model
with the forecast time axis, so every intermediate step of the diffusion process is a real
atmospheric state at a real lead time rather than a noisy mixture.

The original trains at 1° (180x360) on multiple GPUs. We train at 5.625° (32x64) on a single
16 GB consumer GPU, which asks whether DYffusion's results follow from the framework itself or
from capacity and resolution. As a secondary experiment, we compare the model against
[Manifold Free-Form Flows](https://arxiv.org/abs/2312.09852) (M-FFF), a normalizing flow for
density estimation on S², using an extreme-event transport probe.

## What's here

About 1900 lines of Python covering the whole pipeline, from preprocessing through training,
inference and evaluation. The stack is essentially torch, torch-harmonics, xarray and scipy.

| Path | What it does |
|---|---|
| `scripts/prepare_data.py` | Downsample FV3GFS 1° to 5.625°, train/val split, normalization stats |
| `src/data.py` | Variable lists, sliding-window dataset over the NetCDF files |
| `src/normalization.py` | Per-variable standardization from `centering.nc` / `scaling.nc` |
| `src/models/sfno.py` | Spherical Fourier Neural Operator (encoder, FNO blocks, decoder) |
| `src/models/time_embedding.py` | Sinusoidal timestep embedding and feature modulation |
| `src/dyffusion.py` | Interpolator, forecaster, cold sampling |
| `src/train.py` | Two-stage training |
| `src/inference.py` | Autoregressive rollout |
| `src/evaluate.py` | RMSE vs. lead time, persistence skill, temperature drift |
| `scripts/evaluate_rollout.py` | Extended evaluation: stability analysis, power spectra |
| `src/mfff.py` | Manifold Free-Form Flows on S² with a von Mises-Fisher mixture latent |
| `scripts/mfff_climate_probe.py` | Extreme-event transport probe (extract / train / evaluate) |

## Setup

With Nix:

```bash
nix develop   # provides python and uv, runs `uv sync --extra dev`, activates .venv
```

Otherwise:

```bash
uv sync --extra dev
source .venv/bin/activate
```

Torch is pinned to the ROCm 6.4 index (see `pyproject.toml`). Swap the index for CUDA.

## Data

We use the FV3GFS validation subsample published with
[ACE](https://arxiv.org/abs/2310.02074) (Watt-Meyer et al., 2023):

[zenodo.org/records/10791087](https://zenodo.org/records/10791087), *AI2 Climate Emulator
(ACE) model checkpoint and sample data*, CC-BY-4.0, DOI
[10.5281/zenodo.10791087](https://doi.org/10.5281/zenodo.10791087).

Twelve monthly NetCDF files covering 2021 at 1° resolution, 6-hourly, about 1.7 GB each.
The record is 22.9 GB in total because it also contains `ace_ckpt.tar`, which we do not use.
Download the files into `data/`:

```bash
uvx zenodo_get 10.5281/zenodo.10791087 -o data
```

## Pipeline

```bash
# 1. Preprocess: 1° to 5.625°, Jan-Oct train / Nov-Dec val, centering.nc + scaling.nc
python scripts/prepare_data.py --input_dir data --output_dir data_5deg

# 2. Stage 1, interpolator -> checkpoints/interpolator/best.pt
python -m src.train interpolator --data_dir data_5deg --max_epochs 50

# 3. Stage 2, forecaster with the interpolator frozen -> checkpoints/forecaster/best.pt
python -m src.train forecaster --data_dir data_5deg \
    --interpolator_ckpt checkpoints/interpolator/best.pt --max_epochs 50

# 4. Autoregressive rollout (100 steps x 36 h = 150 days)
python -m src.inference \
    --forecaster_ckpt checkpoints/forecaster/best.pt \
    --interpolator_ckpt checkpoints/interpolator/best.pt \
    --data_dir data_5deg --output_dir results/predictions --num_steps 100

# 5. Evaluate -> results/evaluation/{rmse_vs_leadtime,temperature_drift}.png
python -m src.evaluate --predictions results/predictions/rollout_predictions.nc \
    --validation_dir data_5deg/val --output_dir results/evaluation
```

### M-FFF comparison

```bash
python scripts/mfff_climate_probe.py extract   --data_dir data_5deg
python scripts/mfff_climate_probe.py train     --data_dir results/mfff_comparison
python scripts/mfff_climate_probe.py evaluate  --predictions results/predictions/rollout_predictions.nc
```

The two models solve different tasks, field forecasting against density estimation on S², so a
direct benchmark is not meaningful. The probe brings them onto common ground by reducing
DYffusion's gridded output to a point distribution, and asks: given the surface temperature
extremes at t = 0, where do they end up at t = T?

`extract` sweeps the validation set with a sliding window at stride 4 steps (1 day) and, for
each window, records the surface temperature P95 locations at its start and 40 steps (10 days)
later. That builds two point sets, D_0 and D_T, of roughly 4223 points each (41 windows x 103
cells above the threshold). `train` fits one flow per set. `evaluate` takes a single window,
transports its t = 0 points by encoding through M-FFF_0 and decoding through M-FFF_T, and
compares that against the extremes obtained by thresholding DYffusion's forecast field at P95.
DYffusion is read off rollout step 6, so its lead time is 252 h against the ground truth's
240 h; the script prints the mismatch.

## Configuration

Reduced to fit 16 GB of VRAM. Everything else follows the original.

| Setting | Original | Ours |
|---|---|---|
| Resolution | 180x360 | 32x64 |
| Embedding dimension | 256 | 128 |
| FNO blocks | 8 | 4 |
| Spectral layers per block | 3 | 1 |
| Total spectral convolutions | 24 | 4 |

Channel counts are unchanged: 70 in, 34 out. The interpolator sees the initial state, the
target state and 2 forcings; the forecaster sees the interpolated intermediate state x_t,
conditioned on the initial state and the same 2 forcings. The spectral convolution operator
(`dhconv`), the scale factor and the hard thresholding fraction all follow the original.

Training uses AdamW, lr 4e-4, weight decay 5e-3, cosine annealing, gradient clipping at 0.5,
batch size 8, AMP fp16, and a DYffusion horizon of H = 6 (36 h per rollout step).

## Results

The baseline is persistence: repeat the initial condition at every lead time. It is strong at
short range, because atmospheric states are autocorrelated. Skill score is
1 − RMSE_model / RMSE_persistence, so positive means better than persistence, and RMSE is
area-weighted for the latitude-dependent cell sizes. The rollout runs 150 days, but the two
months of validation data run out after 40 steps, so every number below stops at 60 days.

**Short range.** The model beats persistence out to about 7 days for most variables. At 1.5
days the skill score is +55 % on surface temperature, +30 % on surface pressure, +28 to +43 %
on air temperature depending on level, and +23 to +30 % on wind.

**Long range.** Skill turns negative for most variables after 10 to 15 days, and global mean
surface temperature drifts about +3 K over 60 days of autoregressive rollout. We attribute
this to the reduced SFNO capacity (4 spectral convolutions against the original's 24) and to
holding DSWRFtoa forcing constant during inference, which removes the diurnal and seasonal
cycle from the model's input.

**Transport probe.** M-FFF transport reaches an energy distance of 0.666 to the ground truth
extremes at t = T, against 2.035 for DYffusion. Energy distance is zero only when the two
distributions coincide, so lower is better. The second metric scores locations under M-FFF_T
rather than comparing the two models against each other: the ground truth extremes reach a
mean log-probability of −0.845 and DYffusion's predicted extremes −4.527, so the forecast
places its extremes where the learned density of future extremes is low. Read it as a weak
check, since M-FFF_T was fitted on those ground truth points. DYffusion's warm bias lifts the
whole field, which pushes the P95 threshold poleward.

## Limitations

We run a single deterministic rollout from one initial condition, so there are no ensembles
and no probabilistic metrics such as CRPS or spread-skill. Forcing is held constant throughout
the rollout, since the dataset does not provide future forcing fields. The M-FFF flows overfit
the roughly 4223 extracted extreme points (training NLL about −1.1, validation about 0.8).

## License

MIT, see [`LICENSE`](LICENSE). The FV3GFS data is CC-BY-4.0 and is not covered by it.

## References

Cachay et al., *Probabilistic Emulation of a Global Climate Model with Spherical DYffusion*, NeurIPS 2024.
Bonev et al., *Spherical Fourier Neural Operators*, ICML 2023.
Sorrenson et al., *Learning Distributions on Manifolds with Free-Form Flows*, 2023.
Watt-Meyer et al., *ACE: A fast, skillful learned global atmospheric model*, 2023 (FV3GFS data).
