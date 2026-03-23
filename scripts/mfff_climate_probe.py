#!/usr/bin/env python3
"""M-FFF extreme-event density probe: transport comparison with DYffusion.

Both approaches answer "where do temperature extremes move from t=0 to t=T?":
  - M-FFF:     encode D_0 -> latent -> decode with flow_T -> transported points
  - DYffusion: forecast full field, threshold -> predicted extreme locations

Subcommands:
    extract   - Extract extreme-event (lat, lon) at t=0 and t=T from validation data
    train     - Train M-FFF_0 and M-FFF_T on the two distributions
    evaluate  - Transport via M-FFF, compare with DYffusion predictions
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NLAT, NLON = 32, 64


def make_latlon_grid():
    lats = np.linspace(-90 + 180 / (2 * NLAT), 90 - 180 / (2 * NLAT), NLAT)
    lons = np.linspace(0, 360 - 360 / NLON, NLON)
    return lats, lons


def latlon_to_3d(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def xyz_to_latlon(xyz):
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def field_to_extreme_coords(field, lat_mesh, lon_mesh, percentile):
    """Threshold a field and return (lat, lon) of extreme locations."""
    mask = field >= np.percentile(field, percentile)
    return lat_mesh[mask], lon_mesh[mask], mask


def extract(args):
    import xarray as xr

    val_files = sorted(glob.glob(os.path.join(args.data_dir, "val", "*.nc")))
    if not val_files:
        sys.exit(f"No validation files in {args.data_dir}/val/")

    lats, lons = make_latlon_grid()
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    extremes = {"t0": ([], []), "tT": ([], [])}

    for f in val_files:
        ds = xr.open_dataset(f)
        var = ds[args.variable]
        n_times = len(var.time)

        for start in range(0, n_times - args.lead_steps, args.stride):
            end = start + args.lead_steps

            field_0 = var.isel(time=start).values
            field_T = var.isel(time=end).values

            lat0, lon0, _ = field_to_extreme_coords(field_0, lat_mesh, lon_mesh, args.percentile)
            latT, lonT, _ = field_to_extreme_coords(field_T, lat_mesh, lon_mesh, args.percentile)

            extremes["t0"][0].append(lat0)
            extremes["t0"][1].append(lon0)
            extremes["tT"][0].append(latT)
            extremes["tT"][1].append(lonT)

        ds.close()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    for key in ["t0", "tT"]:
        lat_all = np.concatenate(extremes[key][0])
        lon_all = np.concatenate(extremes[key][1])
        csv_path = os.path.join(out_dir, f"extremes_{key}.csv")
        np.savetxt(csv_path, np.column_stack([lat_all, lon_all]),
                   delimiter=",", fmt="%.6f")
        print(f"{key}: {len(lat_all)} points -> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              subplot_kw={"projection": "mollweide"})
    for ax, key, color, title in [
        (axes[0], "t0", "red", f"D_0: extremes at t=0"),
        (axes[1], "tT", "blue", f"D_T: extremes at t=+{args.lead_steps} steps"),
    ]:
        lat_all = np.concatenate(extremes[key][0])
        lon_all = np.concatenate(extremes[key][1])
        ax.scatter(np.deg2rad(lon_all - 180), np.deg2rad(lat_all),
                   s=0.3, alpha=0.2, c=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{args.variable} >= P{args.percentile}, lead={args.lead_steps} steps ({args.lead_steps*6}h)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "extract_preview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Preview -> {out_dir}/extract_preview.png")


def train(args):
    from src.mfff import ManifoldFreeFormFlow, load_s2_dataset, s2_project
    import torch
    import torch.nn as nn
    import csv
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for tag in ["t0", "tT"]:
        csv_path = os.path.join(args.data_dir, f"extremes_{tag}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"{csv_path} not found. Run 'extract' first.")

        print(f"\n{'='*50}")
        print(f"Training M-FFF_{tag}")
        print(f"{'='*50}")

        train_dl, val_dl = load_s2_dataset(csv_path, args.batch_size)

        model = ManifoldFreeFormFlow(
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            n_vmf_modes=args.n_vmf_modes,
        ).to(device)

        ckpt_dir = Path(args.ckpt_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        total_steps = args.max_epochs * len(train_dl)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            div_factor=25.0, final_div_factor=100.0,
        )

        log_file = ckpt_dir / "train_log.csv"
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_nll", "val_nll"])

        best_val_nll = float("inf")

        for epoch in range(args.max_epochs):
            model.train()
            epoch_loss, epoch_nll, n = 0, 0, 0

            for (x,) in train_dl:
                x = x.to(device)
                if args.noise > 0:
                    x = s2_project(x + torch.randn_like(x) * args.noise)

                nll, recon, reg_z, reg_x1 = model.surrogate_nll(x)
                z_c, x_c = model.consistency_losses(x)

                loss = (
                    nll
                    + args.w_recon * recon
                    + args.w_consist * (z_c + x_c)
                    + 1000 * (reg_z + reg_x1)
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_nll += nll.item()
                n += 1

            avg_loss = epoch_loss / n
            avg_nll = epoch_nll / n
            val_nll = float("nan")

            if (epoch + 1) % args.val_every == 0 or epoch == args.max_epochs - 1:
                model.eval()
                val_lps = []
                with torch.no_grad():
                    for (x,) in val_dl:
                        x = x.to(device)
                        lp = model.exact_log_prob(x)
                        val_lps.append(-lp.mean().item())
                val_nll = sum(val_lps) / len(val_lps)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    torch.save(model.state_dict(), ckpt_dir / "best.pt")

            if (epoch + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [{tag}] Epoch {epoch+1:4d}/{args.max_epochs} | "
                      f"loss={avg_loss:.4f} nll={avg_nll:.4f} val_nll={val_nll:.4f} lr={lr:.2e}")

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, avg_loss, avg_nll, val_nll])

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        print(f"  [{tag}] Done. Best val NLL: {best_val_nll:.4f}")


def evaluate(args):
    import torch
    import xarray as xr
    from src.mfff import ManifoldFreeFormFlow, s2_project

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_0 = ManifoldFreeFormFlow(
        hidden_dim=args.hidden_dim, n_blocks=args.n_blocks, n_vmf_modes=args.n_vmf_modes,
    )
    model_0.load_state_dict(torch.load(args.ckpt_t0, map_location=device, weights_only=True))
    model_0.eval().to(device)

    model_T = ManifoldFreeFormFlow(
        hidden_dim=args.hidden_dim, n_blocks=args.n_blocks, n_vmf_modes=args.n_vmf_modes,
    )
    model_T.load_state_dict(torch.load(args.ckpt_tT, map_location=device, weights_only=True))
    model_T.eval().to(device)

    lats, lons = make_latlon_grid()
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    val_files = sorted(glob.glob(os.path.join(args.data_dir, "val", "*.nc")))
    ds = xr.open_dataset(val_files[0])
    gt_field_0 = ds[args.variable].isel(time=0).values
    gt_field_T = ds[args.variable].isel(time=args.lead_steps).values
    ds.close()

    gt_lat_0, gt_lon_0, gt_mask_0 = field_to_extreme_coords(
        gt_field_0, lat_mesh, lon_mesh, args.percentile)
    gt_lat_T, gt_lon_T, gt_mask_T = field_to_extreme_coords(
        gt_field_T, lat_mesh, lon_mesh, args.percentile)
    gt_locs_T = latlon_to_3d(gt_lat_T, gt_lon_T)

    pred_ds = xr.open_dataset(args.predictions)
    # Each DYffusion step covers args.horizon raw timesteps (36h for horizon=6).
    # Map lead_steps in raw timesteps to the closest DYffusion rollout index.
    dyff_step = round(args.lead_steps / args.horizon) - 1
    dyff_step = max(0, min(dyff_step, pred_ds.sizes["time"] - 1))
    actual_hours = (dyff_step + 1) * args.horizon * 6
    gt_hours = args.lead_steps * 6
    print(f"DYffusion rollout index {dyff_step} ({actual_hours}h) for GT lead {gt_hours}h")
    pred_field_T = pred_ds[args.variable].isel(time=dyff_step).values
    pred_ds.close()

    pred_lat_T, pred_lon_T, pred_mask_T = field_to_extreme_coords(
        pred_field_T, lat_mesh, lon_mesh, args.percentile)
    pred_locs_T = latlon_to_3d(pred_lat_T, pred_lon_T)

    gt_3d_0 = torch.from_numpy(
        latlon_to_3d(gt_lat_0, gt_lon_0).astype(np.float32)
    ).to(device)

    with torch.no_grad():
        z = model_0.encode(gt_3d_0)        # D_0 -> latent
        transported = model_T.decode(z)     # latent -> D_T
    transported = transported.cpu().numpy()
    transported_lat, transported_lon = xyz_to_latlon(transported)

    grid_3d = torch.from_numpy(
        latlon_to_3d(lat_mesh.ravel(), lon_mesh.ravel()).astype(np.float32)
    ).to(device)

    with torch.no_grad():
        log_probs = []
        for i in range(0, len(grid_3d), 512):
            lp = model_T.exact_log_prob(grid_3d[i:i + 512])
            log_probs.append(lp.cpu())
    density_T = torch.cat(log_probs).exp().numpy().reshape(NLAT, NLON)

    energy_transport = compute_energy_distance(transported_lat, transported_lon,
                                                gt_lat_T, gt_lon_T)
    energy_dyffusion = compute_energy_distance(pred_lat_T, pred_lon_T,
                                                gt_lat_T, gt_lon_T)

    with torch.no_grad():
        gt_lp = model_T.exact_log_prob(
            torch.from_numpy(gt_locs_T.astype(np.float32)).to(device)
        ).cpu().numpy()
        pred_lp = model_T.exact_log_prob(
            torch.from_numpy(pred_locs_T.astype(np.float32)).to(device)
        ).cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_field_comparison(gt_field_T, pred_field_T, density_T,
                          lats, lons, args.variable, args.lead_steps, args.output_dir)

    plot_transport_comparison(
        gt_lat_T, gt_lon_T,
        pred_lat_T, pred_lon_T,
        transported_lat, transported_lon,
        args.variable, args.percentile, args.lead_steps, args.output_dir,
    )

    lead_hours = args.lead_steps * 6
    print(f"\n{'='*55}")
    print(f"  Extreme-Event Transport Comparison")
    print(f"  {args.variable} >= P{args.percentile}, GT lead = {args.lead_steps} steps ({lead_hours}h)")
    print(f"  DYffusion rollout step {dyff_step} ({actual_hours}h, mismatch {abs(actual_hours - lead_hours)}h)")
    print(f"{'='*55}")
    print(f"  GT extreme points at t=T:            {len(gt_lat_T)}")
    print(f"  DYffusion predicted extreme points:   {len(pred_lat_T)}")
    print(f"  M-FFF transported points:             {len(transported_lat)}")
    print()
    print(f"  Energy distance to GT extremes at T:")
    print(f"    M-FFF transport:  {energy_transport:.4f}")
    print(f"    DYffusion:        {energy_dyffusion:.4f}")
    print()
    print(f"  M-FFF_T log-prob at locations:")
    print(f"    GT extremes:      {gt_lp.mean():.3f} +/- {gt_lp.std():.3f}")
    print(f"    DYffusion preds:  {pred_lp.mean():.3f} +/- {pred_lp.std():.3f}")
    print(f"\n  Plots -> {args.output_dir}")


def compute_energy_distance(lat1, lon1, lat2, lon2, n_sub=2000):
    X = latlon_to_3d(lat1, lon1)
    Y = latlon_to_3d(lat2, lon2)

    rng = np.random.default_rng(42)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    if len(Y) > n_sub:
        Y = Y[rng.choice(len(Y), n_sub, replace=False)]

    def mean_geo(A, B):
        return np.arccos(np.clip(A @ B.T, -1, 1)).mean()

    return 2 * mean_geo(X, Y) - mean_geo(X, X) - mean_geo(Y, Y)


def plot_field_comparison(gt_field, pred_field, density,
                           lats, lons, var_name, lead_steps, output_dir):
    lon_mesh, lat_mesh = np.meshgrid(np.deg2rad(lons - 180), np.deg2rad(lats))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                              subplot_kw={"projection": "mollweide"})

    im0 = axes[0].pcolormesh(lon_mesh, lat_mesh, gt_field, cmap="RdYlBu_r")
    axes[0].set_title(f"Ground truth at t+{lead_steps*6}h")
    plt.colorbar(im0, ax=axes[0], shrink=0.6)

    im1 = axes[1].pcolormesh(lon_mesh, lat_mesh, pred_field, cmap="RdYlBu_r")
    axes[1].set_title(f"DYffusion at t+{lead_steps*6}h")
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    im2 = axes[2].pcolormesh(lon_mesh, lat_mesh, density, cmap="hot")
    axes[2].set_title(f"M-FFF density at t+{lead_steps*6}h")
    plt.colorbar(im2, ax=axes[2], shrink=0.6)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{var_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fields_{var_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_transport_comparison(gt_lat, gt_lon, pred_lat, pred_lon,
                                trans_lat, trans_lon,
                                var_name, percentile, lead_steps, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                              subplot_kw={"projection": "mollweide"})

    axes[0].scatter(np.deg2rad(gt_lon - 180), np.deg2rad(gt_lat),
                    s=8, c="red", alpha=0.7)
    axes[0].set_title(f"GT extremes at t+{lead_steps*6}h")

    axes[1].scatter(np.deg2rad(pred_lon - 180), np.deg2rad(pred_lat),
                    s=8, c="blue", alpha=0.7)
    axes[1].set_title(f"DYffusion predicted extremes")

    axes[2].scatter(np.deg2rad(trans_lon), np.deg2rad(trans_lat),
                    s=3, c="purple", alpha=0.4)
    axes[2].set_title(f"M-FFF transported extremes")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{var_name} >= P{percentile}: transport comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"transport_{var_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--data_dir", default="data_5deg")
    p_ext.add_argument("--variable", default="surface_temperature")
    p_ext.add_argument("--percentile", type=float, default=95)
    p_ext.add_argument("--lead_steps", type=int, default=40)
    p_ext.add_argument("--stride", type=int, default=4)
    p_ext.add_argument("--output_dir", default="results/mfff_comparison")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data_dir", default="results/mfff_comparison")
    p_train.add_argument("--max_epochs", type=int, default=3000)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--n_blocks", type=int, default=4)
    p_train.add_argument("--n_vmf_modes", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=5e-5)
    p_train.add_argument("--noise", type=float, default=0.0015)
    p_train.add_argument("--gradient_clip", type=float, default=10.0)
    p_train.add_argument("--w_recon", type=float, default=10000.0)
    p_train.add_argument("--w_consist", type=float, default=200.0)
    p_train.add_argument("--val_every", type=int, default=50)
    p_train.add_argument("--log_every", type=int, default=10)
    p_train.add_argument("--ckpt_dir", default="checkpoints/mfff")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--ckpt_t0", default="checkpoints/mfff/t0/best.pt")
    p_eval.add_argument("--ckpt_tT", default="checkpoints/mfff/tT/best.pt")
    p_eval.add_argument("--predictions", default="results/predictions/rollout_predictions.nc")
    p_eval.add_argument("--data_dir", default="data_5deg")
    p_eval.add_argument("--variable", default="surface_temperature")
    p_eval.add_argument("--percentile", type=float, default=95)
    p_eval.add_argument("--lead_steps", type=int, default=40)
    p_eval.add_argument("--horizon", type=int, default=6)
    p_eval.add_argument("--hidden_dim", type=int, default=256)
    p_eval.add_argument("--n_blocks", type=int, default=4)
    p_eval.add_argument("--n_vmf_modes", type=int, default=5)
    p_eval.add_argument("--output_dir", default="results/mfff_comparison")

    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
