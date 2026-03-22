"""Train DYffusion: stage 1 (interpolator) or stage 2 (forecaster).

Usage:
    python -m src.train interpolator --data_dir data_5deg
    python -m src.train forecaster --data_dir data_5deg --interpolator_ckpt checkpoints/interpolator.pt
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from src.data import make_dataloaders
from src.dyffusion import Interpolator, DYffusion


def train(model, train_dl, val_dl, max_epochs: int, output_dir: Path, lr: float = 4e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = GradScaler()

    best_val = float("inf")
    log_path = output_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = model.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast("cuda"):
                    val_loss += model.compute_val_loss(batch).item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")

        log_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.6e}"])
        log_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "best.pt")
        torch.save(model.state_dict(), output_dir / "last.pt")

    log_file.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)

    interp = sub.add_parser("interpolator")
    interp.add_argument("--data_dir", required=True)
    interp.add_argument("--max_epochs", type=int, default=50)
    interp.add_argument("--batch_size", type=int, default=8)

    forecast = sub.add_parser("forecaster")
    forecast.add_argument("--data_dir", required=True)
    forecast.add_argument("--interpolator_ckpt", required=True)
    forecast.add_argument("--max_epochs", type=int, default=50)
    forecast.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    train_dl, val_dl = make_dataloaders(args.data_dir, batch_size=args.batch_size)

    if args.stage == "interpolator":
        model = Interpolator(horizon=6, embed_dim=128, num_layers=4, drop_path_rate=0.1)
    else:
        interpolator = Interpolator(horizon=6, embed_dim=128, num_layers=4, drop_path_rate=0.1)
        interpolator.load_state_dict(torch.load(args.interpolator_ckpt, weights_only=True))
        model = DYffusion(interpolator=interpolator, horizon=6, embed_dim=128, num_layers=4)

    train(model, train_dl, val_dl, args.max_epochs, Path(f"checkpoints/{args.stage}"))


if __name__ == "__main__":
    main()
