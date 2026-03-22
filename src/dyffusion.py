"""Two-stage DYffusion: interpolation, forecasting, and cold sampling.

Based on Cachay et al. (NeurIPS 2024). Stage 1 trains an interpolator
I(x_0, x_T, t) -> x_t, stage 2 trains a forecaster F(x_t, x_0, t) -> x_T
with the interpolator frozen. Inference uses cold sampling for refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sfno import SFNO
from src.data import PROGNOSTIC_NAMES, FORCING_NAMES

N_PROG = len(PROGNOSTIC_NAMES)
N_FORCE = len(FORCING_NAMES)


class Interpolator(nn.Module):
    def __init__(self, horizon: int = 6, embed_dim: int = 128, num_layers: int = 4,
                 mlp_ratio: float = 2.0, drop_path_rate: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.sfno = SFNO(
            in_channels=2 * N_PROG + N_FORCE,
            out_channels=N_PROG,
            embed_dim=embed_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            with_time_emb=True,
        )

    def forward(self, x_0, x_T, forcing, t):
        inputs = torch.cat([x_0, x_T, forcing], dim=1)
        return self.sfno(inputs, time=t)

    def compute_loss(self, batch):
        prog = batch["prognostic"]
        forcing = batch["forcing"]
        x_0, x_T, forcing_0 = prog[:, 0], prog[:, -1], forcing[:, 0]
        B = x_0.shape[0]
        t = torch.randint(1, self.horizon, (B,), device=x_0.device)
        target = prog[torch.arange(B, device=x_0.device), t]
        pred = self.forward(x_0, x_T, forcing_0, t)
        return F.l1_loss(pred, target)

    def compute_val_loss(self, batch):
        prog = batch["prognostic"]
        forcing = batch["forcing"]
        x_0, x_T, forcing_0 = prog[:, 0], prog[:, -1], forcing[:, 0]
        B = x_0.shape[0]
        total = sum(
            F.l1_loss(
                self.forward(x_0, x_T, forcing_0, torch.full((B,), t, device=x_0.device, dtype=torch.long)),
                prog[torch.arange(B, device=x_0.device), t],
            )
            for t in range(1, self.horizon)
        )
        return total / (self.horizon - 1)


class DYffusion(nn.Module):
    def __init__(self, interpolator: Interpolator, horizon: int = 6, embed_dim: int = 128,
                 num_layers: int = 4, mlp_ratio: float = 2.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.num_timesteps = horizon

        self.interpolator = interpolator
        self.interpolator.eval()
        for p in self.interpolator.parameters():
            p.requires_grad = False

        self.sfno = SFNO(
            in_channels=N_PROG + N_PROG + N_FORCE,
            out_channels=N_PROG,
            embed_dim=embed_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            with_time_emb=True,
        )

    def predict_x_T(self, x_t, x_0, forcing, t):
        condition = torch.cat([x_0, forcing], dim=1)
        return self.sfno(x_t, time=t, condition=condition)

    def interpolate(self, x_0, x_T, forcing, t, enable_dropout=False):
        if enable_dropout:
            self.interpolator.train()
        else:
            self.interpolator.eval()
        with torch.no_grad():
            x_t = self.interpolator(x_0, x_T, forcing, t)
        self.interpolator.eval()
        return x_t

    def _get_x_s(self, x_0, x_T, forcing_0, s, enable_dropout=False):
        nonzero_mask = s > 0
        if not nonzero_mask.any():
            return x_0.clone()
        x_s = x_0.clone()
        x_interp = self.interpolate(
            x_0[nonzero_mask], x_T[nonzero_mask],
            forcing_0[nonzero_mask], s[nonzero_mask],
            enable_dropout=enable_dropout,
        )
        x_s[nonzero_mask] = x_interp.to(x_s.dtype)
        return x_s

    def compute_loss(self, batch):
        prog = batch["prognostic"]
        forcing = batch["forcing"]
        B = prog.shape[0]
        x_0, x_T, forcing_0 = prog[:, 0], prog[:, -1], forcing[:, 0]
        s = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        x_s = self._get_x_s(x_0, x_T, forcing_0, s, enable_dropout=True)
        x_T_pred = self.predict_x_T(x_s, x_0, forcing_0, s)
        return F.l1_loss(x_T_pred, x_T)

    def compute_val_loss(self, batch):
        prog = batch["prognostic"]
        forcing = batch["forcing"]
        B = prog.shape[0]
        x_0, x_T, forcing_0 = prog[:, 0], prog[:, -1], forcing[:, 0]
        total = 0.0
        for s_val in range(self.num_timesteps):
            s = torch.full((B,), s_val, device=x_0.device, dtype=torch.long)
            x_s = self._get_x_s(x_0, x_T, forcing_0, s)
            x_T_pred = self.predict_x_T(x_s, x_0, forcing_0, s)
            total += F.l1_loss(x_T_pred, x_T)
        return total / self.num_timesteps

    @torch.inference_mode()
    def cold_sample(self, x_0, forcing):
        x_s = x_0.clone()
        x_T_hat = None
        for s in range(self.num_timesteps):
            is_last = s == self.num_timesteps - 1
            s_tensor = torch.full((x_0.shape[0],), s, device=x_0.device, dtype=torch.long)
            x_T_hat = self.predict_x_T(x_s, x_0, forcing, s_tensor)
            I_s = self.interpolate(x_0, x_T_hat, forcing, s_tensor) if s > 0 else x_s
            if is_last:
                I_next = x_T_hat
            else:
                s_next = torch.full_like(s_tensor, s + 1)
                I_next = self.interpolate(x_0, x_T_hat, forcing, s_next)
            x_s = x_s + (I_next - I_s)
        return x_T_hat
