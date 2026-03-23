"""Minimal Manifold Free-Form Flow on S2 (unit sphere).

Reimplements the core of Draxler et al. (2024) for density estimation
on the 2-sphere, without geomstats or lightning-trainable dependencies.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
from torch.utils.data import DataLoader, TensorDataset


def s2_project(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def s2_tangent(v, p):
    """Project v onto the tangent plane at p."""
    return v - (v * p).sum(dim=-1, keepdim=True) * p


def s2_tangent_basis(p):
    """Orthonormal basis (B, 2, 3) for the tangent plane at each point in p (B, 3)."""
    ref = torch.zeros_like(p)
    near_pole = p[:, 0].abs() > 0.9
    ref[~near_pole, 0] = 1.0
    ref[near_pole, 1] = 1.0

    t1 = ref - (ref * p).sum(-1, keepdim=True) * p
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t2 = torch.linalg.cross(p, t1)
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.stack([t1, t2], dim=1)


def s2_random_tangent_vec(p):
    """Sample a unit tangent vector at p on S2."""
    v = torch.randn_like(p)
    v = s2_tangent(v, p)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class VMFMixture(nn.Module):
    """Mixture of von Mises-Fisher distributions on S2."""

    def __init__(self, n_modes=5, learnable=True):
        super().__init__()
        self.n_modes = n_modes

        locs = torch.randn(n_modes, 3)
        locs = locs / locs.norm(dim=-1, keepdim=True)
        self.locs = nn.Parameter(locs, requires_grad=learnable)
        self.log_kappa = nn.Parameter(
            torch.full((n_modes,), math.log(10.0)), requires_grad=learnable
        )
        self.weight_logits = nn.Parameter(
            torch.zeros(n_modes), requires_grad=learnable
        )

    @property
    def kappa(self):
        return self.log_kappa.exp().clamp(min=1e-4, max=1e4)

    @property
    def weights(self):
        return self.weight_logits.softmax(dim=0)

    def _vmf_log_norm(self, kappa):
        # log C(k) = log(k) - log(4*pi) - log(sinh(k))
        # numerically stable via log(sinh(k)) = k + log(1 - exp(-2k)) - log(2)
        return (
            torch.log(kappa + 1e-8)
            - math.log(4 * math.pi)
            - kappa
            - torch.log1p(-torch.exp(-2 * kappa))
            + math.log(2)
        )

    def log_prob(self, x):
        mu = s2_project(self.locs)
        kappa = self.kappa
        log_w = torch.log(self.weights + 1e-10)

        dots = x @ mu.T
        log_norm = self._vmf_log_norm(kappa)
        component_lp = log_norm.unsqueeze(0) + kappa.unsqueeze(0) * dots
        return torch.logsumexp(log_w.unsqueeze(0) + component_lp, dim=1)

    def sample(self, n):
        device = self.locs.device
        mu = s2_project(self.locs)
        kappa = self.kappa

        indices = torch.multinomial(self.weights, n, replacement=True)
        mu_sel = mu[indices]
        kappa_sel = kappa[indices]

        u = torch.rand(n, device=device).clamp(1e-7, 1 - 1e-7)
        w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * kappa_sel)) / kappa_sel

        phi = 2 * math.pi * torch.rand(n, device=device)
        r = torch.sqrt((1 - w * w).clamp(min=0))
        samples_pole = torch.stack([r * torch.cos(phi), r * torch.sin(phi), w], dim=-1)

        return s2_project(_rotate_pole_to_mu(samples_pole, mu_sel))


def _rotate_pole_to_mu(x, mu):
    """Rotate points sampled around [0,0,1] to be centered on mu."""
    pole = torch.tensor([0.0, 0.0, 1.0], device=mu.device).expand_as(mu)
    cross = torch.linalg.cross(pole, mu)
    sin_a = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_a = (pole * mu).sum(dim=-1, keepdim=True)

    near_id = sin_a.squeeze(-1) < 1e-6
    near_anti = cos_a.squeeze(-1) < -0.99
    axis = cross / sin_a

    x_rot = (
        x * cos_a
        + torch.linalg.cross(axis, x) * sin_a
        + axis * (axis * x).sum(dim=-1, keepdim=True) * (1 - cos_a)
    )
    x_rot[near_id] = x[near_id]
    x_rot[near_anti] = x[near_anti] * torch.tensor([1.0, 1.0, -1.0], device=x.device)
    return x_rot


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), Sin(), nn.Linear(dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class S2Network(nn.Module):
    def __init__(self, hidden_dim=256, n_blocks=4):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), Sin()]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ManifoldFreeFormFlow(nn.Module):
    def __init__(self, hidden_dim=256, n_blocks=4, n_vmf_modes=5):
        super().__init__()
        self.encoder = S2Network(hidden_dim, n_blocks)
        self.decoder = S2Network(hidden_dim, n_blocks)
        self.latent = VMFMixture(n_modes=n_vmf_modes, learnable=True)

    def encode(self, x, project=True):
        z = self.encoder(x)
        return s2_project(z) if project else z

    def decode(self, z, project=True):
        x = self.decoder(z)
        return s2_project(x) if project else x

    def surrogate_nll(self, x):
        """Hutchinson surrogate for the change-of-variables log-det.

        Returns (nll, reconstruction, reg_z, reg_x1) losses.
        """
        x = x.detach().requires_grad_(True)
        z_raw = self.encoder(x)
        z = s2_project(z_raw)
        reg_z = (z_raw - z.detach()).pow(2).sum(-1).mean()

        v = s2_random_tangent_vec(z.detach()) * math.sqrt(3)

        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1_raw = self.decoder(dual_z)
            dual_x1 = s2_project(dual_x1_raw)
            x1, v1 = unpack_dual(dual_x1)

        reg_x1 = (unpack_dual(dual_x1_raw)[0] - x1.detach()).pow(2).sum(-1).mean()

        (v2,) = grad(z, x, v, create_graph=True)
        surrogate = (v2 * v1.detach()).sum(-1)

        latent_lp = self.latent.log_prob(z)
        nll = -(latent_lp + surrogate).mean()
        recon = (x1 - x).pow(2).sum(-1).mean()

        return nll, recon, reg_z, reg_x1

    def exact_log_prob(self, x):
        """Exact log-probability via decoder Jacobian (evaluation only)."""
        z = self.encode(x)

        def decode_single(z_single):
            return s2_project(self.decoder(z_single.unsqueeze(0))).squeeze(0)

        jacs = torch.vmap(torch.func.jacrev(decode_single))(z)
        Q_z = s2_tangent_basis(z)
        x1 = self.decode(z)
        Q_x1 = s2_tangent_basis(x1)
        J_tan = Q_x1 @ jacs @ Q_z.transpose(1, 2)
        log_det = torch.linalg.slogdet(J_tan)[1]

        return self.latent.log_prob(z) - log_det

    def consistency_losses(self, x):
        """Round-trip consistency: z_sample and x_sample reconstruction."""
        with torch.no_grad():
            z_sample = self.latent.sample(len(x)).to(x.device)

        x_from_z = self.decode(z_sample)
        z_roundtrip = self.encode(x_from_z)
        z_loss = (z_roundtrip - z_sample.detach()).pow(2).sum(-1).mean()

        x_roundtrip = self.decode(z_roundtrip)
        x_loss = (x_roundtrip - x_from_z.detach()).pow(2).sum(-1).mean()

        return z_loss, x_loss

    @torch.no_grad()
    def sample(self, n):
        z = self.latent.sample(n).to(next(self.parameters()).device)
        return self.decode(z)

    @torch.no_grad()
    def log_prob(self, x):
        return self.exact_log_prob(x)


def load_s2_dataset(csv_path, batch_size=32, seed=42):
    data = np.loadtxt(csv_path, delimiter=",")
    lat, lon = np.deg2rad(data[:, 0]), np.deg2rad(data[:, 1])
    points = np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1).astype(np.float32)

    rng = np.random.default_rng(seed)
    points = torch.from_numpy(points[rng.permutation(len(points))])

    n = len(points)
    n_train, n_val = int(0.8 * n), int(0.1 * n)

    train_dl = DataLoader(
        TensorDataset(points[:n_train]), batch_size=batch_size,
        shuffle=True, num_workers=4, persistent_workers=True,
    )
    val_dl = DataLoader(
        TensorDataset(points[n_train:n_train + n_val]), batch_size=batch_size,
        num_workers=2, persistent_workers=True,
    )
    return train_dl, val_dl


