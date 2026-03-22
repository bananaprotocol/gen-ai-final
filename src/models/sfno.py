"""Spherical Fourier Neural Operator using torch-harmonics.

Reimplements the SFNO from Spherical DYffusion without nvidia-modulus.
"""

from functools import partial

import torch
import torch.nn as nn
import torch_harmonics as th

from src.models.time_embedding import get_time_embedder


class SpectralConv(nn.Module):
    """Spectral convolution on S2 via SHT with dhconv (isotropic) weights."""

    def __init__(
        self,
        forward_transform: th.RealSHT,
        inverse_transform: th.InverseRealSHT,
        embed_dim: int,
    ):
        super().__init__()
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = inverse_transform.lmax
        self.modes_lon = inverse_transform.mmax

        self.scale_residual = (
            (forward_transform.nlat != inverse_transform.nlat)
            or (forward_transform.nlon != inverse_transform.nlon)
            or (getattr(forward_transform, "grid", None) != getattr(inverse_transform, "grid", None))
        )

        scale = 1.0 / (embed_dim * embed_dim)
        self.weight = nn.Parameter(scale * torch.randn(embed_dim, embed_dim, self.modes_lat, 2))
        self.bias = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        residual = x

        x = x.float()
        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x.clone()).to(dtype)

        w = torch.view_as_complex(self.weight.float())
        x_filtered = torch.einsum("bilm,iol->bolm", x[..., : self.modes_lat, :], w)
        out = torch.zeros_like(x)
        out[..., : self.modes_lat, :] = x_filtered
        x = out.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        x = x.to(dtype) + self.bias
        return x, residual


class FNOBlock(nn.Module):
    """FNO block: norm -> [time scale-shift] -> spectral conv -> skip -> MLP -> drop path."""

    def __init__(
        self,
        forward_transform: th.RealSHT,
        inverse_transform: th.InverseRealSHT,
        embed_dim: int,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        time_emb_dim: int | None = None,
    ):
        super().__init__()
        input_shape = (forward_transform.nlat, forward_transform.nlon)
        output_shape = (inverse_transform.nlat, inverse_transform.nlon)

        self.norm0 = nn.InstanceNorm2d(embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        self.norm1 = nn.InstanceNorm2d(embed_dim, eps=1e-6, affine=True, track_running_stats=False)

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, embed_dim * 2),
            )
        else:
            self.time_mlp = None

        self.filter = SpectralConv(forward_transform, inverse_transform, embed_dim)
        self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        self.act = nn.GELU()

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, embed_dim, 1),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm0(x)

        if self.time_mlp is not None and time_emb is not None:
            t = self.time_mlp(time_emb)
            t = t.unsqueeze(-1).unsqueeze(-1)
            scale, shift = t.chunk(2, dim=1)
            x_norm = x_norm * (scale + 1) + shift

        x_filtered, residual = self.filter(x_norm)
        x_out = x_filtered + self.inner_skip(residual)
        x_out = self.act(x_out)

        x_out = self.norm1(x_out)
        x_out = self.mlp(x_out)
        x_out = self.drop_path(x_out)
        x_out = x_out + residual
        return x_out


class SFNO(nn.Module):
    """Encoder -> N x FNOBlock -> Decoder with big skip.

    Uses equiangular data grid and legendre-gauss internal grid.
    Time conditioning via sinusoidal embedding + MLP scale-shift.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_shape: tuple[int, int] = (32, 64),
        embed_dim: int = 128,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        encoder_layers: int = 1,
        drop_path_rate: float = 0.0,
        dropout_filter: float = 0.0,
        dropout_mlp: float = 0.0,
        big_skip: bool = True,
        with_time_emb: bool = True,
        time_dim_mult: int = 2,
        data_grid: str = "equiangular",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_shape = img_shape
        self.embed_dim = embed_dim
        self.big_skip = big_skip
        self.with_time_emb = with_time_emb

        nlat, nlon = img_shape
        modes_lat = nlat
        modes_lon = nlon // 2 + 1

        self.trans_down = th.RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=data_grid).float()
        self.itrans_up = th.InverseRealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=data_grid).float()
        self.trans = th.RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
        self.itrans = th.InverseRealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

        encoder_modules = []
        current_dim = in_channels
        for _ in range(encoder_layers):
            encoder_modules.append(nn.Conv2d(current_dim, embed_dim, 1, bias=True))
            encoder_modules.append(nn.GELU())
            current_dim = embed_dim
        encoder_modules.append(nn.Conv2d(current_dim, embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, nlat, nlon))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_dim = None
        if with_time_emb:
            self.time_dim = embed_dim * time_dim_mult
            self.time_emb_mlp = get_time_embedder(embed_dim, time_dim_mult)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            first = i == 0
            last = i == num_layers - 1
            fwd = self.trans_down if first else self.trans
            inv = self.itrans_up if last else self.itrans

            block = FNOBlock(
                fwd, inv, embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                time_emb_dim=self.time_dim,
            )
            self.blocks.append(block)

        decoder_in = embed_dim + (in_channels if big_skip else 0)
        decoder_modules = []
        current_dim = decoder_in
        for _ in range(encoder_layers):
            decoder_modules.append(nn.Conv2d(current_dim, embed_dim, 1, bias=True))
            decoder_modules.append(nn.GELU())
            current_dim = embed_dim
        decoder_modules.append(nn.Conv2d(current_dim, out_channels, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            x = torch.cat([x, condition], dim=1)

        if self.big_skip:
            residual = x

        x = self.encoder(x)
        x = x + self.pos_embed

        t_repr = None
        if self.with_time_emb and time is not None:
            t_repr = self.time_emb_mlp(time.float())

        for block in self.blocks:
            x = block(x, time_emb=t_repr)

        if self.big_skip:
            x = torch.cat([x, residual], dim=1)

        x = self.decoder(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob
