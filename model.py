import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mamba_ssm import Mamba2

from config import (
    D_MODEL,
    D_STATE,
    EXPAND,
    HEAD_DIM,
    WINDOW_SIZE,
    N_LAYERS,
)

FEATURE_COLS = {"Date", "Ticker", "target_next_ret"}


class CrossSectionalDataset(Dataset):
    """Return time windows for the same stock universe."""

    def __init__(
        self,
        history: pl.DataFrame,
        targets: pl.DataFrame,
        window_size: int = WINDOW_SIZE,
        tickers=None,
    ):
        history = history.sort(["Date", "Ticker"])
        targets = targets.sort(["Date", "Ticker"])

        self.feature_cols = [c for c in history.columns if c not in FEATURE_COLS]
        self.tickers = np.array(
            sorted(tickers if tickers is not None else history["Ticker"].unique().to_list())
        )
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.window_size = window_size

        history_dates = np.array(
            sorted(history["Date"].unique().to_list()),
            dtype="datetime64[D]",
        )
        target_dates = np.array(
            sorted(targets["Date"].unique().to_list()),
            dtype="datetime64[D]",
        )
        all_dates = np.array(
            sorted(set(history_dates.tolist()) | set(target_dates.tolist())),
            dtype="datetime64[D]",
        )

        self.date_to_index = {date: i for i, date in enumerate(all_dates)}
        self.history_dates = all_dates

        self.target_dates = [
            date for date in target_dates
            if self.date_to_index[date] >= window_size
        ]

        # Dense date x ticker array
        self.features = np.zeros(
            (len(all_dates), len(self.tickers), len(self.feature_cols)),
            dtype=np.float32,
        )
        self.available = np.zeros(
            (len(all_dates), len(self.tickers)),
            dtype=np.float32,
        )

        if history.height > 0:
            date_indices = np.array([
                self.date_to_index[np.datetime64(d, "D")]
                for d in history["Date"]
            ])
            ticker_indices = np.array([
                self.ticker_to_idx[t]
                for t in history["Ticker"]
            ])
            raw_feats = history.select(self.feature_cols).to_numpy().astype(np.float32)
            valid_mask = np.isfinite(raw_feats).all(axis=1)

            d_idx = date_indices[valid_mask]
            t_idx = ticker_indices[valid_mask]
            self.features[d_idx, t_idx] = raw_feats[valid_mask]
            self.available[d_idx, t_idx] = 1.0

        # Targets
        self.targets = {}
        for (date,), group in targets.group_by("Date", maintain_order=True):
            target = np.zeros(len(self.tickers), dtype=np.float32)
            mask = np.zeros(len(self.tickers), dtype=np.float32)

            for ticker, value in group.select(["Ticker", "target_next_ret"]).iter_rows():
                if ticker not in self.ticker_to_idx:
                    continue
                if value is None or not np.isfinite(value):
                    continue

                i = self.ticker_to_idx[ticker]
                target[i] = float(value)
                mask[i] = 1.0

            self.targets[np.datetime64(date, "D")] = (target, mask)

        self.target_dates = [d for d in self.target_dates if d in self.targets]

    def __len__(self):
        return len(self.target_dates)

    def __getitem__(self, index):
        date = self.target_dates[index]
        target_index = self.date_to_index[date]
        start = target_index - self.window_size

        x = self.features[start:target_index]
        available = self.available[start:target_index]
        y, target_mask = self.targets[date]

        # Add availability indicator
        x = np.concatenate([x, available[..., None]], axis=-1)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(target_mask),
            str(date),
            self.tickers.tolist(),
        )


def collate_batch(batch):
    x, y, mask, dates, tickers = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(y),
        torch.stack(mask),
        list(dates),
        tickers[0],
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * torch.nn.functional.silu(gate)


class FeatureMasker(nn.Module):
    """Mask input features."""

    def __init__(self, probability=0.15):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x

        mask = (
            torch.rand(x.size(0), 1, 1, x.size(3), device=x.device)
            > self.probability
        ).to(x.dtype)

        # Keep availability indicator
        mask[..., -1] = 1.0
        return x * mask


class MambaBlock(nn.Module):
    def __init__(self, dim, use_glu=False):
        super().__init__()

        # Make headdim divide d_inner
        d_inner = dim * EXPAND
        headdim = HEAD_DIM if d_inner % HEAD_DIM == 0 else 32

        self.norm1 = RMSNorm(dim)
        self.mamba = Mamba2(
            d_model=dim,
            d_state=D_STATE,
            expand=EXPAND,
            headdim=headdim,
        )
        self.norm2 = RMSNorm(dim)
        self.mixer = GLU(dim) if use_glu else nn.Identity()
        self.norm3 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mixer(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x


class CrossSectionalMamba(nn.Module):
    """Temporal Mamba with cross-sectional context."""

    def __init__(
        self,
        input_dim,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        use_glu=False,
        use_masking=False,
    ):
        super().__init__()
        self.masker = FeatureMasker() if use_masking else nn.Identity()
        self.embedding = nn.Linear(input_dim, d_model)
        self.cross_section = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, use_glu) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # [batch, time, stocks, features]
        available = x[..., -1:].clone()
        x = self.masker(x)
        x = self.embedding(x)

        # Cross-sectional mean
        denom = available.sum(dim=2, keepdim=True).clamp_min(1.0)
        context = (x * available).sum(dim=2, keepdim=True) / denom
        context = self.cross_section(context)

        # Remove inactive stocks
        x = (x + context) * available

        batch, time, stocks, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * stocks, time, dim)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x[:, -1])
        x = self.head(x).reshape(batch, stocks)
        return x


ARCHITECTURES = {
    "base": {"use_glu": False, "use_masking": False},
    "glu": {"use_glu": True, "use_masking": False},
    "mask": {"use_glu": False, "use_masking": True},
    "both": {"use_glu": True, "use_masking": True},
}


def model_factory(architecture, input_dim, **kwargs):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from {list(ARCHITECTURES)}"
        )

    return CrossSectionalMamba(
        input_dim=input_dim,
        **ARCHITECTURES[architecture],
        **kwargs,
    )