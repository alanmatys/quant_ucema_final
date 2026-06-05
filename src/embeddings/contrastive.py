"""Contrastive self-supervised embeddings for HRP.

Simple SSL recipe for time-series:
- For each asset, slice the returns history into overlapping windows.
- Positive pairs: two random temporal augmentations of the same window
  (Gaussian jitter on returns).
- Negative pairs: windows from other assets.
- NT-Xent loss (SimCLR-style) drives the encoder.
- Per-asset embedding: mean of the encoder outputs over all windows
  from that asset.

Encoder: tiny 1D CNN, intentionally small so CPU training is feasible.

References:
    Chen et al. (2020). A Simple Framework for Contrastive Learning of
    Visual Representations (SimCLR; we transfer the NT-Xent loss).
    Yue et al. (2022). TS2Vec (we borrow the hierarchical-window idea).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.embeddings.channels import build_asset_channels

# Determinism + single-threaded PyTorch.
#
# set_num_threads(1) / set_num_interop_threads(1): besides avoiding the
# segfaults observed under pytest on macOS (PyTorch's threadpool vs the
# test-runner's signal handlers), a single thread removes the
# non-deterministic summation order of multi-threaded BLAS/ATen
# reductions — the dominant source of run-to-run drift in the trained
# encoder, which propagates through single-linkage clustering into the
# HRP weights.
#
# use_deterministic_algorithms pins kernel selection so a fixed PyTorch
# build yields bit-identical encoder weights. warn_only=True degrades a
# missing deterministic kernel to a warning rather than an exception —
# the HRPContrastive.get_weights() fallback would otherwise swallow it
# silently and quietly revert the strategy to plain HRP.
torch.set_num_threads(1)
try:  # interop threads can only be set before any parallel work starts
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

class TinyConvEncoder(nn.Module):
    """1D CNN with 3 conv layers + global average pooling.

    Input shape: (B, in_channels, window_len). Output shape: (B, emb_dim).
    `in_channels` > 1 feeds multi-channel windows (returns + volume + ...).
    """

    def __init__(self, emb_dim: int = 32, hidden: int = 16, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden * 2, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)  # (B, emb_dim)
        return F.normalize(h, dim=-1)


# ----------------------------------------------------------------------
# Dataset with anchor + positive augmentation
# ----------------------------------------------------------------------

class WindowDataset(Dataset):
    """Per-window samples with on-the-fly augmentation.

    Each item yields (anchor, positive) — two augmented views of the same
    (C, window) multi-channel window. Augmentation = Gaussian noise +
    random scaling, applied independently per channel and timestep.
    """

    def __init__(self, asset_channels: dict[str, np.ndarray], window: int,
                 stride: int = 5, noise_std: float = 0.02):
        """Args:
            asset_channels: {asset: (C, T) standardised array} from
                `channels.build_asset_channels`.
        """
        self.window = window
        self.noise_std = noise_std
        self.windows: list[tuple[str, np.ndarray]] = []
        for asset, arr in asset_channels.items():
            T = arr.shape[1]
            for start in range(0, T - window + 1, stride):
                self.windows.append((asset, arr[:, start:start + window]))

    def __len__(self):
        return len(self.windows)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x2 = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
        scale = 1.0 + np.random.randn() * 0.05
        return x2 * scale

    def __getitem__(self, idx: int):
        _, x = self.windows[idx]
        a = self._augment(x)
        p = self._augment(x)
        return (
            torch.from_numpy(a),  # (C, W)
            torch.from_numpy(p),
        )

    @property
    def assets(self) -> list[str]:
        return [a for a, _ in self.windows]


# ----------------------------------------------------------------------
# NT-Xent contrastive loss
# ----------------------------------------------------------------------

def nt_xent(anchors: torch.Tensor, positives: torch.Tensor,
            temperature: float = 0.5) -> torch.Tensor:
    """Symmetric NT-Xent loss (SimCLR)."""
    B = anchors.shape[0]
    z = torch.cat([anchors, positives], dim=0)  # (2B, D)
    sim = z @ z.t() / temperature                # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: i-th anchor matches (i+B)-th, and vice versa
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ----------------------------------------------------------------------
# Training and embedding extraction
# ----------------------------------------------------------------------

def train_contrastive_encoder(
    returns: pd.DataFrame,
    window: int = 40,
    stride: int = 5,
    emb_dim: int = 32,
    hidden: int = 16,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    extra_channels: list[pd.DataFrame] | None = None,
    log_transform: bool = True,
) -> tuple[TinyConvEncoder, WindowDataset]:
    """Train the contrastive encoder on a returns DataFrame.

    Args:
        extra_channels: optional list of T x N panels (quote volume, trade
            count, ...) fed as additional input channels alongside returns.
        log_transform: see `channels.build_asset_channels` — pass False for
            pre-tamed channels from `features.derive_channel_panels`.

    Returns the trained encoder and the dataset (needed for inference).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    asset_channels = build_asset_channels(returns, extra_channels,
                                          log_transform=log_transform)
    n_channels = next(iter(asset_channels.values())).shape[0]
    ds = WindowDataset(asset_channels, window=window, stride=stride)
    if len(ds) < 4:
        raise RuntimeError(
            f"too few windows ({len(ds)}); reduce window or stride"
        )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    encoder = TinyConvEncoder(emb_dim=emb_dim, hidden=hidden,
                              in_channels=n_channels).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    encoder.train()
    for _ in range(epochs):
        for anchor, positive in dl:
            anchor, positive = anchor.to(device), positive.to(device)
            a = encoder(anchor); p = encoder(positive)
            loss = nt_xent(a, p)
            opt.zero_grad(); loss.backward(); opt.step()

    encoder.eval()
    return encoder, ds


def asset_embeddings_from_encoder(
    encoder: TinyConvEncoder,
    returns: pd.DataFrame,
    window: int,
    stride: int = 5,
    device: str = "cpu",
    extra_channels: list[pd.DataFrame] | None = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Compute per-asset embeddings as the mean encoder output over windows.

    For each asset, slice (C, window) windows (no augmentation), encode
    each, then take the mean of the embeddings as the asset representation.
    `extra_channels` and `log_transform` must match what
    `train_contrastive_encoder` was given.
    """
    encoder.eval()
    asset_channels = build_asset_channels(returns, extra_channels,
                                          log_transform=log_transform)
    asset_emb: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for asset, arr in asset_channels.items():
            T = arr.shape[1]
            if T < window:
                continue
            slices = [arr[:, s:s + window]
                      for s in range(0, T - window + 1, stride)]
            X = np.stack(slices)                       # (M, C, W)
            tens = torch.from_numpy(X).to(device)
            emb = encoder(tens).cpu().numpy()          # (M, D)
            asset_emb[asset] = emb.mean(axis=0)
    if not asset_emb:
        raise RuntimeError("no assets had enough history for embeddings")
    df = pd.DataFrame(np.vstack(list(asset_emb.values())),
                      index=list(asset_emb.keys()))
    return df


def embeddings_to_distance(emb_df: pd.DataFrame) -> pd.DataFrame:
    """L2-normalised cosine distance in [0, sqrt(2)]; diagonal = 0."""
    X = emb_df.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1e-12, norms)
    Xn = X / norms
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    dist = np.sqrt(np.maximum(1.0 - sim, 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return pd.DataFrame(dist, index=emb_df.index, columns=emb_df.index)
