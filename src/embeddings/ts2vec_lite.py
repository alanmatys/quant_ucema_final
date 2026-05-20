"""TS2Vec-lite: minimal time-series contrastive embedding for HRP.

Inspired by Yue et al. (2022) "TS2Vec: Towards Universal Representation
of Time Series", but with substantial simplifications appropriate for
this paper's scope:

- Single-scale (no hierarchical pooling).
- Random-mask augmentation (TS2Vec's signature: same series, two random
  timestamp masks) — distinct from contrastive.py's Gaussian-jitter scheme.
- Tiny dilated-CNN encoder.

Output: per-asset embedding vector → cosine distance → HRP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.embeddings.channels import build_asset_channels

# Single-threaded PyTorch (see contrastive.py for rationale).
torch.set_num_threads(1)


class DilatedConvEncoder(nn.Module):
    """Dilated 1D-CNN encoder (TS2Vec-style).

    `in_channels` > 1 feeds multi-channel windows (returns + volume + ...).
    """

    def __init__(self, emb_dim: int = 32, hidden: int = 16, in_channels: int = 1):
        super().__init__()
        # Three dilated conv layers progressively widening the receptive field
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden * 2, emb_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return F.normalize(h, dim=-1)


class MaskedWindowDataset(Dataset):
    """Each item yields the same window with two random-mask augmentations.

    TS2Vec's signature augmentation: randomly zero out a fraction of
    timestamps with two independent masks per window, generating two
    "views" that the encoder must map to similar representations. The
    timestamp mask is shared across channels of a multi-channel window.
    """

    def __init__(self, asset_channels: dict[str, np.ndarray], window: int,
                 stride: int = 5, mask_ratio: float = 0.3):
        """Args:
            asset_channels: {asset: (C, T) standardised array} from
                `channels.build_asset_channels`.
        """
        self.window = window
        self.mask_ratio = mask_ratio
        self.windows: list[tuple[str, np.ndarray]] = []
        for asset, arr in asset_channels.items():
            T = arr.shape[1]
            for start in range(0, T - window + 1, stride):
                self.windows.append((asset, arr[:, start:start + window]))

    def __len__(self):
        return len(self.windows)

    def _mask(self, x: np.ndarray) -> np.ndarray:
        # x is (C, W); mask whole timestamps (shared across channels)
        mask = (np.random.rand(x.shape[1]) > self.mask_ratio).astype(np.float32)
        return x * mask

    def __getitem__(self, idx: int):
        _, x = self.windows[idx]
        v1 = self._mask(x)
        v2 = self._mask(x)
        return (
            torch.from_numpy(v1),  # (C, W)
            torch.from_numpy(v2),
        )


def _nt_xent(anchors: torch.Tensor, positives: torch.Tensor,
             temperature: float = 0.5) -> torch.Tensor:
    """Symmetric NT-Xent loss."""
    B = anchors.shape[0]
    z = torch.cat([anchors, positives], dim=0)
    sim = z @ z.t() / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def train_ts2vec_lite_encoder(
    returns: pd.DataFrame,
    window: int = 40,
    stride: int = 5,
    emb_dim: int = 32,
    hidden: int = 16,
    mask_ratio: float = 0.3,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    extra_channels: list[pd.DataFrame] | None = None,
) -> DilatedConvEncoder:
    """Args:
        extra_channels: optional list of T x N panels (quote volume, trade
            count, ...) fed as additional input channels alongside returns.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    asset_channels = build_asset_channels(returns, extra_channels)
    n_channels = next(iter(asset_channels.values())).shape[0]
    ds = MaskedWindowDataset(asset_channels, window=window, stride=stride,
                             mask_ratio=mask_ratio)
    if len(ds) < 4:
        raise RuntimeError(f"too few windows ({len(ds)}); reduce window or stride")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    enc = DilatedConvEncoder(emb_dim=emb_dim, hidden=hidden,
                             in_channels=n_channels).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    enc.train()
    for _ in range(epochs):
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            a = enc(v1); p = enc(v2)
            loss = _nt_xent(a, p)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    return enc


def asset_embeddings_from_encoder(
    encoder: DilatedConvEncoder,
    returns: pd.DataFrame,
    window: int,
    stride: int = 5,
    device: str = "cpu",
    extra_channels: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Mean-pooled per-asset embeddings (no masking at inference).

    `extra_channels` must match what `train_ts2vec_lite_encoder` was given.
    """
    encoder.eval()
    asset_channels = build_asset_channels(returns, extra_channels)
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for asset, arr in asset_channels.items():
            T = arr.shape[1]
            if T < window:
                continue
            slices = [arr[:, s:s + window]
                      for s in range(0, T - window + 1, stride)]
            X = np.stack(slices)                   # (M, C, W)
            tens = torch.from_numpy(X).to(device)
            emb = encoder(tens).cpu().numpy()
            out[asset] = emb.mean(axis=0)
    if not out:
        raise RuntimeError("no assets had enough history")
    return pd.DataFrame(np.vstack(list(out.values())), index=list(out.keys()))


def embeddings_to_distance(emb_df: pd.DataFrame) -> pd.DataFrame:
    X = emb_df.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1e-12, norms)
    Xn = X / norms
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    dist = np.sqrt(np.maximum(1.0 - sim, 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return pd.DataFrame(dist, index=emb_df.index, columns=emb_df.index)
