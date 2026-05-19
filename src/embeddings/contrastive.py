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

# Force single-threaded PyTorch ops — avoids segfaults observed under
# pytest on macOS where PyTorch's threadpool interacts badly with the
# test-runner's signal handlers.
torch.set_num_threads(1)


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

class TinyConvEncoder(nn.Module):
    """1D CNN with 3 conv layers + global average pooling.

    Input shape: (B, 1, window_len). Output shape: (B, emb_dim).
    """

    def __init__(self, emb_dim: int = 32, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
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

    Each item yields (anchor, positive) — two augmented views of the
    same window. Augmentation = Gaussian noise + random scaling.
    """

    def __init__(self, returns: pd.DataFrame, window: int, stride: int = 5,
                 noise_std: float = 0.02):
        self.window = window
        self.noise_std = noise_std
        self.windows: list[tuple[str, np.ndarray]] = []
        for asset in returns.columns:
            r = returns[asset].values.astype(np.float32)
            for start in range(0, len(r) - window + 1, stride):
                self.windows.append((asset, r[start:start + window]))

    def __len__(self):
        return len(self.windows)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x2 = x + np.random.randn(len(x)).astype(np.float32) * self.noise_std
        scale = 1.0 + np.random.randn() * 0.05
        return x2 * scale

    def __getitem__(self, idx: int):
        _, x = self.windows[idx]
        a = self._augment(x)
        p = self._augment(x)
        return (
            torch.from_numpy(a).unsqueeze(0),  # (1, W)
            torch.from_numpy(p).unsqueeze(0),
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
) -> tuple[TinyConvEncoder, WindowDataset]:
    """Train the contrastive encoder on a returns DataFrame.

    Returns the trained encoder and the dataset (needed for inference).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = WindowDataset(returns, window=window, stride=stride)
    if len(ds) < 4:
        raise RuntimeError(
            f"too few windows ({len(ds)}); reduce window or stride"
        )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    encoder = TinyConvEncoder(emb_dim=emb_dim, hidden=hidden).to(device)
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
) -> pd.DataFrame:
    """Compute per-asset embeddings as the mean encoder output over windows.

    For each asset, slice windows (no augmentation), encode each, then
    take the mean of the embeddings as the asset representation.
    """
    encoder.eval()
    asset_emb: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for asset in returns.columns:
            r = returns[asset].values.astype(np.float32)
            if len(r) < window:
                continue
            slices = []
            for start in range(0, len(r) - window + 1, stride):
                slices.append(r[start:start + window])
            X = np.stack(slices)
            tens = torch.from_numpy(X).unsqueeze(1).to(device)  # (M, 1, W)
            emb = encoder(tens).cpu().numpy()                   # (M, D)
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
