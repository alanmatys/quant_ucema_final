"""Path signature embeddings for HRP.

The path signature of a multidimensional path is a sequence of iterated
integrals that uniquely characterises the path up to tree-like equivalences
(Chen 1957). Truncated to a finite level, it gives a fixed-dimensional
geometric embedding of the path's shape and order — capturing information
the bare return distribution loses.

For HRP, we treat each asset's rolling (price, volume, time) trajectory as
a path, compute the level-3 truncated signature, normalise, and use cosine
distance between assets' signatures as the dendrogram distance.

Reference: Lyons (1998), Chen (1957). Applied to crypto by Lyons & Akyildirim
(2024) for clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from esig import stream2sig


def _normalise_path(prices: np.ndarray) -> np.ndarray:
    """Z-score normalise + add monotonically-increasing time channel.

    A time channel is essential for path signatures — without it, paths
    that visit the same points in different orders are indistinguishable.

    Args:
        prices: T x C array (T time steps, C feature channels).

    Returns:
        T x (C+1) array with prepended time channel in [0, 1].
    """
    T = prices.shape[0]
    time = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    if prices.std() > 0:
        z = (prices - prices.mean(axis=0, keepdims=True)) / (
            prices.std(axis=0, keepdims=True) + 1e-12
        )
    else:
        z = prices - prices.mean(axis=0, keepdims=True)
    return np.hstack([time, z])


def asset_path_signatures(
    returns: pd.DataFrame,
    window: int = 60,
    level: int = 3,
    use_volume: bool = False,
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-asset path signatures over a rolling window of returns.

    For each asset, we form a path by accumulating its returns into a
    log-price series, take the last `window` observations, normalise, and
    compute the level-`level` signature via `esig.stream2sig`.

    Args:
        returns: T x N returns DataFrame.
        window: rolling-window length (default 60 days).
        level: signature truncation level (default 3; signature dimension
            grows roughly as 2^level for univariate paths, faster for multi).
        use_volume: if True, include a volume channel (requires `volume`).
        volume: optional T x N volume DataFrame aligned with `returns`.

    Returns:
        DataFrame indexed by asset name, columns = signature components.
    """
    if returns.shape[0] < window:
        window = max(20, returns.shape[0])
    tail = returns.iloc[-window:]
    # Clip returns to (-0.99, +∞) so log1p stays finite; daily crypto rarely
    # produces realistic returns below -99% but synthetic test data can.
    clipped = tail.clip(lower=-0.99)
    log_prices = np.log1p(clipped).cumsum()

    rows = {}
    for asset in returns.columns:
        path = log_prices[[asset]].values
        if use_volume and volume is not None and asset in volume.columns:
            v = volume[asset].iloc[-window:].values.reshape(-1, 1)
            path = np.hstack([path, v])
        norm_path = _normalise_path(path)
        try:
            sig = stream2sig(norm_path, level)
            rows[asset] = sig
        except Exception:
            rows[asset] = None

    # Filter assets where signature computation failed
    valid = {k: v for k, v in rows.items() if v is not None}
    if not valid:
        raise RuntimeError("path-signature computation failed for all assets")

    # Stack into a DataFrame
    return pd.DataFrame(np.vstack(list(valid.values())), index=list(valid.keys()))


def signatures_to_distance(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-asset signatures to a pairwise distance matrix.

    Uses cosine distance on L2-normalised signature vectors:
        d_ij = sqrt(1 - <s_i, s_j>)  ∈ [0, sqrt(2)]
    Diagonal set to 0.

    Args:
        sig_df: N x D DataFrame of signatures (N assets, D dimensions).

    Returns:
        N x N distance DataFrame.
    """
    X = sig_df.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1e-12, norms)
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    # Convert similarity to distance in [0, sqrt(2)]
    dist = np.sqrt(np.maximum(1.0 - sim, 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return pd.DataFrame(dist, index=sig_df.index, columns=sig_df.index)
