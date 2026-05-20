"""Path signature embeddings for HRP.

The path signature of a multidimensional path is a sequence of iterated
integrals that uniquely characterises the path up to tree-like equivalences
(Chen 1957). Truncated to a finite level, it gives a fixed-dimensional
geometric embedding of the path's shape and order — capturing information
the bare return distribution loses.

For HRP, we treat each asset's rolling (time, log-price, [extra channels])
trajectory as a path, compute the level-L truncated signature, normalise,
and use cosine distance between assets' signatures as the dendrogram
distance.

Multi-channel paths (Phase 8k): the bare log-price path is information-
equivalent to the return series, so a signature distance built from it
alone cannot out-cluster the sample correlation it is derived from — the
single-channel sweeps confirmed this (0/108 cells beat HRP). Appending
orthogonal channels the return series does NOT contain — quote volume,
trade count — lets the signature capture genuine cross-channel structure
(e.g. price/volume lead-lag via the level-2 mixed term) that correlation
cannot see.

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
    extra_channels: list[pd.DataFrame] | None = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Compute per-asset path signatures over a rolling window of returns.

    For each asset, we form a path by accumulating its returns into a
    log-price series, take the last `window` observations, optionally
    append extra channels, normalise, and compute the level-`level`
    signature via `esig.stream2sig`.

    Args:
        returns: T x N returns DataFrame.
        window: rolling-window length (default 60 days).
        level: signature truncation level (default 3; signature dimension
            grows as (d^(L+1)-1)/(d-1) for a d-channel path, so multi-channel
            paths produce much longer signatures).
        extra_channels: optional list of T x N panels (e.g. quote volume,
            trade count) appended as additional path channels. Aligned to
            `returns` by label; missing values are filled with the channel
            mean (neutral after the per-column z-score in `_normalise_path`).
        log_transform: if True (default), extra channels are log1p-transformed
            before stacking — correct for raw non-negative heavy-tailed
            quantities, whose raw z-scoring would be spike-dominated. Pass
            False when the channels are already pre-transformed into tame
            form (see `features.derive_channel_panels`).

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

    # Pre-slice each extra channel to the same window.
    chan_tails: list[pd.DataFrame] = []
    for ch in extra_channels or []:
        ch_aligned = ch.reindex(index=tail.index, columns=returns.columns)
        if log_transform:  # raw extra channels: tame the heavy tail
            ch_aligned = np.log1p(ch_aligned.clip(lower=0.0))
        # missing → channel mean (neutral after _normalise_path's z-score),
        # not 0 which would be a spurious outlier for an off-centre channel
        ch_aligned = ch_aligned.fillna(ch_aligned.mean())
        chan_tails.append(ch_aligned)

    rows = {}
    for asset in returns.columns:
        path = log_prices[[asset]].values
        for ch in chan_tails:
            path = np.hstack([path, ch[[asset]].values])
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
