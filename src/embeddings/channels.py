"""Shared multi-channel feature preparation for the embedding encoders.

The torch encoders (`contrastive.py`, `ts2vec_lite.py`) are 1D-CNNs. To feed
several features per asset — returns plus orthogonal channels such as quote
volume and trade count — every channel must be on a comparable scale: the
return series is O(1e-2) while quote volume is O(1e8), so an unscaled
multi-channel input would let volume dominate the very first convolution.

We therefore standardise every channel per-asset before windowing. The
return channel is z-scored directly; non-negative heavy-tailed channels
(volume, trade count) are log1p-transformed first, then z-scored.

This mirrors the rationale in `path_signatures.py`: a returns-only encoder
is information-equivalent to the sample correlation, so multi-channel input
is the lever that lets an embedding capture structure correlation cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score a 1D array; return all-zeros if degenerate (zero/NaN std)."""
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def build_asset_channels(
    returns: pd.DataFrame,
    extra_channels: list[pd.DataFrame] | None = None,
    log_transform: bool = True,
) -> dict[str, np.ndarray]:
    """Build per-asset (C, T) standardised channel arrays.

    Channel 0 is the return series; channels 1.. are the extra panels.
    Every channel is z-scored over the asset's own history.

    Args:
        returns: T x N returns DataFrame.
        extra_channels: optional list of T x N panels (quote volume, trade
            count, ...). Aligned to `returns` by label; assets missing from
            a panel become a flat (all-zero) channel for that asset.
        log_transform: if True (default), extra channels are log1p-transformed
            before z-scoring — correct for raw non-negative heavy-tailed
            quantities (volume, trade count). Pass False when the extra
            channels are already pre-transformed into tame form (see
            `features.derive_channel_panels`, whose `log_amihud` channel is
            negative and must NOT be re-logged).

    Returns:
        dict {asset: float32 array of shape (n_channels, T)}.
    """
    panels = [returns] + list(extra_channels or [])
    out: dict[str, np.ndarray] = {}
    for asset in returns.columns:
        chans = []
        for k, panel in enumerate(panels):
            if asset not in panel.columns:
                chans.append(np.zeros(len(returns), dtype=np.float32))
                continue
            v = panel[asset].reindex(returns.index).values.astype(np.float64)
            if k > 0 and log_transform:  # raw extra channels: tame the tail
                v = np.log1p(np.clip(v, 0.0, None))
            # Fill missing with the channel mean → neutral (0) after z-scoring,
            # rather than 0 (which would be a spurious outlier for a channel
            # not centred at 0, e.g. the negative log_amihud channel).
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(v)
            if not np.isfinite(mu):
                mu = 0.0
            v = np.where(np.isfinite(v), v, mu)
            chans.append(_zscore(v))
        out[asset] = np.vstack(chans)  # (C, T)
    return out
