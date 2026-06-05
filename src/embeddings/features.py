"""Microstructure channel panels for multi-channel embeddings.

The single-channel embeddings (returns only) are information-equivalent to
the sample correlation and cannot out-cluster it. Multi-channel embeddings
help only if the extra channels carry information the close-to-close return
series literally cannot reconstruct — squared/absolute returns or EWMA vol
are just transforms of returns and add nothing.

This module derives such genuinely-orthogonal channels from the long-format
Binance OHLCV table. Each panel is returned in a pre-transformed, roughly-
symmetric, tame-tailed form ready for z-scoring — so the channel builders
(`channels.build_asset_channels`, `path_signatures.asset_path_signatures`)
must be called with `log_transform=False`: feature engineering lives here,
not buried in a generic builder.

Channels (all orthogonal to the close-to-close return series):
    log_qvol         log(quote_volume)              — liquidity / interest
    log_ntrades      log(num_trades)                — trade activity
    intraday_range   log1p((high-low)/close)        — realized intraday vol
    taker_buy_ratio  taker_quote_vol/quote_volume   — order-flow imbalance
    log_amihud       log(|ret|/quote_volume)        — Amihud illiquidity
    log_trade_size   log(quote_volume/num_trades)   — whale vs retail texture
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TINY = 1e-15


def derive_channel_panels(prices_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive microstructure channel panels from long-format Binance OHLCV.

    Args:
        prices_long: long DataFrame with columns open_time, symbol, high,
            low, close, quote_volume, num_trades, taker_quote_vol.

    Returns:
        dict {channel_name: wide DataFrame (date x symbol)}, each panel
        pre-transformed and ready to pass to a channel builder with
        log_transform=False.
    """
    df = prices_long.copy()
    for c in ("high", "low", "close", "quote_volume", "num_trades",
              "taker_quote_vol"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _pivot(col: str) -> pd.DataFrame:
        return df.pivot(index="open_time", columns="symbol",
                        values=col).sort_index()

    high    = _pivot("high")
    low     = _pivot("low")
    close   = _pivot("close")
    qvol    = _pivot("quote_volume")
    ntrades = _pivot("num_trades")
    taker_q = _pivot("taker_quote_vol")

    qvol_safe = qvol.clip(lower=_TINY)
    ret = close.pct_change()

    panels = {
        # log of heavy-tailed positive levels
        "log_qvol":       np.log(qvol_safe),
        "log_ntrades":    np.log(ntrades.clip(lower=1.0)),
        "log_trade_size": np.log((qvol / ntrades.clip(lower=1.0)).clip(lower=_TINY)),
        # realized intraday volatility — a day can net 0 return with a huge range
        "intraday_range": np.log1p(((high - low) / close).clip(lower=0.0)),
        # order-flow imbalance — fraction of volume that was aggressive buying
        "taker_buy_ratio": (taker_q / qvol_safe).clip(0.0, 1.0),
        # Amihud illiquidity — price impact per dollar traded
        "log_amihud":     np.log((ret.abs() / qvol_safe).clip(lower=_TINY)),
    }
    return panels
