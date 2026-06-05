"""Point-in-time universe reconstruction (Binance liquidity-based).

Builds a monthly point-in-time universe of top-N crypto USDT pairs ranked by
rolling 30-day Binance quote volume, including pairs later delisted, to remove
survivorship bias from the backtest. See specs/08_universe.md for the full spec
and the methodology note explaining why we use Binance quote volume instead of
CoinGecko market cap (free-tier history cap).

Workflow:
    1. From the curated candidate list, identify which USDT pairs need fetching
       from Binance vs. which are already in the existing CSV.
    2. Fetch missing pairs (including delisted ones such as LUNAUSDT, FTTUSDT)
       from the Binance public klines endpoint.
    3. Compute rolling 30-day quote_volume per symbol and aggregate to monthly
       snapshots.
    4. Apply selection rules (top-N, age, volume floor, exclusions, Binance
       listing dates) with entry/exit buffers.
    5. Emit data/pit_universe.csv plus a human-readable summary.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.binance_data import get_historical_klines


# Stablecoins / wrappers / leveraged tokens excluded from the risky-asset universe.
# Matches the exclusion list referenced in Spec 08 R2.
DEFAULT_EXCLUDED_SYMBOLS: set[str] = {
    # Fiat-backed stablecoins
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FRAX",
    "USDD", "FDUSD", "PYUSD", "USDE",
    # Wrappers / liquid-staking derivatives
    "WBTC", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WBETH",
    # Leveraged / inverse tokens
    "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
    # Exchange tokens of the execution venue (avoid endogeneity per Spec 08 R2)
    "BNB",
}


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader — sets env vars from KEY=VALUE lines."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_candidates(path: str | Path = "data/coingecko_candidates.json") -> list[dict]:
    """Load curated candidate coin list (returns list of {symbol, coingecko_id, category} dicts)."""
    return json.loads(Path(path).read_text())["candidates"]


def load_binance_listings(path: str | Path = "data/binance_listings_manual.json") -> dict[str, dict]:
    """Load hand-curated Binance USDT listing/delisting dates.

    Returns:
        {symbol: {'listed_at': 'YYYY-MM-DD', 'delisted_at': 'YYYY-MM-DD' | None, 'notes': str}}
    """
    payload = json.loads(Path(path).read_text())
    return {row["symbol"]: row for row in payload["listings"]}


def fetch_binance_extended_prices(
    symbols: Iterable[str],
    start: str,
    end: str,
    cache_path: str | Path,
    delay: float = 0.5,
    additional_caches: list[str | Path] | None = None,
) -> pd.DataFrame:
    """Fetch daily klines from Binance with window-aware caching.

    For each requested symbol:
    - If not in cache, fetches the full [start, end] range.
    - If cached but the cached window doesn't cover [start, end], fetches
      only the missing prefix (start → first_cached - 1 day) and suffix
      (last_cached + 1 day → end), then appends to cache.

    Args:
        symbols: Bare symbol tickers (e.g. ["BTC", "ETH", "LUNA"]); "USDT" is appended.
        start: ISO date string (inclusive) e.g. "2017-01-01".
        end:   ISO date string (inclusive) e.g. "2026-05-18".
        cache_path: CSV path to read existing data from and append to.
        delay: Seconds to pause between symbol fetches.
        additional_caches: Optional list of read-only cache files to consult
            when computing coverage (e.g. the original repo CSV); their data
            is NOT modified, but they're used to decide what's already covered.

    Returns:
        Long-format DataFrame with the columns of the existing repo CSV plus
        a 'symbol' column. Includes all rows for the requested symbols in [start, end]
        from BOTH the primary cache and additional_caches.
    """
    cache_path = Path(cache_path)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Read all caches to determine current coverage per symbol
    cached_frames: list[pd.DataFrame] = []
    primary_cached: pd.DataFrame | None = None
    if cache_path.exists():
        primary_cached = pd.read_csv(cache_path, parse_dates=["open_time"])
        cached_frames.append(primary_cached)
    if additional_caches:
        for p in additional_caches:
            p = Path(p)
            if p.exists():
                cached_frames.append(pd.read_csv(p, parse_dates=["open_time"]))

    all_cached = (
        pd.concat(cached_frames, ignore_index=True).drop_duplicates(
            subset=["symbol", "open_time"]
        )
        if cached_frames
        else pd.DataFrame(columns=["symbol", "open_time"])
    )

    coverage: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if not all_cached.empty:
        for pair, group in all_cached.groupby("symbol"):
            coverage[pair] = (group["open_time"].min(), group["open_time"].max())

    new_rows: list[pd.DataFrame] = []
    symbols = list(symbols)

    for i, sym in enumerate(symbols):
        pair = f"{sym}USDT"
        if pair in coverage:
            cov_min, cov_max = coverage[pair]
            ranges_to_fetch: list[tuple[str, str]] = []
            # Missing prefix?
            if start_ts < cov_min:
                ranges_to_fetch.append((start, (cov_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
            # Missing suffix?
            if end_ts > cov_max:
                ranges_to_fetch.append(((cov_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end))
            if not ranges_to_fetch:
                continue  # fully covered
            label = "extend"
        else:
            ranges_to_fetch = [(start, end)]
            label = "new"

        for sub_start, sub_end in ranges_to_fetch:
            try:
                df = get_historical_klines(
                    symbol=pair, interval="1d", start_str=sub_start, end_str=sub_end
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i+1}/{len(symbols)}] FAILED {pair} {sub_start}->{sub_end}: {exc}")
                time.sleep(delay)
                continue

            if df is None or df.empty:
                print(f"  [{i+1}/{len(symbols)}] no data {pair} {sub_start}->{sub_end}")
                time.sleep(delay)
                continue

            df = df.reset_index()
            df["symbol"] = pair
            new_rows.append(df)
            print(
                f"  [{i+1}/{len(symbols)}] {label} {pair} {sub_start}->{sub_end}: "
                f"{len(df)} rows ({df['open_time'].min().date()} -> {df['open_time'].max().date()})"
            )
            time.sleep(delay)

    if new_rows:
        new = pd.concat(new_rows, ignore_index=True)
        combined_primary = (
            pd.concat([primary_cached, new], ignore_index=True)
            if primary_cached is not None
            else new
        )
        combined_primary = combined_primary.drop_duplicates(subset=["symbol", "open_time"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined_primary.to_csv(cache_path, index=False)
    else:
        combined_primary = primary_cached

    # Build the return frame from all caches (primary + additional) + any new
    pieces = [c for c in cached_frames if c is not None]
    if new_rows:
        pieces.append(pd.concat(new_rows, ignore_index=True))
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates(
        subset=["symbol", "open_time"]
    )

    wanted = {f"{s}USDT" for s in symbols}
    mask = (
        combined["symbol"].isin(wanted)
        & (combined["open_time"] >= start_ts)
        & (combined["open_time"] <= end_ts)
    )
    return combined.loc[mask].reset_index(drop=True)


def to_monthly_snapshots(prices: pd.DataFrame) -> pd.DataFrame:
    """Reduce daily Binance price/volume frame to month-end snapshots.

    Args:
        prices: long DataFrame with [open_time, symbol, close, quote_volume, ...]

    Returns:
        DataFrame with [snapshot_date, symbol, rolling_quote_vol_30d, first_seen, close].
        rolling_quote_vol_30d is the rolling mean of the prior 30 daily quote volumes.
    """
    df = prices[["open_time", "symbol", "close", "quote_volume"]].copy()
    df = df.sort_values(["symbol", "open_time"])
    # Use MEAN of 30 daily quote volumes (interpretable as "average daily
    # tradable USDT volume over the past month"). Previously this was
    # MEDIAN — robust to one-day spikes but doesn't match the spec's
    # "30-day quote volume" name. Caught in the 2026-05-19 code review.
    # Effective ranking change is small for stable tokens (median ≈ mean
    # for low vol-of-volume) but material for tokens with one-day spikes
    # (mean correctly rewards sustained volume; median over-weighted typical days).
    df["rolling_quote_vol_30d"] = (
        df.groupby("symbol")["quote_volume"]
        .transform(lambda s: s.rolling(window=30, min_periods=10).mean())
    )
    df["first_seen"] = df.groupby("symbol")["open_time"].transform("min")

    df["month_end"] = df["open_time"] + pd.offsets.MonthEnd(0)
    last_of_month = df.groupby(["symbol", "month_end"]).tail(1)

    out = last_of_month.rename(columns={"month_end": "snapshot_date"})[
        ["snapshot_date", "symbol", "rolling_quote_vol_30d", "first_seen", "close"]
    ]
    return out.sort_values(["snapshot_date", "rolling_quote_vol_30d"],
                           ascending=[True, False]).reset_index(drop=True)


def build_pit_universe(
    monthly: pd.DataFrame,
    top_n: int = 50,
    min_age_days: int = 180,
    min_median_volume_usd: float = 1_000_000.0,
    excluded_symbols: set[str] | None = None,
    entry_buffer_months: int = 2,
    exit_buffer_months: int = 1,
    binance_listings: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Build the monthly point-in-time universe with entry/exit buffers.

    At each snapshot date:
      1. Drop excluded symbols (stables, wrappers, exchange tokens).
      2. Apply age + rolling-30d-volume + listing/delisting window filters.
      3. Take top-N by rolling quote volume as the "candidate" set.
      4. Apply entry buffer (must be candidate for K consecutive months to enter)
         and exit buffer (stays in M months after dropping below).

    Args:
        monthly: DataFrame from `to_monthly_snapshots` — uses pair symbols (e.g. BTCUSDT).
        top_n: number of top-volume symbols per snapshot.
        min_age_days, min_median_volume_usd, excluded_symbols, *_buffer_months:
            `min_median_volume_usd` is a legacy argument name retained for
            compatibility; the current signal is the rolling 30-day quote-volume
            measure computed in `to_monthly_snapshots`.
        binance_listings: optional mapping from `load_binance_listings`; if a symbol
            has a listed_at later than snapshot_date or a delisted_at earlier than
            snapshot_date, it's filtered out.

    Returns:
        Long DataFrame: [date, symbol, rolling_quote_vol_30d, included].
    """
    excluded = excluded_symbols or DEFAULT_EXCLUDED_SYMBOLS
    df = monthly.copy()
    df["bare_symbol"] = df["symbol"].str.replace("USDT", "", regex=False)
    df["age_days"] = (df["snapshot_date"] - df["first_seen"]).dt.days

    # Hard filters
    eligible = df[~df["bare_symbol"].isin(excluded)].copy()
    eligible = eligible[eligible["age_days"] >= min_age_days]
    eligible = eligible[eligible["rolling_quote_vol_30d"].fillna(0) >= min_median_volume_usd]

    # Listing window filter
    if binance_listings:
        def listed_in_window(row: pd.Series) -> bool:
            meta = binance_listings.get(row["bare_symbol"])
            if not meta:
                return True
            listed_at = pd.Timestamp(meta["listed_at"]) if meta.get("listed_at") else None
            delisted_at = pd.Timestamp(meta["delisted_at"]) if meta.get("delisted_at") else None
            if listed_at is not None and row["snapshot_date"] < listed_at:
                return False
            if delisted_at is not None and row["snapshot_date"] > delisted_at:
                return False
            return True
        eligible = eligible[eligible.apply(listed_in_window, axis=1)]

    # Top-N candidate set per snapshot
    candidate_flags: dict[pd.Timestamp, set[str]] = {}
    for snap, group in eligible.groupby("snapshot_date"):
        top = group.nlargest(top_n, "rolling_quote_vol_30d")
        candidate_flags[snap] = set(top["symbol"])

    # Entry/exit buffers across snapshot timeline
    snapshots = sorted(candidate_flags.keys())
    all_symbols = sorted({s for snap_set in candidate_flags.values() for s in snap_set})
    included_state: dict[str, dict[pd.Timestamp, bool]] = {}

    for sym in all_symbols:
        candidate_streak = 0
        out_streak = 0
        currently_in = False
        included_state[sym] = {}
        for snap in snapshots:
            is_candidate = sym in candidate_flags[snap]
            if is_candidate:
                candidate_streak += 1
                out_streak = 0
                if not currently_in and candidate_streak >= entry_buffer_months:
                    currently_in = True
            else:
                candidate_streak = 0
                out_streak += 1
                if currently_in and out_streak > exit_buffer_months:
                    currently_in = False
            included_state[sym][snap] = currently_in

    # Build long output
    rows = []
    for snap in snapshots:
        snap_df = eligible[eligible["snapshot_date"] == snap]
        for _, row in snap_df.iterrows():
            sym = row["symbol"]
            rows.append({
                "date": snap,
                "symbol": sym,
                "rolling_quote_vol_30d": row["rolling_quote_vol_30d"],
                "included": included_state.get(sym, {}).get(snap, False),
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["date", "rolling_quote_vol_30d"],
                           ascending=[True, False]).reset_index(drop=True)


def load_pit_universe(path: str | Path = "data/pit_universe.csv") -> pd.DataFrame:
    """Load the built PIT universe artifact."""
    return pd.read_csv(path, parse_dates=["date"])


def summarize_pit_universe(pit: pd.DataFrame) -> pd.DataFrame:
    """Produce a human-readable per-snapshot summary.

    Returns one row per snapshot date with: n_included, n_eligible, included_symbols.
    """
    rows = []
    for snap, group in pit.groupby("date"):
        included = group[group["included"]].sort_values("rolling_quote_vol_30d", ascending=False)
        rows.append({
            "date": snap,
            "n_eligible": len(group),
            "n_included": len(included),
            "included_symbols": ",".join(included["symbol"].tolist()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
