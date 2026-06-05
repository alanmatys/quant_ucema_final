"""Build the final committed PIT universe from committed CSV inputs.

This script is the authoritative builder for the expanded top-50 point-in-
time universe used by the final committed paper artifacts.

Inputs:
- data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv
- data/binance_pit_supplement_2019-2024_1d.csv
- data/binance_listings_manual.json

Outputs:
- data/pit_universe.csv
- data/pit_universe_summary.csv

The build logic intentionally matches the current expanded artifact:
- rolling 30-day mean quote volume
- top 50 candidates per month
- age >= 180 days
- rolling quote-volume threshold >= 1,000,000
- entry buffer 2 months, exit buffer 1 month
- sparse manual listing/delisting constraints where coverage exists
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXISTING_CSV = DATA / "binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv"
SUPPLEMENT_CSV = DATA / "binance_pit_supplement_2019-2024_1d.csv"
LISTINGS_JSON = DATA / "binance_listings_manual.json"
PIT_OUT = DATA / "pit_universe.csv"
SUMMARY_OUT = DATA / "pit_universe_summary.csv"


DEFAULT_EXCLUDED_SYMBOLS: set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FRAX",
    "USDD", "FDUSD", "PYUSD", "USDE",
    "WBTC", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WBETH",
    "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
    "BNB",
}


@dataclass(frozen=True)
class BuildConfig:
    top_n: int = 50
    min_age_days: int = 180
    min_quote_volume_usd: float = 1_000_000.0
    entry_buffer_months: int = 2
    exit_buffer_months: int = 1


def load_binance_listings(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {row["symbol"]: row for row in payload["listings"]}


def load_all_prices() -> pd.DataFrame:
    ex = pd.read_csv(EXISTING_CSV, parse_dates=["open_time"])
    sp = pd.read_csv(SUPPLEMENT_CSV, parse_dates=["open_time"])
    return pd.concat([ex, sp], ignore_index=True).drop_duplicates(subset=["symbol", "open_time"])


def to_monthly_snapshots(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices[["open_time", "symbol", "close", "quote_volume"]].copy()
    df = df.sort_values(["symbol", "open_time"])
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
    return out.sort_values(["snapshot_date", "rolling_quote_vol_30d"], ascending=[True, False]).reset_index(drop=True)


def build_pit_universe(monthly: pd.DataFrame, listings: dict[str, dict], cfg: BuildConfig) -> pd.DataFrame:
    df = monthly.copy()
    df["bare_symbol"] = df["symbol"].str.replace("USDT", "", regex=False)
    df["age_days"] = (df["snapshot_date"] - df["first_seen"]).dt.days

    eligible = df[~df["bare_symbol"].isin(DEFAULT_EXCLUDED_SYMBOLS)].copy()
    eligible = eligible[eligible["age_days"] >= cfg.min_age_days]
    eligible = eligible[eligible["rolling_quote_vol_30d"].fillna(0) >= cfg.min_quote_volume_usd]

    def listed_in_window(row: pd.Series) -> bool:
        meta = listings.get(row["bare_symbol"])
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

    candidate_flags: dict[pd.Timestamp, set[str]] = {}
    for snap, group in eligible.groupby("snapshot_date"):
        top = group.nlargest(cfg.top_n, "rolling_quote_vol_30d")
        candidate_flags[snap] = set(top["symbol"])

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
                if not currently_in and candidate_streak >= cfg.entry_buffer_months:
                    currently_in = True
            else:
                candidate_streak = 0
                out_streak += 1
                if currently_in and out_streak > cfg.exit_buffer_months:
                    currently_in = False
            included_state[sym][snap] = currently_in

    rows = []
    for snap in snapshots:
        snap_df = eligible[eligible["snapshot_date"] == snap]
        for _, row in snap_df.iterrows():
            sym = row["symbol"]
            rows.append(
                {
                    "date": snap,
                    "symbol": sym,
                    "rolling_quote_vol_30d": row["rolling_quote_vol_30d"],
                    "included": included_state.get(sym, {}).get(snap, False),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["date", "rolling_quote_vol_30d"], ascending=[True, False]).reset_index(drop=True)


def summarize_pit_universe(pit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for snap, group in pit.groupby("date"):
        included = group[group["included"]].sort_values("rolling_quote_vol_30d", ascending=False)
        rows.append(
            {
                "date": snap,
                "n_eligible": len(group),
                "n_included": len(included),
                "included_symbols": ",".join(included["symbol"].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compare_against_committed(pit: pd.DataFrame, current: pd.DataFrame) -> dict[str, int | bool]:
    merged = pit.merge(current, on=["date", "symbol"], how="outer", suffixes=("_new", "_cur"), indicator=True)
    common = merged[merged["_merge"] == "both"]
    return {
        "same_shape": pit.shape == current.shape,
        "left_only_rows": int((merged["_merge"] == "left_only").sum()),
        "right_only_rows": int((merged["_merge"] == "right_only").sum()),
        "included_diff_rows": int((common["included_new"] != common["included_cur"]).sum()),
        "rqv_diff_rows": int(((common["rolling_quote_vol_30d_new"] - common["rolling_quote_vol_30d_cur"]).abs() > 1e-6).sum()),
    }


def main() -> None:
    cfg = BuildConfig()
    prices = load_all_prices()
    listings = load_binance_listings(LISTINGS_JSON)
    monthly = to_monthly_snapshots(prices)
    pit = build_pit_universe(monthly, listings, cfg)
    summary = summarize_pit_universe(pit)

    current = pd.read_csv(PIT_OUT, parse_dates=["date"]) if PIT_OUT.exists() else pit.copy()
    comp = compare_against_committed(pit, current)

    pit.to_csv(PIT_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    included = pit[pit["included"]]
    print("Wrote:")
    print(f"  {PIT_OUT}")
    print(f"  {SUMMARY_OUT}")
    print("Summary:")
    print(f"  snapshots={pit['date'].nunique()}")
    print(f"  ever_included={included['symbol'].nunique()}")
    print(f"  dropped_during_window={included['symbol'].nunique() - included[included['date'] == included['date'].max()]['symbol'].nunique()}")
    print("Comparison vs committed artifact:")
    for k, v in comp.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
