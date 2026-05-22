"""Tests for the final PIT universe builder and artifact."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from scripts.build_pit_universe_final import (
    BuildConfig,
    DEFAULT_EXCLUDED_SYMBOLS,
    build_pit_universe,
    compare_against_committed,
    load_all_prices,
    load_binance_listings,
    summarize_pit_universe,
    to_monthly_snapshots,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@lru_cache(maxsize=1)
def _committed_pit() -> pd.DataFrame:
    return pd.read_csv(DATA / "pit_universe.csv", parse_dates=["date"])


@lru_cache(maxsize=1)
def _listings() -> dict[str, dict]:
    return load_binance_listings(DATA / "binance_listings_manual.json")


@lru_cache(maxsize=1)
def _rebuilt_pit() -> pd.DataFrame:
    prices = load_all_prices()
    monthly = to_monthly_snapshots(prices)
    return build_pit_universe(monthly, _listings(), BuildConfig())


class TestCommittedPITArtifact:
    def test_committed_artifact_has_expected_expanded_counts(self):
        pit = _committed_pit()
        included = pit[pit["included"]]
        last_date = included["date"].max()
        at_end = included.loc[included["date"] == last_date, "symbol"].nunique()
        ever_included = included["symbol"].nunique()
        dropped = ever_included - at_end

        assert pit["date"].nunique() == 100
        assert ever_included == 146
        assert at_end == 49
        assert dropped == 97

    def test_excluded_symbol_set_never_appears_included(self):
        pit = _committed_pit()
        included_bare = set(
            pit.loc[pit["included"], "symbol"].str.replace("USDT", "", regex=False)
        )
        assert included_bare.isdisjoint(DEFAULT_EXCLUDED_SYMBOLS)


class TestManualListingConstraints:
    def test_no_inclusion_before_manual_listed_at(self):
        pit = _committed_pit()
        listings = _listings()
        included = pit.loc[pit["included"]].copy()
        included["bare_symbol"] = included["symbol"].str.replace("USDT", "", regex=False)

        violations = []
        for _, row in included.iterrows():
            meta = listings.get(row["bare_symbol"])
            if not meta or not meta.get("listed_at"):
                continue
            listed_at = pd.Timestamp(meta["listed_at"])
            if row["date"] < listed_at:
                violations.append((row["bare_symbol"], row["date"], listed_at))

        assert not violations

    def test_no_inclusion_after_manual_delisted_at(self):
        pit = _committed_pit()
        listings = _listings()
        included = pit.loc[pit["included"]].copy()
        included["bare_symbol"] = included["symbol"].str.replace("USDT", "", regex=False)

        violations = []
        for _, row in included.iterrows():
            meta = listings.get(row["bare_symbol"])
            if not meta or not meta.get("delisted_at"):
                continue
            delisted_at = pd.Timestamp(meta["delisted_at"])
            if row["date"] > delisted_at:
                violations.append((row["bare_symbol"], row["date"], delisted_at))

        assert not violations


class TestFinalBuilder:
    def test_final_builder_reproduces_committed_artifact(self):
        rebuilt = _rebuilt_pit()
        committed = _committed_pit()
        comparison = compare_against_committed(rebuilt, committed)

        assert comparison["same_shape"] is True
        assert comparison["left_only_rows"] == 0
        assert comparison["right_only_rows"] == 0
        assert comparison["included_diff_rows"] == 0
        assert comparison["rqv_diff_rows"] == 0

    def test_summary_matches_committed_steady_state(self):
        summary = summarize_pit_universe(_rebuilt_pit())

        assert len(summary) == 100
        assert int(summary["n_included"].max()) == 51
        assert int(summary["n_included"].iloc[-1]) == 49


class TestSyntheticEntryExitBuffers:
    def test_entry_and_exit_buffers_apply_as_expected(self):
        dates = pd.to_datetime(
            [
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
                "2020-04-30",
                "2020-05-31",
            ]
        )
        monthly = pd.DataFrame(
            [
                {"snapshot_date": dates[0], "symbol": "AAAUSDT", "rolling_quote_vol_30d": 2_000_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[0], "symbol": "BBBUSDT", "rolling_quote_vol_30d": 1_500_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[1], "symbol": "AAAUSDT", "rolling_quote_vol_30d": 2_100_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[1], "symbol": "BBBUSDT", "rolling_quote_vol_30d": 1_400_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[2], "symbol": "AAAUSDT", "rolling_quote_vol_30d": 2_200_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[2], "symbol": "CCCUSDT", "rolling_quote_vol_30d": 2_000_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[3], "symbol": "AAAUSDT", "rolling_quote_vol_30d": 1_900_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[3], "symbol": "CCCUSDT", "rolling_quote_vol_30d": 2_100_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[4], "symbol": "AAAUSDT", "rolling_quote_vol_30d": 1_800_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
                {"snapshot_date": dates[4], "symbol": "CCCUSDT", "rolling_quote_vol_30d": 2_200_000.0, "first_seen": pd.Timestamp("2019-01-01"), "close": 1.0},
            ]
        )

        pit = build_pit_universe(
            monthly,
            listings={},
            cfg=BuildConfig(top_n=1, min_age_days=180, min_quote_volume_usd=1_000_000.0, entry_buffer_months=2, exit_buffer_months=1),
        )

        aaa = pit.loc[pit["symbol"] == "AAAUSDT", ["date", "included"]].reset_index(drop=True)
        ccc = pit.loc[pit["symbol"] == "CCCUSDT", ["date", "included"]].reset_index(drop=True)

        # AAA enters only after appearing in the top-1 candidate set twice.
        assert bool(aaa.loc[0, "included"]) is False
        assert bool(aaa.loc[1, "included"]) is True
        # AAA remains included for one extra month after dropping out.
        assert bool(aaa.loc[2, "included"]) is True
        assert bool(aaa.loc[3, "included"]) is True
        assert bool(aaa.loc[4, "included"]) is False

        # CCC also requires two consecutive candidate months to enter.
        assert bool(ccc.loc[0, "included"]) is False
        assert bool(ccc.loc[1, "included"]) is False
        assert bool(ccc.loc[2, "included"]) is True
