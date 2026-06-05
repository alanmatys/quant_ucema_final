"""Sanity tests for the walk-forward backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    WalkForwardBacktest, CostModel, COST_SCENARIOS,
    monthly_rebalance_dates, summarize_performance,
)
from src.portfolio_maker import HRP, IVP


def _make_toy_dataset(seed: int = 0, n_assets: int = 8, n_days: int = 800):
    """Synthetic prices + a trivial PIT universe that includes all assets at all snapshots."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    daily = rng.standard_normal((n_days, n_assets)) * 0.02
    prices = pd.DataFrame(
        np.cumprod(1 + daily, axis=0) * 100,
        index=dates,
        columns=[f"A{i:02d}" for i in range(n_assets)],
    )
    snap_dates = pd.date_range(dates[0], dates[-1], freq="ME")
    pit_rows = []
    for snap in snap_dates:
        for col in prices.columns:
            pit_rows.append({"date": snap, "symbol": col, "included": True})
    pit = pd.DataFrame(pit_rows)
    return prices, pit


class TestCostModel:
    def test_zero_cost(self):
        m = COST_SCENARIOS["zero"]
        assert m.cost_pct(turnover=1.0) == 0.0

    def test_conservative_cost(self):
        m = COST_SCENARIOS["conservative_cex"]
        # 12 bps per 100% turnover
        c = m.cost_pct(turnover=1.0)
        assert c == pytest.approx(0.0012, abs=1e-6)

    def test_liquidation_premium(self):
        m = CostModel(fee_bps=10, slippage_bps=0, liquidation_premium=2.0)
        c_normal = m.cost_pct(turnover=1.0, liquidation_share=0.0)
        c_with_liquid = m.cost_pct(turnover=1.0, liquidation_share=0.5)
        # 50% of turnover at 2x cost
        assert c_with_liquid > c_normal


class TestMonthlyRebalanceDates:
    def test_picks_month_ends(self):
        dates = monthly_rebalance_dates(
            pd.Timestamp("2023-01-15"), pd.Timestamp("2023-06-15")
        )
        # Jan 31, Feb 28, Mar 31, Apr 30, May 31 (Jun 30 is past end)
        assert len(dates) == 5
        assert all(d.is_month_end for d in dates)


class TestWalkForwardEngine:
    def setup_method(self):
        self.prices, self.pit = _make_toy_dataset(seed=42, n_assets=8, n_days=800)

    def test_runs_without_crashing(self):
        bt = WalkForwardBacktest(
            prices=self.prices,
            pit_universe=self.pit,
            strategy_factory=lambda r: IVP(r),
            cost_model=COST_SCENARIOS["conservative_cex"],
        )
        result = bt.run(start="2020-12-31", end="2022-06-30")
        assert not result["daily_returns"].empty
        assert not result["weights_history"].empty

    def test_weights_sum_to_one_at_every_snapshot(self):
        bt = WalkForwardBacktest(
            prices=self.prices,
            pit_universe=self.pit,
            strategy_factory=lambda r: IVP(r),
        )
        result = bt.run(start="2020-12-31", end="2022-06-30")
        # Each row of weights_history should sum to 1 (within tolerance)
        sums = result["weights_history"].sum(axis=1)
        assert (sums - 1.0).abs().max() < 1e-6

    def test_no_lookahead_estimation_window(self):
        """At rebalance date T, estimation window uses prices up to and including T,
        but the OOS returns start AFTER T (next-day returns only)."""
        bt = WalkForwardBacktest(
            prices=self.prices,
            pit_universe=self.pit,
            strategy_factory=lambda r: IVP(r),
        )
        result = bt.run(start="2020-12-31", end="2022-06-30")
        # First daily return must come AFTER the first rebalance date
        first_snap = result["snapshot_dates"][0]
        first_return_date = result["daily_returns"].index[0]
        assert first_return_date > first_snap

    def test_turnover_first_rebalance_is_one(self):
        """Going from no position to fully-invested is L1 turnover = 1.0."""
        bt = WalkForwardBacktest(
            prices=self.prices, pit_universe=self.pit,
            strategy_factory=lambda r: IVP(r),
        )
        result = bt.run(start="2020-12-31", end="2022-06-30")
        first = result["turnover_history"].iloc[0]
        assert first == pytest.approx(1.0)

    def test_costs_reduce_net_returns_vs_gross(self):
        bt = WalkForwardBacktest(
            prices=self.prices, pit_universe=self.pit,
            strategy_factory=lambda r: IVP(r),
            cost_model=COST_SCENARIOS["conservative_cex"],
        )
        result = bt.run(start="2020-12-31", end="2022-06-30")
        net = result["daily_returns"].sum()
        gross = result["gross_daily_returns"].sum()
        assert net < gross, f"net {net:.4f} should be < gross {gross:.4f}"


class TestSummarizePerformance:
    def test_iid_zero_mean_has_low_sharpe(self):
        # Average across multiple seeds — single seeds can hit |Sharpe| > 0.5 by chance
        sharpes = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            r = pd.Series(rng.standard_normal(500) * 0.01)
            sharpes.append(summarize_performance(r)["sharpe"])
        # Mean |Sharpe| should be small for zero-drift iid noise
        assert abs(np.mean(sharpes)) < 0.3

    def test_positive_drift_gives_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.standard_normal(500) * 0.01 + 0.0008)
        metrics = summarize_performance(r)
        assert metrics["sharpe"] > 0.5
