# Spec 08 — Point-in-Time Universe Reconstruction

**Status:** Current contract for the final committed PIT artifact

## 1. Scope

This spec defines the point-in-time universe used by the final committed
backtest artifacts and the UCEMA journal paper.

It supersedes earlier top-30 / 2019-2024 drafts. Historical notes about
CoinGecko-based or smaller-universe variants are no longer the active
contract.

## 2. Current Artifact State

The committed artifact `data/pit_universe.csv` reflects:

- Snapshot range: `2018-02-28` to `2026-05-31`
- `100` monthly snapshots
- `146` ever-included symbols
- `97` symbols dropped during the window
- steady-state included universe near `49-51` names

This is the authoritative PIT universe for the current paper draft.

## 3. Method

### 3.1 Ranking signal

Universe membership is based on **rolling 30-day Binance USDT quote
volume** as a tradable-liquidity proxy.

The active implementation computes this signal as a **rolling mean** of
daily quote volume per symbol before month-end snapshotting.

### 3.2 Selection rules

At each monthly snapshot:

1. Exclude stablecoins, wrappers/liquid-staking derivatives, leveraged
   tokens, and venue-native exchange tokens.
2. Require at least `180` days of observed history.
3. Require at least `USD 1M` on the rolling 30-day quote-volume signal.
4. Apply manual listing/delisting constraints from
   `data/binance_listings_manual.json` when coverage exists.
5. Rank eligible symbols by rolling 30-day quote volume.
6. Take the top `50` candidates.
7. Apply entry/exit buffers:
   - enter after `2` consecutive candidate months,
   - exit after `1` additional month outside the candidate set.

## 4. Manual Listings File

`data/binance_listings_manual.json` is **sparse by design**.

It covers symbols where listing or delisting timing is expected to matter
materially for the PIT universe. Symbols not included in that file are
treated as tradable throughout the relevant historical window unless the
price dataset itself implies otherwise.

This is an explicit simplifying assumption and should be acknowledged in
paper-facing methodology text.

## 5. Builder Status

The authoritative final builder is:

- `scripts/build_pit_universe_final.py`

The committed notebook `notebooks/build_pit_universe.ipynb` contains an
older top-30 / 2019-2024 build path and should be treated as historical
context unless updated to wrap the final builder logic.

## 6. Active Interface

The active implementation lives in `src/universe.py` and exposes:

```python
def load_candidates(...)
def load_binance_listings(...)
def fetch_binance_extended_prices(...)
def to_monthly_snapshots(...)
def build_pit_universe(...)
def load_pit_universe(...)
def summarize_pit_universe(...)
```

## 7. Acceptance Criteria

AC1. `data/pit_universe.csv` remains the authoritative committed PIT
artifact for the paper unless explicitly regenerated and versioned.

AC2. No included row may occur before `listed_at` for symbols covered by
`data/binance_listings_manual.json`.

AC3. No included row may occur after `delisted_at` for symbols covered by
`data/binance_listings_manual.json`.

AC4. The spec, implementation comments, and paper-facing universe
description all refer to the same top-50 expanded PIT design.

AC5. `scripts/build_pit_universe_final.py` reproduces the committed
artifact or fails loudly on mismatch.
