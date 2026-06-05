# Spec 05 — Repository Cleanup

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez

---

## 1. Motivation

The repo has accumulated branches, a stash, and a few duplicate paper folders during
the MACI 2025 work and the momentum-strategies extension. Before publishing the
continuation paper, the repo should be in a clean state so reviewers and future
collaborators can navigate it without ambiguity.

## 2. Requirements

R1. **Local branches:** prune merged feature branches once their changes are
    confirmed on `origin/main`:
    - `feature/improve-documentation`
    - `feature/momentum-strategies`
    - `notebook-execution`

R2. **Stash:** inspect `stash@{0}: On feature/improve-documentation: WIP: hrp.ipynb
    before switching to main` — either apply, save to a branch, or drop. No
    stashes should remain at the end.

R3. **Paper folders:** the `paper/` directory currently contains:
    - `paper/MACI_latex_eng/` — original MACI 2025 submission (keep as published).
    - `paper/latex/` — duplicate of MACI source.
    - `paper/latex 2/` — duplicate of MACI source.
    - `paper/latex copia/` — duplicate of MACI source.
    - `paper/latex (2).zip` — zipped duplicate.
    - Loose `.png` files at the `paper/` root used by both the MACI source and
      `paper/figures/` (after Spec 03).

    After cleanup:
    - Keep `paper/MACI_latex_eng/` intact.
    - Move the new continuation paper to `paper/ucema_journal/` (per Spec 04).
    - Remove or archive the three duplicate `latex*` folders and the zip.
    - Move the loose `.png` figures into `paper/figures/` (or `paper/MACI_latex_eng/`
      if they only belong to the MACI submission).

R4. **`.gitignore`:** ensure the following are ignored
    (audit-corrected after Phase 1 pivot dropped parquet and after the
    pre-existing blanket `*.csv` rule was found to hide commitable artifacts):
    - LaTeX build artifacts: `*.aux`, `*.log`, `*.synctex.gz`, `*.fdb_latexmk`,
      `*.fls`, `*.out`, `*.toc`, `*.bbl`, `*.blg`.
    - macOS metadata: `.DS_Store`, `__MACOSX/`.
    - Notebook checkpoints: `.ipynb_checkpoints/`.
    - Python cache: `__pycache__/`, `*.pyc`, `.pytest_cache/`.
    - Virtual env: `venv/`, `.venv/`.
    - Secrets: `.env`, `.env.local`, `*.key`.
    - Large reproducible caches only — NOT a blanket `*.csv` rule:
      - `data/coingecko_market_caps_*.csv` (Pro-tier robustness check, large)
      - `data/backtest_v2_weights_history.{parquet,csv}` (per-strategy weight panels)
      - `data/backtest_v2_returns_history.{parquet,csv}` (per-strategy return series)

R4b. **Files that MUST be committed despite previous blanket rules** (the
    audit found these were being silently ignored under the old `*.csv` and
    `paper/` rules):
    - All `data/*.json` (PIT inputs: `coingecko_candidates.json`,
      `binance_listings_manual.json`)
    - `data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv` (existing survivor data)
    - `data/binance_pit_supplement_2019-2024_1d.csv` (PIT supplementary data, Spec 08)
    - `data/pit_universe.csv` (Spec 08 output)
    - `data/pit_universe_summary.csv` (Spec 08 output)
    - `data/backtest_v2_static_results.csv` (Spec 03 output)
    - `data/backtest_v2_rebalance_results.csv` (Spec 03 output)
    - `data/backtest_v2_threshold_results.csv` (Spec 03 output, incl. Scenario C')
    - `data/cluster_stability.csv` (Spec 03 R5b + Spec 09 §2.4 output)
    - `data/hrp_shrunkcov_intensity.csv` (Spec 03 output)
    - `data/inference_sharpe_diff.csv` (Spec 09 output)
    - `data/inference_spa.csv` (Spec 09 output)
    - `data/inference_bootstrap_cis.csv` (Spec 09 output)
    - All `paper/figures/*.png` (the existing `paper/` blanket-ignore must be lifted)

R4c. **Secrets hygiene:** `.env.example` must contain placeholder values only.
    If a real key is ever pasted into `.env.example` (it is git-tracked), the
    cleanup PR must (a) reset the file to placeholders and (b) instruct the
    author to rotate the key at the provider. `.env` (gitignored) is the
    only acceptable location for real keys.

R5. **`specs/` directory:** committed to the repo, listed in the top-level README.
    Index of specs in `specs/README.md` linking all 9 spec files.

R6. **New code/data files inventory** (introduced in Phases 1–3, must be
    tracked by git):
    - `src/universe.py` (Phase 1)
    - `src/denoising.py` (Phase 2)
    - `src/hrp_variants.py` (Phase 2)
    - `tests/__init__.py`, `tests/test_denoising.py`,
      `tests/test_hrp_variants.py`, `tests/test_comparators.py` (Phases 2–3)
    - `notebooks/build_pit_universe.ipynb` (Phase 1)
    - All artifacts listed in R4b.

## 3. Interface

This spec is procedural — no code interface. The deliverable is a single PR
("repo cleanup") containing:
- Deleted directories/files per R3.
- Updated `.gitignore` per R4.
- Updated `README.md` section pointing to `specs/`.
- A short `CHANGELOG.md` entry noting the cleanup (optional).

## 4. Acceptance Criteria

AC1. `git branch` shows only `main` locally (and any in-progress feature branches
     for new work, but none of the three R1 branches).

AC2. `git stash list` is empty.

AC3. `paper/` contains exactly:
   - `paper/MACI_latex_eng/` (intact, all sub-files preserved)
   - `paper/ucema_journal/` (new continuation paper, per Spec 04)
   - `paper/figures/` (shared figures consumed by the new paper)
   - No `latex/`, `latex 2/`, `latex copia/`, `latex (2).zip`.

AC4. `.gitignore` covers all categories in R4. Running `git status` after a fresh
     notebook execution or LaTeX compile shows no auto-generated files.

AC4b. Every file in R4b is tracked by git (`git ls-files` includes it). The
     blanket `*.csv` and blanket `paper/` rules from the pre-audit `.gitignore`
     are NOT present (only the specific cache patterns from R4 are).

AC4c. `.env.example` contains no `CG-`, `sk-`, `pk-`, or other obvious
     key-prefix substrings. Audit grep:
     `grep -E '(CG-|sk-|pk-|api[_-]?key.*=.{20,})' .env.example` returns
     no matches.

AC5. `README.md` Project Structure section references `specs/` and the new
     `tests/` directory.

AC6. Cleanup PR is reviewed and merged before paper submission.

## 5. Out of Scope

- Renaming or restructuring `src/`, `notebooks/`, or `data/`.
- Migrating from `requirements.txt` + `environment.yml` to a single tool (e.g. poetry).
- CI/CD setup.

## 6. Ordering Constraint

This spec executes **last** (after Specs 01–04 and 06–09 are complete).
Performing the cleanup earlier risks losing files still referenced by
in-progress work.

## 7. References

- Repo state at start of execution: `git log` first 25 commits (see scan output).
- `git stash list` output: one stash on `feature/improve-documentation`.
