"""Tests for src/denoising.py — Specs 01 (denoising) and 02 (detoning)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.denoising import (
    mp_pdf,
    fit_max_eigenvalue,
    denoise_corr_constant_residual,
    detone_corr,
)


RNG = np.random.default_rng(42)


def _make_synthetic_returns(T: int = 1000, N: int = 100, signal_rank: int = 5,
                             signal_strength: float = 0.4) -> np.ndarray:
    """Generate returns with a known low-rank common-factor structure plus noise.

    Each asset loads on `signal_rank` latent factors, ensuring the correlation
    matrix has roughly `signal_rank` signal eigenvalues above the MP bound.
    """
    factors = RNG.standard_normal((T, signal_rank))
    loadings = RNG.standard_normal((signal_rank, N)) * signal_strength
    noise = RNG.standard_normal((T, N)) * np.sqrt(1 - signal_strength ** 2 * signal_rank)
    return factors @ loadings + noise


# ----------------------------------------------------------------------
# Spec 01 — Denoising
# ----------------------------------------------------------------------

class TestMPPDF:
    def test_pdf_is_nonneg_and_integrates_to_one(self):
        pdf = mp_pdf(var=1.0, q=10.0, pts=2000)
        assert (pdf.values >= 0).all()
        # Trapezoidal integration over the support
        area = np.trapezoid(pdf.values, pdf.index.values)
        assert abs(area - 1.0) < 0.01, f"area={area:.4f}"

    def test_requires_q_gt_one(self):
        with pytest.raises(ValueError):
            mp_pdf(var=1.0, q=0.5, pts=100)


class TestDenoisingCorrectness:
    """Spec 01 AC1 — shape, unit diagonal, symmetry, PSD."""

    def setup_method(self):
        returns = _make_synthetic_returns(T=1000, N=50, signal_rank=5)
        corr = pd.DataFrame(returns).corr().values
        self.corr = corr
        self.q = 1000 / 50

    def test_denoised_shape(self):
        out = denoise_corr_constant_residual(self.corr, q=self.q)
        assert out.shape == self.corr.shape

    def test_unit_diagonal(self):
        out = denoise_corr_constant_residual(self.corr, q=self.q)
        assert np.allclose(np.diag(out), 1.0, atol=1e-10)

    def test_symmetric(self):
        out = denoise_corr_constant_residual(self.corr, q=self.q)
        assert np.allclose(out, out.T, atol=1e-10)

    def test_psd(self):
        out = denoise_corr_constant_residual(self.corr, q=self.q)
        eigvals = np.linalg.eigvalsh(out)
        assert eigvals.min() >= -1e-10, f"min eigenvalue = {eigvals.min():.2e}"


class TestDenoisingSignalRecovery:
    """Spec 01 AC2 — identifies a signal subspace of rank ~5 and reduces noise variance."""

    def test_signal_rank_identification_and_noise_reduction(self):
        returns = _make_synthetic_returns(T=1000, N=100, signal_rank=5)
        corr = pd.DataFrame(returns).corr().values
        q = 1000 / 100
        eigvals_in = np.linalg.eigvalsh(corr)[::-1]
        out = denoise_corr_constant_residual(corr, q=q)
        eigvals_out = np.linalg.eigvalsh(out)[::-1]

        # Count signal eigenvalues (above MP upper bound = (1 + 1/sqrt(q))^2 for var=1)
        e_max_theory = (1 + (1 / q) ** 0.5) ** 2
        n_signal_in = int((eigvals_in > e_max_theory).sum())
        # Loose bound: identified signal in [3, 8] (AC2 explicitly allows 3-8)
        assert 3 <= n_signal_in <= 8, f"identified signal rank = {n_signal_in}"

        # Spread of small (noise) eigenvalues should collapse to a constant after
        # denoising; check std of bottom (N - signal) eigenvalues drops by >50%.
        noise_in = eigvals_in[n_signal_in:]
        noise_out = eigvals_out[n_signal_in:]
        reduction = 1.0 - noise_out.std() / max(noise_in.std(), 1e-12)
        assert reduction > 0.5, f"noise std reduction = {reduction:.2%}"


class TestDenoisingTracePreserved:
    """Bonus AC — trace of denoised correlation matrix equals N (preserved by construction)."""

    def test_trace_preserved(self):
        returns = _make_synthetic_returns(T=500, N=20)
        corr = pd.DataFrame(returns).corr().values
        out = denoise_corr_constant_residual(corr, q=500 / 20)
        assert abs(np.trace(out) - corr.shape[0]) < 1e-8


# ----------------------------------------------------------------------
# Spec 02 — Detoning
# ----------------------------------------------------------------------

class TestDetoningCorrectness:
    """Spec 02 AC1 — shape, unit diagonal, symmetry, PSD."""

    def setup_method(self):
        returns = _make_synthetic_returns(T=1000, N=50, signal_rank=5)
        corr = pd.DataFrame(returns).corr().values
        # Apply denoising first (per spec — detoning is on top of denoised)
        self.corr_denoised = denoise_corr_constant_residual(corr, q=1000 / 50)

    def test_detoned_shape(self):
        out = detone_corr(self.corr_denoised, n_market_components=1)
        assert out.shape == self.corr_denoised.shape

    def test_unit_diagonal(self):
        out = detone_corr(self.corr_denoised, n_market_components=1)
        assert np.allclose(np.diag(out), 1.0, atol=1e-10)

    def test_symmetric(self):
        out = detone_corr(self.corr_denoised, n_market_components=1)
        assert np.allclose(out, out.T, atol=1e-10)

    def test_psd_after_renorm(self):
        out = detone_corr(self.corr_denoised, n_market_components=1)
        eigvals = np.linalg.eigvalsh(out)
        # Bottom eigenvalue may be very close to zero due to component removal;
        # accept small numerical negatives.
        assert eigvals.min() >= -1e-8, f"min eigenvalue = {eigvals.min():.2e}"


class TestDetoningSemantics:
    """Spec 02 AC2 — mean off-diagonal correlation strictly decreases after detoning
    in the presence of a genuine market mode (all-positive common factor loadings)."""

    def test_off_diagonal_correlation_decreases_with_market_mode(self):
        # Construct data with an explicit positive-loaded common factor — this
        # mimics the crypto market mode where everything co-moves with BTC.
        T, N = 1500, 30
        rng = np.random.default_rng(7)
        market = rng.standard_normal(T) * 1.0
        loadings = np.abs(rng.standard_normal(N)) * 0.7 + 0.3  # all positive in [0.3, 1.3]
        idio = rng.standard_normal((T, N)) * 0.6
        returns = np.outer(market, loadings) + idio
        corr = pd.DataFrame(returns).corr().values

        corr_denoised = denoise_corr_constant_residual(corr, q=T / N)
        corr_detoned = detone_corr(corr_denoised, n_market_components=1)

        def mean_off_diag(m: np.ndarray) -> float:
            mask = ~np.eye(m.shape[0], dtype=bool)
            return float(m[mask].mean())

        before = mean_off_diag(corr_denoised)
        after = mean_off_diag(corr_detoned)
        assert after < before, f"detoning did not reduce mean off-diag corr: {before:.3f} -> {after:.3f}"

    def test_top_eigenvector_loadings_positive_with_market_mode(self):
        """Spec 02 AC3 — the removed (top) eigenvector loadings are all same-signed
        when a positive market mode dominates."""
        T, N = 1500, 30
        rng = np.random.default_rng(7)
        market = rng.standard_normal(T)
        loadings = np.abs(rng.standard_normal(N)) * 0.7 + 0.3
        idio = rng.standard_normal((T, N)) * 0.6
        returns = np.outer(market, loadings) + idio
        corr = pd.DataFrame(returns).corr().values
        corr_denoised = denoise_corr_constant_residual(corr, q=T / N)
        eigvals, eigvecs = np.linalg.eigh(corr_denoised)
        top_vec = eigvecs[:, -1]  # largest eigenvalue is last in ascending order
        # Sign-normalize: same-signed means abs(sum) close to sum(abs)
        assert abs(top_vec.sum()) / np.abs(top_vec).sum() > 0.9, (
            "top eigenvector is not a market mode (loadings not aligned)"
        )


class TestDetoningInvalidInput:
    def test_rejects_n_components_geq_N(self):
        N = 5
        c = np.eye(N)
        with pytest.raises(ValueError):
            detone_corr(c, n_market_components=N)

    def test_rejects_n_components_zero(self):
        with pytest.raises(ValueError):
            detone_corr(np.eye(5), n_market_components=0)
