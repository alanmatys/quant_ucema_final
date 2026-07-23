# SSRN submission pack — DT 928

Everything below is ready to paste into the SSRN "Submit a Paper" form.

PDF to upload: `paper/dt928_published.pdf` — the official UCEMA-published
DT 928 (version of record, includes the UCEMA series cover). Alternative:
`paper/ucema_journal/main.pdf` (local rebuild, adds the JEL line but is not
the published version). Recommended: the published one; the JEL codes go
into SSRN's metadata form regardless.

## Title

Hierarchical Risk Parity Variants for Cryptocurrency Portfolios: Denoising, Detoning, Tail Dependence, Embeddings, and Statistical Inference, 2020–2026

## Authors (in order)

1. Alan Matys — Universidad del CEMA — alan.matys92@gmail.com — ORCID: https://orcid.org/0009-0004-3288-1970
2. Federico Martin Rodriguez — Universidad del CEMA — federico.m.rodriguez.h@gmail.com
3. Emiliano Delfau — Universidad del CEMA — ed11@ucema.edu.ar

## Abstract (plain text)

We extend the 2025 conference study on Hierarchical Risk Parity (HRP) by addressing its two stated methodological follow-ups (Marchenko-Pastur denoising and spectral detoning) and by evaluating a broader cross-section of portfolio construction choices on a survivorship-bias-corrected, point-in-time Binance universe (146 ever-included symbols, 2020-2026). The comparison includes 27 constructed strategies spanning HRP-family variants, risk-based allocators, nested-clustering optimisers, momentum rules, and signal-aware extensions.

Across 76 monthly rebalances, the HRP-family variants remain tightly clustered in risk-adjusted performance (annualised Sharpe 0.69-0.75), and this result persists under a 547-cell hyperparameter sweep (combined Hansen Superior Predictive Ability (SPA) consistent p-value of 0.91). Strategies that alter the allocation step produce larger dispersion (notably the Minimum Variance Portfolio (MVP), the Correlation-Regularised Iterative Shrinkage Portfolio (CRISP), the Nested Clustered Optimization with CRISP allocation (NCO-CRISP), and Nested Clustered Optimization (NCO)), with CRISP and NCO-CRISP significant in pairwise Ledoit-Wolf tests versus baseline HRP in both full and post-COVID windows. However, after accounting for multiple comparisons, the headline SPA does not reject (consistent p-value of 0.51), and the Deflated Sharpe Ratio (DSR) is directionally consistent with that conclusion.

Post-COVID results indicate strong regime sensitivity: diversified HRP variants lose 51-64% of capital while Bitcoin (BTC) and BTC-concentrating allocators hold up better. We also revise an earlier interpretation on cluster stability: over the full panel, Pearson-based clustering is more stable than lower-tail dependence at monthly frequency. Overall, the evidence supports a cautious interpretation: within this universe, horizon, and search space, allocation choices are more strongly associated with performance differences than clustering perturbations, but those edges remain statistically fragile under search-adjusted inference.

## Keywords

Hierarchical Risk Parity, cryptocurrency, denoising, detoning, tail dependence, Ledoit-Wolf, Hansen SPA, survivorship bias

## JEL classification

G11, C58, C63, G17

## Working paper series field

UCEMA Working Papers (Documentos de Trabajo), No. 928, June 2026

## Related URLs

- UCEMA DT 928 page: https://ucema.edu.ar/documento-trabajo/hierarchical-risk-parity-variants-cryptocurrency-portfolios-denoising-detoning
- Published PDF: https://ucema.edu.ar/sites/default/files/2026-06/dt928.pdf
- Code repository: https://github.com/alanmatys/quant_ucema_final

## Suggested networks / eJournal classifications

- Financial Economics Network (FEN) — primary
- Econometric Modeling: Capital Markets — Portfolio Theory eJournal
- Cryptocurrency Research eJournal

## Submission checklist

- [ ] All three authors have SSRN accounts under the emails above
- [ ] Upload `dt928_published.pdf` (official UCEMA version of record)
- [ ] Paste title, abstract, keywords, JEL codes from this file
- [ ] Add co-authors by email; they confirm via the email SSRN sends
- [ ] Fill working-paper-series field (UCEMA DT 928)
- [ ] Declare: also available as UCEMA Working Paper (permitted — confirmed by UCEMA research direction, July 2026)
- [ ] After it goes live (1-3 business days): add SSRN link to README and GitHub release
