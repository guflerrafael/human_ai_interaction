# AI Literacy & Trust Calibration — Analysis

Statistical analysis pipeline for the study *"The Role of AI Literacy in Trust
Calibration toward AI-Generated Content."*

## Research question & hypotheses
Does **objective AI literacy** (OLS) predict **trust in AI tools** among
university students?

- **H1** — Higher AI literacy → **lower** Competence Trust (expected β < 0).
- **H2** — Higher AI literacy → **higher** Perceived Risk (expected β > 0).

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```
Everything is regenerated into `outputs/` each run.

## Project layout
```
data/survey_data.csv          Raw Google-Forms export (input)
analysis/
  config.py                   Paths, column resolution, answer key, mappings
  data_preparation.py         Step 1: parse, exclusions, scoring
  reliability.py              Step 2: Cronbach α, McDonald ω, optional CFA
  descriptives.py             Step 3: M/SD, correlations, overconfidence test
  hypotheses.py               Step 4: the two multiple regressions (H1, H2)
  domain_analysis.py          Step 5: domain-specific trust (exploratory)
  reporting.py                Figure generation
run_analysis.py               Orchestrator — run this
outputs/
  analysis_report.md          Full written report
  prepared_data.csv           Cleaned, scored, analysis-ready dataset
  tables/*.csv                Every table as CSV
  figures/*.png               Figures
```

## Pipeline steps
1. **Data preparation** — exclude attention-check failures (item 2.4 "4+4"=8;
   item 4.12 "select Agree"); score OLS 0–11 against the Hornberger et al. (2023)
   answer key; compute the overconfidence gap (self-estimate − OLS); build the
   two trust subscales and the domain composites.
2. **Reliability** — Cronbach's α (95% CI) and McDonald's ω for each subscale,
   compared to the Sun et al. (2025) benchmarks; item-total diagnostics when
   α < .70. Two-factor CFA runs only if N ≥ 150 (per protocol).
3. **Descriptives** — means/SDs, correlation matrix with the pre-registered
   directional checks, one-sample t-test (+ Wilcoxon) on the overconfidence gap,
   and item-level OLS difficulty.
4. **Hypothesis testing** — two multiple regressions controlling for age,
   gender, field (STEM vs non-STEM) and AI-use frequency. Reports standardized
   β, R², p, and Cohen's f² for the focal predictor, plus VIF and residual
   diagnostics.
5. **Domain trust (exploratory)** — per-domain means, high- vs low-stakes paired
   t-test, Bonferroni-corrected targeted comparisons, and OLS × domain
   correlations.
6. **Supplementary / robustness (post-hoc)** — stress-tests of the H1/H2 nulls:
   Cook's-distance influence diagnostics with leave-out refits, subjective vs.
   objective literacy as competing predictors, Perceived-Risk item-trimming,
   the Trust Differentiation Index (Appendix B calibration measure), a power
   analysis (was the design even capable of detecting the planned effect?),
   and gender/STEM moderation checks.

## Decisions & deviations from the written protocol
These were the underspecified or conflicting points, resolved deliberately and
flagged in the report; all are centralised in `analysis/config.py` so they are
easy to change.

- **Completion-time exclusion — skipped.** The export has only a submission
  timestamp, not a per-response duration, so the "bottom 5th percentile
  completion time" rule cannot be applied.
- **Perceived Risk scoring — raw (high = high risk).** Items 4.7–4.11 are
  risk-worded, so a raw "strongly agree" already means high perceived risk. This
  honours the protocol's repeatedly stated direction ("high score = high risk",
  H2 expects β > 0); the literal "subtract from 6" wording would have *reversed*
  that direction, so it was not applied. Toggle: `REVERSE_CODE_PERCEIVED_RISK`.
- **Field of study control — STEM vs non-STEM.** The raw item has 18 fields with
  many singletons; a binary STEM indicator preserves degrees of freedom and
  matches the protocol's rationale. Mapping: `config.STEM_FIELDS`.
- **Gender — Male = 1, Female = 0**, "Prefer not to say" (n = 1) set missing and
  dropped listwise from the regressions only.
- **Attention check 2.4** accepts "8" (with any surrounding text) or the word
  "eight"; rejects other numbers.
