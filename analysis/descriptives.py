"""Step 3 — Descriptive statistics.

  * Means, SDs (and range / skew) for all constructs and key controls.
  * Bivariate correlation matrix among the focal variables, with the
    pre-registered directional checks (OLS vs Competence Trust should be
    negative; OLS vs Perceived Risk should be positive).
  * Overconfidence gap: one-sample t-test against zero, plus a non-parametric
    Wilcoxon signed-rank fallback and a normality check.
  * Item-level OLS difficulty (proportion correct) with floor/ceiling flags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config

# Variables summarised in the descriptives table (tidy name -> label).
SUMMARY_VARS = {
    "age": "Age (years)",
    "OLS": "Objective Literacy Score (0-11)",
    "self_estimate": "Self-estimated score (0-11)",
    "overconfidence_gap": "Overconfidence gap (est - OLS)",
    "competence_trust": "Competence Trust (1-5)",
    "perceived_risk": "Perceived Risk (1-5)",
    "confidence": "Self-rated confidence (1-5)",
    "self_rated_knowledge": "Self-rated knowledge (1-5)",
    "ai_frequency": "AI use frequency (1-6)",
    "domain_low_stakes": "Domain trust: low-stakes (1-5)",
    "domain_high_stakes": "Domain trust: high-stakes (1-5)",
}

CORR_VARS = ["OLS", "competence_trust", "perceived_risk",
             "age", "ai_frequency", "self_rated_knowledge"]


@dataclass
class OverconfidenceResult:
    n: int
    mean_gap: float
    sd_gap: float
    t: float
    df: int
    p: float
    cohen_d: float
    shapiro_p: float
    wilcoxon_stat: float
    wilcoxon_p: float
    pct_overconfident: float
    pct_underconfident: float


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, label in SUMMARY_VARS.items():
        s = pd.to_numeric(df[var], errors="coerce").dropna()
        rows.append({
            "Variable": label, "N": int(s.size), "Mean": s.mean(), "SD": s.std(),
            "Min": s.min(), "Max": s.max(), "Skew": stats.skew(s) if s.size > 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("Variable").round(3)


def correlation_matrix(df: pd.DataFrame, vars_=CORR_VARS):
    """Return (r matrix, p matrix) of pairwise Pearson correlations."""
    data = df[vars_].apply(pd.to_numeric, errors="coerce")
    r = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    p = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    for a in vars_:
        for b in vars_:
            pair = data[[a, b]].dropna()
            if a == b:
                r.loc[a, b], p.loc[a, b] = 1.0, 0.0
            elif len(pair) > 2:
                rr, pp = stats.pearsonr(pair[a], pair[b])
                r.loc[a, b], p.loc[a, b] = rr, pp
    return r.astype(float).round(3), p.astype(float)


def directional_checks(r: pd.DataFrame) -> list[str]:
    """Human-readable check of the two pre-registered correlation directions."""
    out = []
    r1 = r.loc["OLS", "competence_trust"]
    out.append(
        f"OLS x Competence Trust  r = {r1:+.3f}  "
        f"({'as predicted (negative)' if r1 < 0 else 'OPPOSITE to H1 prediction'})"
    )
    r2 = r.loc["OLS", "perceived_risk"]
    out.append(
        f"OLS x Perceived Risk    r = {r2:+.3f}  "
        f"({'as predicted (positive)' if r2 > 0 else 'OPPOSITE to H2 prediction'})"
    )
    return out


def overconfidence_test(df: pd.DataFrame) -> OverconfidenceResult:
    gap = pd.to_numeric(df["overconfidence_gap"], errors="coerce").dropna()
    n = gap.size
    t, p = stats.ttest_1samp(gap, 0.0)
    cohen_d = gap.mean() / gap.std(ddof=1) if gap.std(ddof=1) else np.nan
    shapiro_p = stats.shapiro(gap).pvalue if 3 <= n <= 5000 else np.nan
    # Wilcoxon needs non-zero differences.
    nonzero = gap[gap != 0]
    if nonzero.size:
        w_stat, w_p = stats.wilcoxon(nonzero)
    else:
        w_stat, w_p = np.nan, np.nan
    return OverconfidenceResult(
        n=n, mean_gap=gap.mean(), sd_gap=gap.std(ddof=1), t=t, df=n - 1, p=p,
        cohen_d=cohen_d, shapiro_p=shapiro_p, wilcoxon_stat=w_stat, wilcoxon_p=w_p,
        pct_overconfident=float((gap > 0).mean() * 100),
        pct_underconfident=float((gap < 0).mean() * 100),
    )


def ols_item_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion correct per OLS item, with floor/ceiling flags (>.85 / <.15)."""
    rows = []
    for item in config.OLS_ITEMS:
        p_correct = df[f"{item}_correct"].mean()
        flag = "ceiling" if p_correct > 0.85 else "floor" if p_correct < 0.15 else ""
        rows.append({"Item": item, "Prop. correct": round(p_correct, 3), "Flag": flag})
    return pd.DataFrame(rows).set_index("Item")
