"""Step 5 — Domain-specific trust (exploratory).

  * Mean trust per domain (5.1-5.8), ranked.
  * Paired t-test: high-stakes vs low-stakes composites (per respondent).
  * Targeted paired comparisons among the stakes-defining domains, Bonferroni
    corrected (protocol: correct when comparing >2 domains).
  * Exploratory correlations of OLS with trust in each domain — the slide
    predicts literacy effects are strongest in high-stakes domains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config


@dataclass
class DomainResults:
    means: pd.DataFrame                 # per-domain mean/SD, ranked
    composite_test: dict                # high vs low stakes paired t-test
    pairwise: pd.DataFrame              # targeted pairs, Bonferroni corrected
    ols_correlations: pd.DataFrame      # OLS x each domain


def domain_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in config.DOMAIN_ITEMS:
        s = pd.to_numeric(df[item], errors="coerce").dropna()
        stake = "high" if item in config.HIGH_STAKES_DOMAINS else "low"
        rows.append({"Domain": config.DOMAIN_LABELS[item], "Stakes": stake,
                     "N": s.size, "Mean": round(s.mean(), 3), "SD": round(s.std(), 3)})
    return pd.DataFrame(rows).sort_values("Mean", ascending=False).set_index("Domain")


def composite_paired_test(df: pd.DataFrame) -> dict:
    pair = df[["domain_high_stakes", "domain_low_stakes"]].dropna()
    hi, lo = pair["domain_high_stakes"], pair["domain_low_stakes"]
    t, p = stats.ttest_rel(hi, lo)
    diff = hi - lo
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    return {"n": len(pair), "mean_high": hi.mean(), "mean_low": lo.mean(),
            "mean_diff": diff.mean(), "t": t, "df": len(pair) - 1, "p": p, "cohen_d": d}


def pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Targeted high- vs low-stakes domain pairs with Bonferroni correction."""
    pairs = [("5.6", "5.1"), ("5.7", "5.1"), ("5.6", "5.4"), ("5.7", "5.4")]
    rows = []
    for hi, lo in pairs:
        d = df[[hi, lo]].apply(pd.to_numeric, errors="coerce").dropna()
        t, p = stats.ttest_rel(d[hi], d[lo])
        rows.append({
            "High-stakes": config.DOMAIN_LABELS[hi],
            "Low-stakes": config.DOMAIN_LABELS[lo],
            "Mean diff": round(d[hi].mean() - d[lo].mean(), 3),
            "t": round(t, 3), "p_raw": p,
        })
    res = pd.DataFrame(rows)
    res["p_bonferroni"] = np.minimum(res["p_raw"] * len(res), 1.0)
    res["sig (a=.05)"] = res["p_bonferroni"] < 0.05
    res["p_raw"] = res["p_raw"].round(4)
    res["p_bonferroni"] = res["p_bonferroni"].round(4)
    return res.set_index(["High-stakes", "Low-stakes"])


def ols_domain_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in config.DOMAIN_ITEMS:
        d = df[["OLS", item]].apply(pd.to_numeric, errors="coerce").dropna()
        r, p = stats.pearsonr(d["OLS"], d[item])
        rows.append({"Domain": config.DOMAIN_LABELS[item],
                     "Stakes": "high" if item in config.HIGH_STAKES_DOMAINS else "low",
                     "r (OLS)": round(r, 3), "p": round(p, 4)})
    return pd.DataFrame(rows).set_index("Domain")


def run_domain_analysis(df: pd.DataFrame) -> DomainResults:
    return DomainResults(
        means=domain_means(df),
        composite_test=composite_paired_test(df),
        pairwise=pairwise_comparisons(df),
        ols_correlations=ols_domain_correlations(df),
    )
