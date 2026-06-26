"""Step 2 — Scale reliability (and optional dimensionality check).

For each trust subscale we report:
  * Cronbach's alpha (with 95% CI) — internal consistency.
  * McDonald's omega (total) — from a single-factor model; the metric the HAITS
    authors report, so it is directly comparable to their benchmarks.
  * Item-total correlations, and the alpha-if-item-dropped diagnostic used to
    decide which item to remove should alpha fall below .70.

Benchmarks (Sun et al., 2025): Competence Trust omega = .966; Perceived Risk
omega = .959.

A two-factor CFA is run only if N >= 150 (per protocol); otherwise it is skipped
and noted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.multivariate.factor import Factor

from . import config

ALPHA_TARGET = 0.70
OMEGA_BENCHMARKS = {"competence_trust": 0.966, "perceived_risk": 0.959}


@dataclass
class ScaleReliability:
    name: str
    items: list[str]
    n: int
    alpha: float
    alpha_ci: tuple[float, float]
    omega: float
    item_stats: pd.DataFrame  # per-item: item-total r, alpha-if-dropped


def mcdonalds_omega(item_df: pd.DataFrame) -> float:
    """McDonald's omega total from a single common-factor model.

    A one-factor model is fit (ML) to the items' correlation structure. With
    standardised loadings, omega = (Σλ)² / [(Σλ)² + Σ(1 − λ²)].
    """
    data = item_df.dropna()
    if len(data) < 3 or data.shape[1] < 2:
        return np.nan
    fa = Factor(data.values, n_factor=1, method="ml").fit()
    loadings = np.real(fa.loadings[:, 0])
    # Align signs so a predominantly negative solution does not deflate omega.
    if np.sum(loadings) < 0:
        loadings = -loadings
    sum_l = np.sum(loadings)
    uniqueness = np.sum(1.0 - loadings ** 2)
    return float(sum_l ** 2 / (sum_l ** 2 + uniqueness))


def _item_diagnostics(item_df: pd.DataFrame) -> pd.DataFrame:
    """Corrected item-total correlations and alpha-if-item-dropped."""
    data = item_df.dropna()
    rows = []
    for item in data.columns:
        rest = data.drop(columns=item).sum(axis=1)
        item_total_r = data[item].corr(rest)
        remaining = data.drop(columns=item)
        alpha_dropped = (
            pg.cronbach_alpha(data=remaining)[0] if remaining.shape[1] >= 2 else np.nan
        )
        rows.append(
            {"item": item, "item_total_r": item_total_r, "alpha_if_dropped": alpha_dropped}
        )
    return pd.DataFrame(rows).set_index("item")


def analyse_scale(df: pd.DataFrame, name: str, items: list[str]) -> ScaleReliability:
    item_df = df[items].apply(pd.to_numeric, errors="coerce")
    data = item_df.dropna()
    alpha, ci = pg.cronbach_alpha(data=data)
    return ScaleReliability(
        name=name,
        items=items,
        n=len(data),
        alpha=float(alpha),
        alpha_ci=(float(ci[0]), float(ci[1])),
        omega=mcdonalds_omega(item_df),
        item_stats=_item_diagnostics(item_df),
    )


def run_reliability(df: pd.DataFrame) -> dict[str, ScaleReliability]:
    return {
        "competence_trust": analyse_scale(df, "Competence Trust", config.COMPETENCE_ITEMS),
        "perceived_risk": analyse_scale(df, "Perceived Risk", config.PERCEIVED_RISK_ITEMS),
    }


# --------------------------------------------------------------------------- #
# Optional two-factor CFA (protocol: only if N >= 150)
# --------------------------------------------------------------------------- #
def run_cfa(df: pd.DataFrame, min_n: int = 150) -> dict | None:
    """Two-factor CFA via semopy. Returns fit indices, or None if skipped."""
    n = len(df)
    if n < min_n:
        return None
    try:
        import semopy
    except ImportError:  # pragma: no cover
        return {"skipped": "semopy not available"}

    comp = " + ".join(config.COMPETENCE_ITEMS)
    risk = " + ".join(config.PERCEIVED_RISK_ITEMS)
    model_desc = f"Competence =~ {comp}\nRisk =~ {risk}"
    # semopy needs syntactically valid variable names; rename "4.1" -> "i4_1".
    rename = {c: "i" + c.replace(".", "_") for c in
              config.COMPETENCE_ITEMS + config.PERCEIVED_RISK_ITEMS}
    for old, new in rename.items():
        model_desc = model_desc.replace(old, new)
    data = df[list(rename)].rename(columns=rename).dropna()

    model = semopy.Model(model_desc)
    model.fit(data)
    stats = semopy.calc_stats(model).T[0].to_dict()
    return {
        "n": len(data),
        "CFI": stats.get("CFI"),
        "RMSEA": stats.get("RMSEA"),
        "SRMR": stats.get("SRMR"),
    }
