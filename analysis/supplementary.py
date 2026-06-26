"""Step 6 — Supplementary / robustness analyses (post-hoc, exploratory).

These go beyond the pre-registered protocol to stress-test the H1/H2 null
results and probe alternative explanations:

  1. Influence diagnostics — are the null results driven by a few outliers?
  2. Subjective vs. objective literacy — does self-rated knowledge/confidence
     predict trust even though the *objective* score (OLS) does not?
  3. Perceived Risk item-reduction — does trimming the weakest item(s) rescue
     the subscale's reliability or change the H2 conclusion?
  4. Trust Differentiation Index (TDI) — per-respondent SD across the 8
     domain-trust items (protocol Appendix B calibration measure): do more
     literate respondents differentiate trust more sharply across contexts?
  5. Power analysis — was the study positioned to detect the planned medium
     effect (f² = .15), or are the H1/H2 nulls simply underpowered?
  6. Moderation — does gender or STEM background change the literacy-trust
     relationship?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.power import FTestPower

from . import config
from .hypotheses import PREDICTORS as BASE_PREDICTORS  # ["OLS", age, gender_male, is_stem, ai_frequency]

CONTROLS = [p for p in BASE_PREDICTORS if p != "OLS"]


# --------------------------------------------------------------------------- #
# 1. Influence diagnostics
# --------------------------------------------------------------------------- #
def influence_sensitivity(df: pd.DataFrame, outcome: str, top_k=(1, 3, 5)) -> dict:
    cols = [outcome] + BASE_PREDICTORS
    data = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).dropna().reset_index(drop=True)
    y, X = data[outcome], sm.add_constant(data[BASE_PREDICTORS])
    model = sm.OLS(y, X).fit()
    cooks = np.asarray(OLSInfluence(model).cooks_distance[0])
    order = np.argsort(cooks)[::-1]

    top_rows = data.iloc[order[:5]].copy()
    top_rows.insert(0, "cooks_d", cooks[order[:5]])

    refits = []
    refits.append({"excluded": 0, "beta_OLS": model.params["OLS"],
                    "p": model.pvalues["OLS"], "R2": model.rsquared, "n": len(data)})
    for k in top_k:
        sub = data.drop(data.index[order[:k]])
        m = sm.OLS(sub[outcome], sm.add_constant(sub[BASE_PREDICTORS])).fit()
        refits.append({"excluded": k, "beta_OLS": m.params["OLS"], "p": m.pvalues["OLS"],
                        "R2": m.rsquared, "n": len(sub)})
    return {
        "cooks_threshold": 4 / len(data),
        "top_influential": top_rows.round(3),
        "sensitivity": pd.DataFrame(refits).set_index("excluded").round(4),
    }


# --------------------------------------------------------------------------- #
# 2. Subjective vs. objective literacy
# --------------------------------------------------------------------------- #
ALT_PREDICTORS = {
    "OLS": "Objective literacy (OLS, 0-11)",
    "self_rated_knowledge": "Subjective: self-rated AI knowledge (2.5)",
    "confidence": "Subjective: self-rated AI confidence (2.3)",
    "overconfidence_gap": "Overconfidence gap (self-estimate - OLS)",
}


def _standardized_beta(data: pd.DataFrame, outcome: str, focal: str, controls: list[str]):
    z = (data - data.mean()) / data.std(ddof=0)
    return sm.OLS(z[outcome], sm.add_constant(z[[focal] + controls])).fit().params[focal]


def alternative_predictors_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["competence_trust", "perceived_risk"]:
        for focal, label in ALT_PREDICTORS.items():
            cols = [outcome, focal] + CONTROLS
            data = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).dropna()
            y, X = data[outcome], sm.add_constant(data[[focal] + CONTROLS])
            m = sm.OLS(y, X).fit()
            beta = _standardized_beta(data, outcome, focal, CONTROLS)
            rows.append({
                "Outcome": outcome, "Predictor": label,
                "beta_std": round(beta, 3), "p": round(m.pvalues[focal], 4),
                "R2": round(m.rsquared, 3), "n": len(data),
            })
    return pd.DataFrame(rows).set_index(["Outcome", "Predictor"])


# --------------------------------------------------------------------------- #
# 3. Perceived Risk item-reduction
# --------------------------------------------------------------------------- #
def perceived_risk_item_reduction(df: pd.DataFrame) -> pd.DataFrame:
    items = config.PERCEIVED_RISK_ITEMS
    drop_sets = [[], ["4.7"], ["4.7", "4.11"]]
    rows = []
    for drop in drop_sets:
        keep = [c for c in items if c not in drop]
        item_data = df[keep].apply(pd.to_numeric, errors="coerce").dropna()
        alpha, _ = pg.cronbach_alpha(data=item_data)
        scale = df[keep].mean(axis=1)
        cols = pd.concat([scale.rename("scale"), df[["OLS"] + CONTROLS]], axis=1)
        cols = cols.apply(pd.to_numeric, errors="coerce").astype(float).dropna()
        m = sm.OLS(cols["scale"], sm.add_constant(cols[["OLS"] + CONTROLS])).fit()
        beta = _standardized_beta(cols, "scale", "OLS", CONTROLS)
        rows.append({
            "Dropped items": ", ".join(drop) if drop else "(none — full 5-item scale)",
            "k items": len(keep), "alpha": round(alpha, 3),
            "H2 beta_std (OLS)": round(beta, 3), "p": round(m.pvalues["OLS"], 4),
        })
    return pd.DataFrame(rows).set_index("Dropped items")


# --------------------------------------------------------------------------- #
# 4. Trust Differentiation Index
# --------------------------------------------------------------------------- #
@dataclass
class TDIResult:
    descriptives: pd.Series
    r_with_ols: float
    p_with_ols: float
    n: int


def trust_differentiation_index(df: pd.DataFrame) -> TDIResult:
    tdi = df[config.DOMAIN_ITEMS].apply(pd.to_numeric, errors="coerce").std(axis=1)
    data = pd.concat([tdi.rename("tdi"), df["OLS"]], axis=1).dropna()
    r, p = stats.pearsonr(data["tdi"], data["OLS"])
    return TDIResult(descriptives=tdi.describe().round(3), r_with_ols=r, p_with_ols=p, n=len(data))


# --------------------------------------------------------------------------- #
# 5. Power analysis
# --------------------------------------------------------------------------- #
@dataclass
class PowerResult:
    n: int
    df_resid: int
    min_detectable_f2: float
    power_at_planned_f2: float
    power_at_observed: dict[str, float]


def _power_for_f2(f2: float, df_resid: int) -> float:
    fp = FTestPower()
    # statsmodels' df_num/df_denom naming is reversed for this routine: pass the
    # residual (denominator) df as df_num and the 1-df numerator as df_denom.
    return fp.power(effect_size=np.sqrt(f2), df_num=df_resid, df_denom=1, alpha=0.05)


def power_analysis(n: int, n_predictors: int, observed_f2: dict[str, float],
                   planned_f2: float = 0.15) -> PowerResult:
    df_resid = n - n_predictors - 1
    f2_min = brentq(lambda f2: _power_for_f2(f2, df_resid) - 0.80, 1e-5, 3.0)
    return PowerResult(
        n=n, df_resid=df_resid, min_detectable_f2=f2_min,
        power_at_planned_f2=_power_for_f2(planned_f2, df_resid),
        power_at_observed={k: _power_for_f2(v, df_resid) for k, v in observed_f2.items()},
    )


# --------------------------------------------------------------------------- #
# 6. Moderation (gender, STEM)
# --------------------------------------------------------------------------- #
def moderation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["competence_trust", "perceived_risk"]:
        for mod in ["gender_male", "is_stem"]:
            others = [p for p in CONTROLS if p != mod]
            cols = list(dict.fromkeys([outcome, "OLS", mod] + others))
            data = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).dropna()
            data["interaction"] = data["OLS"] * data[mod]
            X = sm.add_constant(data[["OLS", mod, "interaction"] + others])
            m = sm.OLS(data[outcome], X).fit()
            rows.append({
                "Outcome": outcome, "Moderator": mod,
                "interaction_b": round(m.params["interaction"], 4),
                "p": round(m.pvalues["interaction"], 4), "n": len(data),
            })
    return pd.DataFrame(rows).set_index(["Outcome", "Moderator"])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class SupplementaryResults:
    influence: dict[str, dict] = field(default_factory=dict)
    alt_predictors: pd.DataFrame = None
    risk_item_reduction: pd.DataFrame = None
    tdi: TDIResult = None
    power: PowerResult = None
    moderation: pd.DataFrame = None


def run_supplementary(df: pd.DataFrame, observed_f2: dict[str, float]) -> SupplementaryResults:
    return SupplementaryResults(
        influence={o: influence_sensitivity(df, o) for o in
                   ["competence_trust", "perceived_risk"]},
        alt_predictors=alternative_predictors_table(df),
        risk_item_reduction=perceived_risk_item_reduction(df),
        tdi=trust_differentiation_index(df),
        power=power_analysis(n=len(df.dropna(subset=BASE_PREDICTORS + ["competence_trust"])),
                             n_predictors=len(BASE_PREDICTORS), observed_f2=observed_f2),
        moderation=moderation_table(df),
    )
