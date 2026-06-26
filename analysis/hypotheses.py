"""Step 4 — Hypothesis testing (two pre-registered multiple regressions).

H1:  OLS -> Competence Trust   (expected beta < 0)
H2:  OLS -> Perceived Risk     (expected beta > 0)

Both models control for age, gender (Male=1/Female=0), field (STEM=1/non-STEM=0)
and AI use frequency. For each model we report:
  * unstandardised coefficients, SEs, t, p, 95% CI;
  * standardised beta for every predictor (from a fully z-scored refit);
  * model R², adjusted R², overall F-test;
  * Cohen's f² for the focal predictor (OLS), via the partial-R² method:
        f² = (R²_full - R²_reduced) / (1 - R²_full)
    where the reduced model omits OLS. Benchmarks: .02 / .15 / .35.
  * VIFs (multicollinearity) and a Shapiro test on residuals (diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

PREDICTORS = ["OLS", "age", "gender_male", "is_stem", "ai_frequency"]
FOCAL = "OLS"


@dataclass
class RegressionResult:
    name: str
    outcome: str
    expected_sign: str
    n: int
    coefficients: pd.DataFrame   # unstd + std betas, SE, t, p, CI
    r2: float
    adj_r2: float
    f_stat: float
    f_pvalue: float
    focal_beta_std: float
    focal_p: float
    f2_focal: float
    f2_label: str
    vif: pd.DataFrame
    resid_shapiro_p: float
    supported: bool


def _f2_label(f2: float) -> str:
    if np.isnan(f2):
        return "n/a"
    if f2 >= 0.35:
        return "large"
    if f2 >= 0.15:
        return "medium"
    if f2 >= 0.02:
        return "small"
    return "negligible"


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    rows = [
        {"predictor": col, "VIF": variance_inflation_factor(Xc.values, i)}
        for i, col in enumerate(Xc.columns) if col != "const"
    ]
    return pd.DataFrame(rows).set_index("predictor").round(3)


def run_regression(df: pd.DataFrame, outcome: str, name: str,
                   expected_sign: str) -> RegressionResult:
    cols = [outcome] + PREDICTORS
    # Cast to plain float64 (gender uses a nullable Float64 dtype that statsmodels
    # cannot consume) and drop rows missing any model variable (listwise).
    data = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).dropna()
    y = data[outcome]
    X = data[PREDICTORS]

    # --- Unstandardised model ---
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # --- Standardised model (z-score everything) for comparable betas ---
    zdata = (data - data.mean()) / data.std(ddof=0)
    zmodel = sm.OLS(zdata[outcome], sm.add_constant(zdata[PREDICTORS])).fit()

    coef = pd.DataFrame({
        "b": model.params, "SE": model.bse, "t": model.tvalues, "p": model.pvalues,
        "CI_low": model.conf_int()[0], "CI_high": model.conf_int()[1],
    })
    coef["beta_std"] = zmodel.params.reindex(coef.index)
    coef = coef.loc[["const"] + PREDICTORS].round(4)

    # --- Cohen's f² for the focal predictor (partial R²) ---
    reduced_X = X.drop(columns=FOCAL)
    reduced = sm.OLS(y, sm.add_constant(reduced_X)).fit()
    f2 = (model.rsquared - reduced.rsquared) / (1 - model.rsquared) \
        if (1 - model.rsquared) > 0 else np.nan

    focal_beta_std = float(zmodel.params[FOCAL])
    focal_p = float(model.pvalues[FOCAL])
    # "Supported" = significant AND in the predicted direction.
    if expected_sign == "negative":
        supported = focal_p < 0.05 and focal_beta_std < 0
    else:
        supported = focal_p < 0.05 and focal_beta_std > 0

    return RegressionResult(
        name=name, outcome=outcome, expected_sign=expected_sign, n=len(data),
        coefficients=coef, r2=model.rsquared, adj_r2=model.rsquared_adj,
        f_stat=model.fvalue, f_pvalue=model.f_pvalue,
        focal_beta_std=focal_beta_std, focal_p=focal_p, f2_focal=f2,
        f2_label=_f2_label(f2), vif=_vif_table(X),
        resid_shapiro_p=float(stats.shapiro(model.resid).pvalue),
        supported=supported,
    )


def run_hypotheses(df: pd.DataFrame) -> dict[str, RegressionResult]:
    return {
        "H1": run_regression(df, "competence_trust",
                             "H1: OLS -> Competence Trust", "negative"),
        "H2": run_regression(df, "perceived_risk",
                             "H2: OLS -> Perceived Risk", "positive"),
    }
