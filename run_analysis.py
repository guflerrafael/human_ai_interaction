#!/usr/bin/env python3
"""End-to-end analysis pipeline for the AI Literacy & Trust Calibration study.

Runs the four-step protocol (preparation -> reliability -> descriptives ->
hypothesis testing) plus the exploratory domain analysis, prints a readable
report to the console, and writes:
  * outputs/tables/*.csv      — every table as machine-readable CSV
  * outputs/figures/*.png     — figures
  * outputs/analysis_report.md — the full written report
  * outputs/prepared_data.csv — the cleaned, scored, analysis-ready dataset

Usage:  python run_analysis.py
"""

from __future__ import annotations

import pandas as pd

from analysis import (
    config,
    data_preparation,
    descriptives,
    domain_analysis,
    hypotheses,
    reliability,
    reporting,
)


class Report:
    """Accumulates Markdown for the report file while echoing to the console."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def h(self, text: str, level: int = 2) -> None:
        self.line("\n" + "#" * level + " " + text)

    def line(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def table(self, df: pd.DataFrame, floatfmt: str = "%.3f") -> None:
        body = df.to_string(float_format=lambda x: format(x, ".3f"))
        print(body)
        self.lines.append("```\n" + body + "\n```")

    def save(self, path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def _p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def main() -> None:
    rep = Report()
    rep.h("AI Literacy & Trust Calibration — Analysis Report", level=1)

    # ------------------------------------------------------------------ #
    # Step 1 — Data preparation
    # ------------------------------------------------------------------ #
    rep.h("Step 1 — Data preparation & exclusions")
    prep = data_preparation.prepare_data()
    df = prep.df

    rep.line(f"- Raw responses:                 {prep.n_raw}")
    rep.line(f"- Failed math attention check:   {prep.n_failed_math}")
    rep.line(f"- Failed 'Agree' attention check:{prep.n_failed_agree}")
    rep.line(f"- Excluded (failed either):      {prep.n_failed_either}")
    rep.line(f"- Final analysis sample (N):     {prep.n_final}")
    rep.line("\nNotes / deviations:")
    for note in prep.notes:
        rep.line(f"  * {note}")

    # Sample composition snapshot.
    rep.h("Sample composition", level=3)
    rep.line(f"- Age: M = {df['age'].mean():.1f}, SD = {df['age'].std():.1f} "
             f"(range {df['age'].min():.0f}-{df['age'].max():.0f})")
    rep.line(f"- Gender: {df['gender_label'].value_counts().to_dict()}")
    rep.line(f"- STEM vs non-STEM: {df['is_stem'].map({1: 'STEM', 0: 'non-STEM'}).value_counts().to_dict()}")

    df.to_csv(config.OUTPUT_DIR / "prepared_data.csv", index=False)

    # ------------------------------------------------------------------ #
    # Step 2 — Reliability
    # ------------------------------------------------------------------ #
    rep.h("Step 2 — Scale reliability")
    rel = reliability.run_reliability(df)
    rel_rows = []
    for key, sc in rel.items():
        bench = reliability.OMEGA_BENCHMARKS[key]
        rel_rows.append({
            "Scale": sc.name, "k items": len(sc.items), "N": sc.n,
            "alpha": round(sc.alpha, 3),
            "alpha 95% CI": f"[{sc.alpha_ci[0]:.2f}, {sc.alpha_ci[1]:.2f}]",
            "omega": round(sc.omega, 3),
            "omega (Sun 2025)": bench,
            "alpha >= .70": "yes" if sc.alpha >= reliability.ALPHA_TARGET else "NO",
        })
    rel_table = pd.DataFrame(rel_rows).set_index("Scale")
    rep.table(rel_table)
    rel_table.to_csv(config.TABLES_DIR / "reliability.csv")

    for key, sc in rel.items():
        if sc.alpha < reliability.ALPHA_TARGET:
            rep.line(f"\n{sc.name}: alpha below .70 — item diagnostics "
                     f"(remove lowest item-total r if needed):")
            rep.table(sc.item_stats.round(3))
            sc.item_stats.round(3).to_csv(
                config.TABLES_DIR / f"item_stats_{key}.csv")

    cfa = reliability.run_cfa(df)
    if cfa is None:
        rep.line(f"\nTwo-factor CFA skipped (N = {len(df)} < 150, per protocol).")
    else:
        rep.line(f"\nTwo-factor CFA: {cfa}")

    # ------------------------------------------------------------------ #
    # Step 3 — Descriptives
    # ------------------------------------------------------------------ #
    rep.h("Step 3 — Descriptive statistics")
    summ = descriptives.summary_table(df)
    rep.table(summ)
    summ.to_csv(config.TABLES_DIR / "descriptives.csv")

    rep.h("Correlation matrix", level=3)
    corr_r, corr_p = descriptives.correlation_matrix(df)
    rep.table(corr_r)
    corr_r.to_csv(config.TABLES_DIR / "correlations_r.csv")
    corr_p.round(4).to_csv(config.TABLES_DIR / "correlations_p.csv")
    rep.line("\nPre-registered directional checks:")
    for chk in descriptives.directional_checks(corr_r):
        rep.line(f"  * {chk}")

    rep.h("Overconfidence gap (one-sample t-test vs 0)", level=3)
    oc = descriptives.overconfidence_test(df)
    rep.line(f"- N = {oc.n}; mean gap = {oc.mean_gap:+.3f} (SD = {oc.sd_gap:.3f})")
    rep.line(f"- t({oc.df}) = {oc.t:.3f}, p {_p(oc.p)}, Cohen's d = {oc.cohen_d:+.3f}")
    rep.line(f"- Wilcoxon (non-parametric): W = {oc.wilcoxon_stat:.1f}, p {_p(oc.wilcoxon_p)} "
             f"(Shapiro normality p {_p(oc.shapiro_p)})")
    rep.line(f"- {oc.pct_overconfident:.0f}% overconfident, "
             f"{oc.pct_underconfident:.0f}% underconfident")
    direction = ("overestimate" if oc.mean_gap > 0 else "underestimate")
    verdict = "significant" if oc.p < 0.05 else "not significant"
    rep.line(f"  => Students {direction} their AI knowledge; effect is {verdict}.")

    rep.h("OLS item difficulty", level=3)
    diff = descriptives.ols_item_difficulty(df)
    rep.table(diff)
    diff.to_csv(config.TABLES_DIR / "ols_item_difficulty.csv")

    # ------------------------------------------------------------------ #
    # Step 4 — Hypothesis testing
    # ------------------------------------------------------------------ #
    rep.h("Step 4 — Hypothesis testing (multiple regression)")
    hyp = hypotheses.run_hypotheses(df)
    for key, res in hyp.items():
        rep.h(res.name, level=3)
        rep.line(f"Outcome: {res.outcome} | N = {res.n} | expected beta "
                 f"{'<' if res.expected_sign == 'negative' else '>'} 0")
        rep.table(res.coefficients)
        rep.line(f"\n- Model R² = {res.r2:.3f} (adj. R² = {res.adj_r2:.3f}); "
                 f"F = {res.f_stat:.2f}, p {_p(res.f_pvalue)}")
        rep.line(f"- Focal predictor OLS: standardized beta = {res.focal_beta_std:+.3f}, "
                 f"p {_p(res.focal_p)}")
        rep.line(f"- Cohen's f² (OLS) = {res.f2_focal:.3f} ({res.f2_label})")
        rep.line(f"- Residual normality (Shapiro) p {_p(res.resid_shapiro_p)}; "
                 f"max VIF = {res.vif['VIF'].max():.2f}")
        rep.line(f"  => {key} {'SUPPORTED' if res.supported else 'NOT supported'} "
                 f"(significant & in predicted direction: {res.supported}).")
        res.coefficients.to_csv(config.TABLES_DIR / f"regression_{key}.csv")
        res.vif.to_csv(config.TABLES_DIR / f"vif_{key}.csv")

    # ------------------------------------------------------------------ #
    # Step 5 — Domain-specific trust (exploratory)
    # ------------------------------------------------------------------ #
    rep.h("Step 5 — Domain-specific trust (exploratory)")
    dom = domain_analysis.run_domain_analysis(df)
    rep.table(dom.means)
    dom.means.to_csv(config.TABLES_DIR / "domain_means.csv")

    ct = dom.composite_test
    rep.line(f"\nHigh- vs low-stakes (paired): high M = {ct['mean_high']:.3f}, "
             f"low M = {ct['mean_low']:.3f}, diff = {ct['mean_diff']:+.3f}")
    rep.line(f"  t({ct['df']}) = {ct['t']:.3f}, p {_p(ct['p'])}, "
             f"Cohen's d = {ct['cohen_d']:+.3f}")

    rep.h("Targeted pairwise comparisons (Bonferroni)", level=3)
    rep.table(dom.pairwise)
    dom.pairwise.to_csv(config.TABLES_DIR / "domain_pairwise.csv")

    rep.h("Exploratory: OLS x domain trust correlations", level=3)
    rep.table(dom.ols_correlations)
    dom.ols_correlations.to_csv(config.TABLES_DIR / "domain_ols_correlations.csv")

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    rep.h("Figures")
    figs = reporting.generate_all_figures(prep, corr_r, dom)
    for f in figs:
        rep.line(f"- {f}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    rep.h("Summary of hypothesis tests")
    for key, res in hyp.items():
        rep.line(f"- {key}: beta = {res.focal_beta_std:+.3f}, p {_p(res.focal_p)}, "
                 f"f² = {res.f2_focal:.3f} -> "
                 f"{'SUPPORTED' if res.supported else 'NOT supported'}")

    rep.save(config.REPORT_FILE)
    rep.line(f"\nReport written to: {config.REPORT_FILE}")
    rep.line(f"Tables written to:  {config.TABLES_DIR}")
    rep.line(f"Figures written to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
