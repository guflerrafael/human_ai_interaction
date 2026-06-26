"""Figure generation and small formatting helpers.

All figures are written to ``outputs/figures`` as PNGs. Plotting uses matplotlib
only (no seaborn dependency) to keep the environment light.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})
ACCENT = "#235A82"


def fig_ols_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(-0.5, config.OLS_MAX + 1.5, 1)
    ax.hist(df["OLS"].dropna(), bins=bins, color=ACCENT, edgecolor="white")
    ax.axvline(df["OLS"].mean(), color="crimson", ls="--",
               label=f"Mean = {df['OLS'].mean():.2f}")
    ax.set(xlabel="Objective Literacy Score (0-11)", ylabel="Count",
           title="Distribution of Objective Literacy Score")
    ax.set_xticks(range(0, config.OLS_MAX + 1))
    ax.legend()
    path = config.FIGURES_DIR / "ols_distribution.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def fig_overconfidence(df: pd.DataFrame) -> str:
    gap = df["overconfidence_gap"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gap, bins=np.arange(gap.min() - 0.5, gap.max() + 1.5, 1),
            color=ACCENT, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(gap.mean(), color="crimson", ls="--", label=f"Mean = {gap.mean():.2f}")
    ax.set(xlabel="Overconfidence gap (self-estimate - OLS)", ylabel="Count",
           title="Overconfidence gap (positive = overconfident)")
    ax.legend()
    path = config.FIGURES_DIR / "overconfidence_gap.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def fig_correlation_heatmap(r: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(r.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(r.columns)), r.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(r.index)), r.index)
    for i in range(len(r.index)):
        for j in range(len(r.columns)):
            ax.text(j, i, f"{r.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(r.values[i, j]) > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation matrix (focal variables)")
    path = config.FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def fig_hypothesis_scatter(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    specs = [("competence_trust", "Competence Trust", "H1 (expected: negative)"),
             ("perceived_risk", "Perceived Risk", "H2 (expected: positive)")]
    for ax, (col, label, title) in zip(axes, specs):
        d = df[["OLS", col]].dropna()
        ax.scatter(d["OLS"], d[col], alpha=0.5, color=ACCENT)
        if len(d) > 2:
            b, a = np.polyfit(d["OLS"], d[col], 1)
            xs = np.array([d["OLS"].min(), d["OLS"].max()])
            ax.plot(xs, a + b * xs, color="crimson", lw=2)
            ax.text(0.05, 0.95, f"slope = {b:+.3f}", transform=ax.transAxes, va="top")
        ax.set(xlabel="Objective Literacy Score", ylabel=label, title=title)
    fig.suptitle("OLS vs trust subscales")
    path = config.FIGURES_DIR / "hypothesis_scatter.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def fig_domain_means(means: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["crimson" if s == "high" else ACCENT for s in means["Stakes"]]
    ax.barh(means.index[::-1], means["Mean"][::-1], color=colors[::-1])
    ax.set(xlabel="Mean trust (1-5)", title="Domain-specific trust (red = high-stakes)")
    ax.set_xlim(1, 5)
    path = config.FIGURES_DIR / "domain_trust.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def fig_subjective_vs_objective(alt_predictors: pd.DataFrame) -> str:
    """Standardized betas: objective OLS vs. subjective literacy measures."""
    d = alt_predictors.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    outcomes = [("competence_trust", "Competence Trust"), ("perceived_risk", "Perceived Risk")]
    for ax, (key, label) in zip(axes, outcomes):
        sub = d[d["Outcome"] == key]
        colors = ["crimson" if "Objective" in p else ACCENT for p in sub["Predictor"]]
        labels = [p.split(":")[-1].split("(")[0].strip() for p in sub["Predictor"]]
        bars = ax.barh(labels[::-1], sub["beta_std"][::-1], color=colors[::-1])
        for bar, p in zip(bars, sub["p"][::-1]):
            marker = "*" if p < 0.05 else ""
            ax.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2, marker,
                    va="center", ha="left" if bar.get_width() >= 0 else "right",
                    color="crimson", fontsize=14)
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="Standardized beta", title=f"-> {label}")
    fig.suptitle("Objective (red) vs. subjective (blue) literacy as predictors of trust\n(* p < .05)")
    path = config.FIGURES_DIR / "subjective_vs_objective.png"
    fig.savefig(path); plt.close(fig)
    return str(path)


def generate_all_figures(prep, corr_r, domain, alt_predictors=None) -> list[str]:
    df = prep.df
    figs = [
        fig_ols_distribution(df),
        fig_overconfidence(df),
        fig_correlation_heatmap(corr_r),
        fig_hypothesis_scatter(df),
        fig_domain_means(domain.means),
    ]
    if alt_predictors is not None:
        figs.append(fig_subjective_vs_objective(alt_predictors))
    return figs
