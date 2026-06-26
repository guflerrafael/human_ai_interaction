"""Step 1 — Data preparation.

Loads the raw export, parses free-text / Likert responses into numbers, applies
the pre-registered exclusions, scores the OLS and the two trust subscales, and
returns a tidy analysis-ready DataFrame plus a record of what happened.

Deviation from protocol (documented): the export contains only a submission
timestamp, not a per-response completion duration, so the "exclude bottom 5th
percentile completion time" rule cannot be applied and is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Low-level parsers
# --------------------------------------------------------------------------- #
def parse_likert(value) -> float:
    """Extract the leading 1-5 code from responses like '4 - Agree' or '3'."""
    if pd.isna(value):
        return np.nan
    m = re.match(r"\s*([1-5])", str(value))
    return float(m.group(1)) if m else np.nan


def parse_option_letter(value) -> str | None:
    """Extract the option letter (a-e) from responses like 'c) It is ...'."""
    if pd.isna(value):
        return None
    m = re.match(r"\s*([a-eA-E])\)", str(value))
    return m.group(1).lower() if m else None


def parse_self_estimate(value) -> float:
    """Parse the 0-11 self-estimate (item 3.12).

    Handles free text: '10 to 11' -> 10 (first integer), 'i don't know' -> NaN.
    Values outside 0-11 are treated as missing.
    """
    if pd.isna(value):
        return np.nan
    m = re.search(r"\d+", str(value))
    if not m:
        return np.nan
    n = int(m.group())
    return float(n) if 0 <= n <= config.OLS_MAX else np.nan


def passes_math_check(value) -> bool:
    """Attention check 2.4 ('What is 4 + 4?'): accept 8 or the word 'eight'.

    Robust to embellished answers ('8, assuming a decimal base.') and rejects
    wrong numbers ('67', '9', '4').
    """
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if "eight" in text:
        return True
    numbers = re.findall(r"\d+", text)
    return any(int(n) == config.ATTENTION_MATH_ANSWER for n in numbers)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """Output of the preparation step."""

    df: pd.DataFrame                      # analysis-ready, exclusions applied
    raw: pd.DataFrame                     # parsed but pre-exclusion
    cols: dict[str, str]                  # tidy code -> raw column name
    n_raw: int = 0
    n_failed_math: int = 0
    n_failed_agree: int = 0
    n_failed_either: int = 0
    n_final: int = 0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def prepare_data(path=config.DATA_FILE) -> PreparedData:
    raw = pd.read_csv(path)
    cols = config.resolve_columns(raw)
    n_raw = len(raw)

    out = pd.DataFrame(index=raw.index)

    # ---- Demographics & controls ----------------------------------------- #
    out["age"] = pd.to_numeric(raw[cols["1.1"]], errors="coerce")

    gender_raw = raw[cols["1.2"]].astype("string").str.strip()
    out["gender_label"] = gender_raw
    out["gender_male"] = gender_raw.map(config.GENDER_ENCODING).astype("Float64")

    field_raw = raw[cols["1.3"]].astype("string").str.strip()
    out["field_label"] = field_raw
    out["is_stem"] = field_raw.isin(config.STEM_FIELDS).astype(int)

    out["academic_status"] = raw[cols["1.4"]].astype("string").str.strip()
    out["ai_course_experience"] = raw[cols["1.5"]].astype("string").str.strip()
    out["ai_usage_duration"] = raw[cols["1.6"]].astype("string").str.strip()

    freq_raw = raw[cols["2.1"]].astype("string").str.strip()
    out["ai_frequency_label"] = freq_raw
    out["ai_frequency"] = freq_raw.map(config.AI_FREQUENCY_ORDER)

    out["weekly_time"] = raw[cols["2.2"]].astype("string").str.strip()
    out["confidence"] = raw[cols["2.3"]].map(parse_likert)          # 2.3 (1-5)
    out["self_rated_knowledge"] = raw[cols["2.5"]].map(parse_likert)  # 2.5 (1-5)

    # ---- OLS scoring ------------------------------------------------------ #
    for item in config.OLS_ITEMS:
        letters = raw[cols[item]].map(parse_option_letter)
        out[f"{item}_correct"] = (letters == config.OLS_ANSWER_KEY[item]).astype(int)
    out["OLS"] = out[[f"{i}_correct" for i in config.OLS_ITEMS]].sum(axis=1)

    # Self-estimate & overconfidence gap (positive = overconfident).
    out["self_estimate"] = raw[cols[config.SELF_ESTIMATE_ITEM]].map(parse_self_estimate)
    out["overconfidence_gap"] = out["self_estimate"] - out["OLS"]

    # ---- Trust subscales -------------------------------------------------- #
    for item in config.COMPETENCE_ITEMS + config.PERCEIVED_RISK_ITEMS:
        out[item] = raw[cols[item]].map(parse_likert)

    if config.REVERSE_CODE_PERCEIVED_RISK:
        for item in config.PERCEIVED_RISK_ITEMS:
            out[item] = 6 - out[item]

    out["competence_trust"] = out[config.COMPETENCE_ITEMS].mean(axis=1)
    out["perceived_risk"] = out[config.PERCEIVED_RISK_ITEMS].mean(axis=1)

    # ---- Domain-specific trust ------------------------------------------- #
    for item in config.DOMAIN_ITEMS:
        out[item] = raw[cols[item]].map(parse_likert)
    out["domain_low_stakes"] = out[config.LOW_STAKES_DOMAINS].mean(axis=1)
    out["domain_high_stakes"] = out[config.HIGH_STAKES_DOMAINS].mean(axis=1)

    # ---- Attention checks ------------------------------------------------- #
    math_pass = raw[cols[config.ATTENTION_MATH_ITEM]].map(passes_math_check)
    agree_pass = raw[cols[config.ATTENTION_AGREE_ITEM]].map(parse_likert) == \
        config.ATTENTION_AGREE_VALUE
    out["passed_math_check"] = math_pass
    out["passed_agree_check"] = agree_pass
    out["passed_attention"] = math_pass & agree_pass

    out["timestamp"] = raw[config.TIMESTAMP_COL]

    # ---- Apply exclusions ------------------------------------------------- #
    n_failed_math = int((~math_pass).sum())
    n_failed_agree = int((~agree_pass).sum())
    n_failed_either = int((~out["passed_attention"]).sum())

    df = out[out["passed_attention"]].copy().reset_index(drop=True)

    notes = [
        "Completion-time exclusion (bottom 5th percentile) SKIPPED: the export "
        "contains only a submission timestamp, not a per-response duration.",
        f"Perceived-Risk reverse coding: "
        f"{'applied (6 - x)' if config.REVERSE_CODE_PERCEIVED_RISK else 'NOT applied — raw scores, high = high risk'}.",
        "Field of study collapsed to STEM vs non-STEM; gender encoded Male=1/"
        "Female=0 with 'Prefer not to say' set missing for the regressions.",
    ]

    return PreparedData(
        df=df, raw=out, cols=cols, n_raw=n_raw,
        n_failed_math=n_failed_math, n_failed_agree=n_failed_agree,
        n_failed_either=n_failed_either, n_final=len(df), notes=notes,
    )
