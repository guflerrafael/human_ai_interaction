"""Central configuration: paths, column resolution, scoring keys and mappings.

The raw Google-Forms export uses long, human-readable column headers. Rather than
hard-coding those exact strings everywhere, we resolve columns by their stable
item-number prefix (e.g. ``"3.10."``) once, here, and expose tidy short names to
the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "survey_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_FILE = OUTPUT_DIR / "analysis_report.md"

for _d in (OUTPUT_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Objective Literacy Score (OLS) — answer key
# Source: Hornberger, Bewersdorff & Nerdel (2023), correct options marked in the
# survey documentation (docs/Survey_Doc.pdf). Each value is the correct option
# letter; scoring matches on the leading letter of the stored response.
# --------------------------------------------------------------------------- #
OLS_ITEMS = [f"3.{i}" for i in range(1, 12)]  # 3.1 .. 3.11
OLS_ANSWER_KEY = {
    "3.1": "a",
    "3.2": "a",
    "3.3": "b",
    "3.4": "c",
    "3.5": "c",
    "3.6": "b",
    "3.7": "c",
    "3.8": "a",
    "3.9": "d",
    "3.10": "b",
    "3.11": "b",
}
OLS_MAX = len(OLS_ITEMS)  # 11
SELF_ESTIMATE_ITEM = "3.12"  # used for the overconfidence gap

# --------------------------------------------------------------------------- #
# HAITS trust subscales
# --------------------------------------------------------------------------- #
COMPETENCE_ITEMS = [f"4.{i}" for i in range(1, 7)]      # 4.1 .. 4.6
PERCEIVED_RISK_ITEMS = [f"4.{i}" for i in range(7, 12)]  # 4.7 .. 4.11
# Per study decision: Perceived-Risk items are RISK-worded, so raw responses are
# used (5 = "strongly agree it is risky" = high perceived risk). No 6 - x flip.
REVERSE_CODE_PERCEIVED_RISK = False

# Domain-specific trust (exploratory), 1 = Not at all .. 5 = Completely
DOMAIN_ITEMS = [f"5.{i}" for i in range(1, 9)]  # 5.1 .. 5.8
DOMAIN_LABELS = {
    "5.1": "General knowledge",
    "5.2": "Learning / study",
    "5.3": "Academic writing",
    "5.4": "Programming / technical",
    "5.5": "Professional / work",
    "5.6": "Health-related",
    "5.7": "Personal advice / life",
    "5.8": "News & current events",
}
# Stakes grouping (per presentation, Section 5).
LOW_STAKES_DOMAINS = ["5.1", "5.2", "5.4", "5.5"]
HIGH_STAKES_DOMAINS = ["5.3", "5.6", "5.7", "5.8"]

# --------------------------------------------------------------------------- #
# Attention checks
# --------------------------------------------------------------------------- #
ATTENTION_MATH_ITEM = "2.4"        # "What is 4 + 4?"  -> 8
ATTENTION_MATH_ANSWER = 8
ATTENTION_AGREE_ITEM = "4.12"      # "Please select 'Agree'"
ATTENTION_AGREE_VALUE = 4          # 4 = Agree on the 1-5 Likert

# --------------------------------------------------------------------------- #
# Control / covariate value mappings
# --------------------------------------------------------------------------- #
# 2.1 — AI use frequency, encoded as an ordinal 1..6 (low -> high).
AI_FREQUENCY_ORDER = {
    "Never or almost never": 1,
    "Less than once a month": 2,
    "A few times a month": 3,
    "A few times a week": 4,
    "Daily": 5,
    "Multiple times per day": 6,
}

# 1.2 — Gender. Encoded Male = 1, Female = 0; "Prefer not to say" -> missing
# (a single respondent; dropped listwise from the regressions only).
GENDER_ENCODING = {"Male": 1, "Female": 0, "Prefer not to say": pd.NA}

# 1.3 — Field of study collapsed to a STEM (1) vs non-STEM (0) indicator to
# preserve degrees of freedom (the raw item has 18 fields, many singletons).
# Mapping is explicit and editable; whitespace is stripped before lookup.
STEM_FIELDS = {
    "Computer Science/Informatics",
    "AI/Data Science/Machine Learning",
    "Engineering/Technology",
    "Mathematics/Statistics",
    "Natural Sciences",
    "Health/Medicine/Psychology",
    "Neuroscience",
    "Physics",
}
# Everything else (Business, Law, Social Sciences, Communication, Arts/Design,
# Economics/Econometrics, Architecture, HR, Political Science, ...) -> non-STEM.

# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
TIMESTAMP_COL = "Zeitstempel"


def _find_col(columns: list[str], token: str) -> str:
    """Return the single column whose header contains ``token``.

    Raises if zero or more than one column matches, so a changed export fails
    loudly rather than silently scoring the wrong column.
    """
    matches = [c for c in columns if token in c]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one column containing {token!r}, found {len(matches)}: "
            f"{matches}"
        )
    return matches[0]


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map tidy item codes -> actual long column names in the export."""
    cols = list(df.columns)
    mapping: dict[str, str] = {}

    # Demographics / usage (matched by leading "N.N. ").
    simple = {
        "1.1": "1.1.", "1.2": "1.2.", "1.3": "1.3.", "1.4": "1.4.",
        "1.5": "1.5.", "1.6": "1.6.", "2.1": "2.1.", "2.2": "2.2.",
        "2.3": "2.3.", "2.4": "2.4.", "2.5": "2.5.",
    }
    for code, token in simple.items():
        mapping[code] = _find_col(cols, token)

    # OLS knowledge items + self-estimate (3.1. .. 3.12.).
    for i in range(1, 13):
        mapping[f"3.{i}"] = _find_col(cols, f"3.{i}.")

    # Trust items are inside brackets, e.g. "[4.1. The AI tools are reliable.]".
    for i in range(1, 13):
        mapping[f"4.{i}"] = _find_col(cols, f"[4.{i}.")

    # Domain-specific trust "[5.1. ...]".
    for i in range(1, 9):
        mapping[f"5.{i}"] = _find_col(cols, f"[5.{i}.")

    return mapping
