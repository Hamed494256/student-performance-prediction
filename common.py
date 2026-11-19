"""Shared configuration, cleaning and feature-engineering code.

Used by every numbered script AND by the Streamlit app, so training and
deployment always apply exactly the same transformations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # project root (no personal paths)
RAW_PATH = ROOT / "data" / "raw" / "student_performance_synthetic_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "student_performance_clean.csv"
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"
MODEL_DIR = ROOT / "models"
for d in (CLEAN_PATH.parent, FIG_DIR, RES_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "performance_status"
CLASS_ORDER = ["Fail", "Withdrawn", "Pass", "Distinction"]
AT_RISK = ["Fail", "Withdrawn"]                       # business definition of "at risk"

# ------------------------------------------------------------------ cleaning
PROGRAM_MAP = {"comp sci": "computer science"}        # abbreviation seen in the raw data

# Valid logical ranges used to flag IMPOSSIBLE values
VALID_RANGES = {
    "age": (16, 60),
    "attendance_rate": (0, 100),
    "midterm_score": (0, 100),
    "assignment_avg_score": (0, 100),
    "quiz_avg_score": (0, 100),
    "video_completion_rate": (0, 100),
    "assignment_submission_rate": (0, 100),
    "entry_exam_score": (0, 100),
    "prior_gpa": (0, 4),
}


def standardise_program(s: pd.Series) -> pd.Series:
    """Fix spelling / case / whitespace variants of the program label."""
    cleaned = s.map(lambda x: x.strip().lower() if isinstance(x, str) else x)
    cleaned = cleaned.map(lambda x: PROGRAM_MAP.get(x, x) if isinstance(x, str) else x)
    return cleaned.map(lambda x: x.title() if isinstance(x, str) else x)


def clean_data(df: pd.DataFrame):
    """Return (clean_df, log). Every step is counted so it can be reported."""
    log = {"rows_raw": int(len(df))}
    df = df.copy()

    # 1. inconsistent labels (count distinct raw strings BEFORE any clean-up)
    log["program_labels_before"] = int(df["program"].nunique())
    # 2. whitespace clean-up on all text columns
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    df["program"] = standardise_program(df["program"])
    log["program_labels_after"] = int(df["program"].nunique())
    # 3. duplicates
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    # 4. impossible values -> NaN (imputed later inside the ML pipeline)
    impossible = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        impossible[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    log["impossible_values_to_nan"] = impossible
    log["rows_clean"] = int(len(df))
    return df, log


# --------------------------------------------------------- feature engineering
RAW_NUMERIC_FOR_MISSING = [
    "prior_gpa", "entry_exam_score", "commute_minutes", "employment_hours_week",
    "attendance_rate", "study_hours_week", "lms_logins_week", "video_completion_rate",
    "assignment_avg_score", "quiz_avg_score", "midterm_score", "support_sessions", "age",
]
ENGINEERED = [
    "avg_assessment_score", "midterm_vs_coursework", "online_engagement",
    "study_per_credit", "total_weekly_load", "attendance_x_submission",
    "academic_gap", "low_connectivity", "n_missing_values",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (nothing learned from other rows -> no leakage)."""
    d = df.copy()
    d["avg_assessment_score"] = d[["assignment_avg_score", "quiz_avg_score", "midterm_score"]].mean(axis=1, skipna=True)
    d["midterm_vs_coursework"] = d["midterm_score"] - d[["assignment_avg_score", "quiz_avg_score"]].mean(axis=1, skipna=True)
    d["online_engagement"] = d["lms_logins_week"] * d["video_completion_rate"] / 100
    d["study_per_credit"] = d["study_hours_week"] / d["course_load_credits"]
    d["total_weekly_load"] = d["study_hours_week"] + d["employment_hours_week"] + d["commute_minutes"] / 60 * 5
    d["attendance_x_submission"] = d["attendance_rate"] * d["assignment_submission_rate"] / 100
    d["academic_gap"] = d["prior_gpa"] / 4 * 100 - d["entry_exam_score"]
    d["low_connectivity"] = (
        d["internet_access"].isin(["Limited", "Unreliable"]) | d["device_access"].isin(["Phone Only", "Shared Device"])
    ).astype(int)
    d["n_missing_values"] = d[RAW_NUMERIC_FOR_MISSING].isna().sum(axis=1)
    return d


# ----------------------------------------------------------- column groups
DROP_COLS = ["student_id", TARGET]
CATEGORICAL = [
    "academic_year", "semester", "institution_type", "program", "study_level", "enrollment_status",
    "admission_type", "scholarship_status", "financial_aid", "internet_access", "device_access",
    "course_difficulty",
]


def split_columns(df: pd.DataFrame, use_engineered: bool = True):
    feats = [c for c in df.columns if c not in DROP_COLS and (use_engineered or c not in ENGINEERED)]
    cat = [c for c in feats if c in CATEGORICAL]
    num = [c for c in feats if c not in cat]
    return feats, num, cat
