"""
Task 3 — Detection Engine: manual rules + unsupervised models.

Owner: [fill in name(s) — rules and models can split between two people]
Branch: task3-detectors

Combines hand-written rules with Isolation Forest + LOF into one ranked,
comparable score. See SCHEMA.md for the exact output contract.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Feature columns fed to the models — numeric, account-relative features only.
# Do NOT include raw amount, IDs, or timestamps directly.
MODEL_FEATURES = [
    "log_amount", "amount_ratio", "tx_count_24h", "sum_48h_window",
    "hour_deviation", "declined_burst_count",
]

# Real fraud rates run ~0.1-0.5%, NOT sklearn's ~10% default.
CONTAMINATION = 0.005


def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rule flags and model scores to a feature-engineered dataframe.

    Args:
        df: output of features.build() — must contain MODEL_FEATURES columns

    Returns:
        Same dataframe + iso_score, lof_score, combined_score, rule_flags
        per SCHEMA.md Stage 3.
    """
    df = df.copy()
    df["rule_flags"] = df.apply(_apply_rules, axis=1)
    df["iso_score"] = _isolation_forest_score(df)
    df["lof_score"] = _lof_score(df)
    df["combined_score"] = _combine_scores(df["iso_score"], df["lof_score"])
    return df


def _apply_rules(row) -> str:
    """
    Apply the 5 manual rules, written the way a real fraud analyst would,
    using the account-relative features from Task 2.

    Returns a string of which rule(s) fired, e.g. "rule_1,rule_3" or "" if none.

    TODO: define the 5 rules as a team and document the business reasoning
    for each — these get asked about directly in the defense.
    Example shape:
        IF amount_ratio > 8 AND new_device_flag THEN "large_new_device"
        IF sum_48h_window crosses just under reporting threshold across
           >=5 transactions THEN "structuring"
    """
    raise NotImplementedError


def _isolation_forest_score(df: pd.DataFrame) -> pd.Series:
    """
    Fit Isolation Forest on MODEL_FEATURES, return rank-normalized scores
    (0-1, higher = weirder).
    """
    raise NotImplementedError


def _lof_score(df: pd.DataFrame) -> pd.Series:
    """
    Fit LOF on MODEL_FEATURES, return rank-normalized scores (0-1, higher = weirder).
    """
    raise NotImplementedError


def _combine_scores(iso: pd.Series, lof: pd.Series) -> pd.Series:
    """
    Combine two rank-normalized score series into one final score.
    Team decision: average (conservative) vs max (sensitive) — pick one,
    document why, be ready to defend it.
    """
    raise NotImplementedError


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Rank-normalize raw scores to 0-1. Never compare raw IsoForest/LOF scores directly."""
    ranks = scores.argsort().argsort()
    return ranks / (len(scores) - 1)
