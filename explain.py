"""
Task 4 — Explanation Layer: turn scores into plain-English justifications.

Owner: [fill in name]
Branch: task4-explain

This is the piece most directly graded on Demo Day — every alert must
carry a reason a non-technical analyst can act on. "The model said so"
is a failing answer. See SCHEMA.md for the exact output contract.
"""

import pandas as pd


def justify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alert_reason and flag_source columns to a scored dataframe.

    Args:
        df: output of detectors.score()

    Returns:
        Same dataframe + alert_reason, flag_source per SCHEMA.md Stage 4.
    """
    df = df.copy()
    df["flag_source"] = df.apply(_classify_source, axis=1)
    df["alert_reason"] = df.apply(_build_reason, axis=1)
    return df


def _classify_source(row) -> str:
    """Return 'rule', 'model', or 'both' based on rule_flags and combined_score."""
    raise NotImplementedError


def _build_reason(row) -> str:
    """
    Build a human-readable justification sentence from this row's own
    feature values. Different template depending on flag_source.

    Rule-triggered example:
        "Matched rule: transfer > 10,000 EGP to new country within 30 days
        of account opening."

    Model-triggered example (pick the 2-4 most extreme features for this row):
        "amount is 8.2x this account's 30-day average - first-ever
        transaction in country: GB - 3rd transaction within one hour
        (usual: 2/week) - device never seen on this account"

    TODO: pick the top N most extreme features per row rather than
    listing everything, so the explanation stays readable.
    """
    raise NotImplementedError
