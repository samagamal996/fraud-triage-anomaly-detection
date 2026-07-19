"""
Task 2 — Feature Engineering: per-account behavioral baselining.

Owner: [fill in name]
Branch: task2-features

Takes data_layer's clean output and adds account-relative behavioral
features. Every feature answers "is this weird for THIS account" —
not "is this weird globally." See SCHEMA.md for the exact output contract.
"""

import numpy as np
import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavioral feature columns to a validated transactions dataframe.

    Args:
        df: output of data_layer.validate() — must have `timestamp`,
            `IBAN`, `Transaction Amount`, `Transaction Country`,
            `Channel`, `Device ID`, `Beneficiary Name`, `Status`

    Returns:
        Same dataframe + feature columns per SCHEMA.md Stage 2.

    Important: sort by (IBAN, timestamp) before computing rolling
    features, and use windows anchored at each row's own timestamp —
    a transaction must never "see" transactions that happen after it.
    """
    df = df.sort_values(["IBAN", "timestamp"]).copy()
    grouped = df.groupby("IBAN")

    df["log_amount"] = _log_amount(df)
    df["amount_ratio"] = _amount_ratio(df, grouped)
    df["tx_count_24h"] = _tx_count_24h(df, grouped)
    df["sum_48h_window"] = _sum_48h_window(df, grouped)
    df["new_country_flag"], df["country_signal_strength"] = _country_novelty(df, grouped)
    df["new_device_flag"] = _device_novelty(df, grouped)
    df["new_merchant_flag"] = _merchant_novelty(df, grouped)
    df["hour_deviation"] = _hour_deviation(df, grouped)
    df["declined_burst_count"] = _declined_burst(df, grouped)

    return df


def _log_amount(df: pd.DataFrame) -> pd.Series:
    """log-transform Transaction Amount."""
    raise NotImplementedError


def _amount_ratio(df, grouped) -> pd.Series:
    """amount / account's own rolling average (min_periods to handle new accounts)."""
    raise NotImplementedError


def _tx_count_24h(df, grouped) -> pd.Series:
    """count of transactions by this account in the trailing 24h."""
    raise NotImplementedError


def _sum_48h_window(df, grouped) -> pd.Series:
    """rolling 48h sum per account — structuring signal."""
    raise NotImplementedError


def _country_novelty(df, grouped):
    """
    First-time-seen country per account, weighted by channel trust:
    POS/ATM = solid, E-Commerce = weak, Mobile/Web transfer = weakest.
    Returns (new_country_flag: bool series, country_signal_strength: category series)
    """
    raise NotImplementedError


def _device_novelty(df, grouped) -> pd.Series:
    """
    First-time-seen Device ID per account.
    Must be null (not False) where Device ID doesn't apply for that channel.
    """
    raise NotImplementedError


def _merchant_novelty(df, grouped) -> pd.Series:
    """First-time-seen Beneficiary Name per account. Null where no beneficiary applies."""
    raise NotImplementedError


def _hour_deviation(df, grouped) -> pd.Series:
    """Deviation of this transaction's hour from the account's usual hour distribution."""
    raise NotImplementedError


def _declined_burst(df, grouped) -> pd.Series:
    """Count of recent Declined transactions by this account before this row (card-testing signal)."""
    raise NotImplementedError
