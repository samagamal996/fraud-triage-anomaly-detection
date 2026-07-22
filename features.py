"""
Task 2 — Feature Engineering: per-account behavioral baselining.

The functions in this module are deliberately causal: a row may use the
current transaction and transactions before it, but never a later row.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections import Counter, deque

import numpy as np
import pandas as pd
from numpy.typing import NDArray


WINDOW_24H = pd.Timedelta(hours=24)
WINDOW_48H = pd.Timedelta(hours=48)

REQUIRED_COLUMNS = {
    "IBAN",
    "Account Open Date",
    "Nationality",
    "Transaction ID",
    "timestamp",
    "Channel",
    "Transaction Type",
    "Debit/Credit",
    "Transaction Amount",
    "Status",
    "Transaction Country",
    "Beneficiary Name",
    "Beneficiary IBAN/Wallet",
    "Device ID",
    "Device Add Date",
}

SIGNAL_STRENGTHS = ["solid", "weak", "weakest"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add account-relative behavioral features to validated transactions.

    Rows are returned in stable ``(IBAN, timestamp)`` order. Numeric features
    use neutral zero/one values when an account has no usable history, and
    flags whose underlying concept does not apply use pandas' nullable boolean
    dtype.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "features.build() is missing required columns: " + ", ".join(missing)
        )

    df = df.sort_values(["IBAN", "timestamp"], kind="mergesort").copy()
    _validate_inputs(df)
    grouped = df.groupby("IBAN", sort=False, dropna=False)
    # Separate grouping for anything that SUMS Transaction Amount. Without a
    # real exchange rate, summing e.g. 5000 EGP + 100 USD as "5100" is
    # meaningless -- confirmed on real data (14,697.21 was actually
    # 64.26 EGP + 14,557.00 EGP + 75.95 USD added as if one currency).
    # Counts (tx_count_24h, declined_burst_count) and time-based features
    # (dormancy_days, hour_deviation) are currency-agnostic and unaffected.
    grouped_currency = df.groupby(["IBAN", "Currency"], sort=False, dropna=False)

    # Group A — core baseline
    df["log_amount"] = _log_amount(df)
    df["amount_ratio"] = _amount_ratio(df, grouped_currency)
    df["tx_count_24h"] = _tx_count_24h(df, grouped)
    df["sum_48h_window"] = _sum_48h_window(df, grouped)
    df["new_country_flag"], df["country_signal_strength"] = _country_novelty(
        df, grouped
    )
    df["new_device_flag"] = _device_novelty(df, grouped)
    df["new_merchant_flag"] = _merchant_novelty(df, grouped)
    df["hour_deviation"] = _hour_deviation(df, grouped)

    # Group B — decline-then-approve probing
    df["declined_burst_count"] = _declined_burst(df, grouped)
    df["recent_decline_density"] = _recent_decline_density(df, grouped)
    df["decline_then_approve_flag"] = (
        df["Status"].astype("string").str.casefold().eq("approved")
        & df["declined_burst_count"].ge(2)
    ).astype(bool)

    # Group C — new account + big transaction
    df["account_age_days"] = _age_in_days(df["timestamp"], df["Account Open Date"])
    df["is_new_account"] = df["account_age_days"].lt(30)

    # Group D — mule detection
    transfer = df["Transaction Type"].astype("string").str.casefold().eq("transfer")
    approved = df["Status"].astype("string").str.casefold().eq("approved")
    credit = df["Debit/Credit"].astype("string").str.casefold().eq("credit")
    debit = df["Debit/Credit"].astype("string").str.casefold().eq("debit")
    incoming = transfer & approved & credit
    outgoing = transfer & approved & debit

    df["inflow_sum_24h"] = _rolling_masked_sum(df, incoming, WINDOW_24H, keys=("IBAN", "Currency"))
    df["outflow_sum_24h"] = _rolling_masked_sum(df, outgoing, WINDOW_24H, keys=("IBAN", "Currency"))
    # No inflow means that pass-through behaviour is not measurable yet.
    df["pass_through_ratio"] = pd.Series(
        np.divide(
            df["outflow_sum_24h"].to_numpy(dtype=float),
            df["inflow_sum_24h"].to_numpy(dtype=float),
            out=np.zeros(len(df), dtype=float),
            where=df["inflow_sum_24h"].to_numpy(dtype=float) > 0,
        ),
        index=df.index,
        dtype="float64",
    )

    counterparty = (
        df["Beneficiary IBAN/Wallet"]
        .astype("string")
        .fillna(df["Beneficiary Name"].astype("string"))
    )
    df["distinct_senders_24h"] = _rolling_distinct_count(
        df, counterparty, incoming, WINDOW_24H
    )
    df["distinct_recipients_24h"] = _rolling_distinct_count(
        df, counterparty, outgoing, WINDOW_24H
    )
    df["dormancy_days"] = _dormancy_days(df, grouped)

    # Group E — country mismatch
    country = _normalise_text(df["Transaction Country"])
    nationality = _normalise_text(df["Nationality"])
    df["country_mismatch"] = country.ne(nationality).astype(bool)
    df["country_mismatch_strength"] = pd.Categorical(
        df["country_signal_strength"],
        categories=SIGNAL_STRENGTHS,
        ordered=True,
    )

    # Group F — device age
    df["device_age_days"] = _age_in_days(
        df["timestamp"], df["Device Add Date"], nullable=True
    )
    df["hours_since_last_tx"] = (
    grouped["timestamp"]
    .diff()
    .dt.total_seconds()
    .div(3600)
    .fillna(np.inf)
    )

    return df


def _validate_inputs(df: pd.DataFrame) -> None:
    if df["IBAN"].isna().any():
        raise ValueError("IBAN must not contain null values")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must have a datetime64 dtype")
    if df["timestamp"].isna().any():
        raise ValueError("timestamp must not contain null values")

    amount = pd.Series(
        pd.to_numeric(df["Transaction Amount"].to_numpy(), errors="coerce"),
        index=df.index,
        dtype="float64",
    )
    if amount.isna().any() or amount.le(0).any():
        raise ValueError("Transaction Amount must contain positive numeric values")


def _log_amount(df: pd.DataFrame) -> pd.Series:
    amounts = df["Transaction Amount"].to_numpy(dtype=float)
    return pd.Series(np.log1p(amounts), index=df.index, dtype="float64")


def _amount_ratio(df: pd.DataFrame, grouped) -> pd.Series:
    prior_sum = grouped["Transaction Amount"].cumsum() - df["Transaction Amount"]
    prior_count = grouped.cumcount()
    prior_average = prior_sum.div(prior_count.replace(0, np.nan))
    ratio = df["Transaction Amount"].div(prior_average)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float)


def _tx_count_24h(df: pd.DataFrame, grouped) -> pd.Series:
    return _rolling_count(df, WINDOW_24H)


def _sum_48h_window(df: pd.DataFrame, grouped) -> pd.Series:
    all_rows = pd.Series(True, index=df.index)
    return _rolling_masked_sum(df, all_rows, WINDOW_48H, keys=("IBAN", "Currency"))


def _country_novelty(df: pd.DataFrame, grouped):
    countries = _normalise_text(df["Transaction Country"])
    flag = _first_seen_by_account(df, countries)

    tx_type = df["Transaction Type"].astype("string").str.casefold()
    channel = df["Channel"].astype("string").str.casefold()
    strength = np.select(
        [
            tx_type.str.startswith("e-commerce", na=False),
            tx_type.eq("transfer") & channel.isin(["mobile app", "web app"]),
        ],
        ["weak", "weakest"],
        default="solid",
    )
    category = pd.Series(
        pd.Categorical(strength, categories=SIGNAL_STRENGTHS, ordered=True),
        index=df.index,
        name="country_signal_strength",
    )
    return flag.astype(bool), category


def _device_novelty(df: pd.DataFrame, grouped) -> pd.Series:
    devices = _normalise_text(df["Device ID"])
    return _nullable_first_seen_by_account(df, devices, df["Device ID"].notna())


def _merchant_novelty(df: pd.DataFrame, grouped) -> pd.Series:
    merchants = _normalise_text(df["Beneficiary Name"])
    return _nullable_first_seen_by_account(
        df, merchants, df["Beneficiary Name"].notna()
    )


def _hour_deviation(df: pd.DataFrame, grouped) -> pd.Series:
    hour = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
        + df["timestamp"].dt.second / 3600
    )
    angle = hour * (2 * np.pi / 24)
    sin_value = pd.Series(
        np.sin(angle.to_numpy(dtype=float)), index=df.index, dtype="float64"
    )
    cos_value = pd.Series(
        np.cos(angle.to_numpy(dtype=float)), index=df.index, dtype="float64"
    )

    key = df["IBAN"]
    prior_sin = sin_value.groupby(key, sort=False).cumsum() - sin_value
    prior_cos = cos_value.groupby(key, sort=False).cumsum() - cos_value
    count = grouped.cumcount()
    usual_angle = pd.Series(
        np.arctan2(
            prior_sin.to_numpy(dtype=float),
            prior_cos.to_numpy(dtype=float),
        ),
        index=df.index,
        dtype="float64",
    )
    usual_hour = usual_angle.mod(2 * np.pi) * (24 / (2 * np.pi))
    raw_distance = (hour - usual_hour).abs()
    distance = raw_distance.where(raw_distance.le(12), 24 - raw_distance)
    return distance.where(count.gt(0), 0.0).astype(float)


def _declined_burst(df: pd.DataFrame, grouped) -> pd.Series:
    declined = df["Status"].astype("string").str.casefold().eq("declined")
    counts, _ = _prior_window_counts(df, declined, WINDOW_24H)
    return counts


def _recent_decline_density(df: pd.DataFrame, grouped) -> pd.Series:
    declined = df["Status"].astype("string").str.casefold().eq("declined")
    decline_count, total_count = _prior_window_counts(df, declined, WINDOW_24H)
    return pd.Series(
        np.divide(
            decline_count.to_numpy(dtype=float),
            total_count.to_numpy(dtype=float),
            out=np.zeros(len(df), dtype=float),
            where=total_count.to_numpy(dtype=np.int64) > 0,
        ),
        index=df.index,
        dtype="float64",
    )


def _dormancy_days(df: pd.DataFrame, grouped) -> pd.Series:
    previous = grouped["timestamp"].shift(1)
    elapsed = (df["timestamp"] - previous).dt.total_seconds().div(86_400)
    return elapsed.fillna(0.0).clip(lower=0.0).astype(float)


def _age_in_days(
    timestamp: pd.Series, date: pd.Series, *, nullable: bool = False
) -> pd.Series:
    parsed_date = pd.to_datetime(date, errors="coerce")
    age = (timestamp - parsed_date).dt.total_seconds().div(86_400)
    if not nullable and age.isna().any():
        raise ValueError("Account Open Date must contain valid dates")
    return age.astype(float)


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _first_seen_by_account(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    duplicated = pd.DataFrame(
        {"IBAN": df["IBAN"].to_numpy(), "value": values.to_numpy()},
        index=df.index,
    ).duplicated(["IBAN", "value"])
    return ~duplicated


def _nullable_first_seen_by_account(
    df: pd.DataFrame, values: pd.Series, applicable: pd.Series
) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if applicable.any():
        subset = pd.DataFrame(
            {
                "IBAN": df.loc[applicable, "IBAN"].to_numpy(),
                "value": values.loc[applicable].to_numpy(),
            },
            index=df.index[applicable],
        )
        result.loc[applicable] = ~subset.duplicated(["IBAN", "value"])
    return result


def _rolling_count(df: pd.DataFrame, window: pd.Timedelta) -> pd.Series:
    values = np.empty(len(df), dtype=np.int64)
    for positions in _group_positions(df):
        left = 0
        times = df["timestamp"].iloc[positions].tolist()
        for local, timestamp in enumerate(times):
            cutoff = timestamp - window
            while left < local and times[left] <= cutoff:
                left += 1
            values[positions[local]] = local - left + 1
    return pd.Series(values, index=df.index, dtype="int64")


def _rolling_masked_sum(
    df: pd.DataFrame,
    mask: pd.Series,
    window: pd.Timedelta,
    keys: tuple[str, ...] = ("IBAN",),
) -> pd.Series:
    result = np.zeros(len(df), dtype=float)
    amounts = df["Transaction Amount"].astype(float).to_numpy()
    selected = mask.fillna(False).to_numpy(dtype=bool)

    for positions in _group_positions(df, keys=keys):
        left = 0
        running = 0.0
        times = df["timestamp"].iloc[positions].tolist()
        for local, timestamp in enumerate(times):
            pos = positions[local]
            cutoff = timestamp - window
            while left < local and times[left] <= cutoff:
                old_pos = positions[left]
                if selected[old_pos]:
                    running -= amounts[old_pos]
                left += 1
            if selected[pos]:
                running += amounts[pos]
            result[pos] = running
    return pd.Series(result, index=df.index, dtype=float)


def _prior_window_counts(
    df: pd.DataFrame, event_mask: pd.Series, window: pd.Timedelta
) -> tuple[pd.Series, pd.Series]:
    event_counts = np.zeros(len(df), dtype=np.int64)
    total_counts = np.zeros(len(df), dtype=np.int64)
    selected = event_mask.fillna(False).to_numpy(dtype=bool)

    for positions in _group_positions(df):
        left = 0
        events = 0
        times = df["timestamp"].iloc[positions].tolist()
        for local, timestamp in enumerate(times):
            cutoff = timestamp - window
            while left < local and times[left] <= cutoff:
                if selected[positions[left]]:
                    events -= 1
                left += 1
            pos = positions[local]
            event_counts[pos] = events
            total_counts[pos] = local - left
            if selected[pos]:
                events += 1

    return (
        pd.Series(event_counts, index=df.index, dtype="int64"),
        pd.Series(total_counts, index=df.index, dtype="int64"),
    )


def _rolling_distinct_count(
    df: pd.DataFrame,
    keys: pd.Series,
    mask: pd.Series,
    window: pd.Timedelta,
) -> pd.Series:
    result = np.zeros(len(df), dtype=np.int64)
    selected = mask.fillna(False).to_numpy(dtype=bool)
    key_values = keys.to_numpy()

    for positions in _group_positions(df):
        active: deque[tuple[pd.Timestamp, object]] = deque()
        counts: Counter = Counter()
        for pos in positions:
            timestamp = df["timestamp"].iloc[pos]
            cutoff = timestamp - window
            while active and active[0][0] <= cutoff:
                _, expired_key = active.popleft()
                counts[expired_key] -= 1
                if counts[expired_key] == 0:
                    del counts[expired_key]

            key = key_values[pos]
            if selected[pos] and not pd.isna(key):
                active.append((timestamp, key))
                counts[key] += 1
            result[pos] = len(counts)

    return pd.Series(result, index=df.index, dtype="int64")


def _group_positions(
    df: pd.DataFrame, keys: tuple[str, ...] = ("IBAN",)
) -> Iterator[NDArray[np.intp]]:
    groups = df.groupby(list(keys), sort=False, dropna=False).indices
    for positions in groups.values():
        yield np.asarray(positions, dtype=np.intp)