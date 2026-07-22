"""
Task 3 — Detection Engine: manual rules + unsupervised models.

Branch: task3-detectors

Three independent detection layers, combined into one ranked, comparable
score, plus a per-row rule_flags list. See SCHEMA.md for the exact
output contract.

Currency handling: all threshold-based rules convert Transaction Amount
to an EGP-equivalent using a fixed exchange rate (config/thresholds.json
-> fx_to_egp), then compare against a single set of EGP thresholds.
Only EGP and USD are supported (see config/schema.json) — an unsupported
currency is rejected upstream by data_layer.py's enum check, so it
should never reach this module, but the fallback below stays as a
defensive guard in case detectors.py is ever called directly on
unvalidated data (e.g. in a test script).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config_loader import load_config

SETTINGS = load_config("settings.json")
THRESHOLDS_CFG = load_config("thresholds.json")
EGP_THRESHOLDS = THRESHOLDS_CFG["egp_thresholds"]
FX_TO_EGP = THRESHOLDS_CFG["fx_to_egp"]

MIN_HISTORY = SETTINGS["history"]["minimum_transactions"]  # min txns in (IBAN, Currency) group before stats are trusted

def compute_stat_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account, per-currency MAD and IQR flags on log-transformed amount.
    Grouping by (IBAN, Currency) means an account that uses both EGP and
    USD gets separate baselines per currency, so a normal-sized USD
    transaction never gets compared against an EGP baseline.
    """
    df = df.copy()
    df["log_amount"] = np.log(df["Transaction Amount"])  # safe: amount asserted positive upstream

    grp = df.groupby(["IBAN", "Currency"])["log_amount"]
    group_count = grp.transform("count")
    group_median = grp.transform("median")
    group_mad = grp.transform(lambda s: (s - s.median()).abs().median())
    group_q1 = grp.transform(lambda s: s.quantile(0.25))
    group_q3 = grp.transform(lambda s: s.quantile(0.75))
    group_iqr = group_q3 - group_q1

    mod_z = np.where(
        group_mad > 0, 0.6745 * (df["log_amount"] - group_median) / group_mad, 0.0
    )
    mad_flag_raw = np.abs(mod_z) > 3.5
    iqr_flag_raw = (df["log_amount"] < group_q1 - 1.5 * group_iqr) | (
        df["log_amount"] > group_q3 + 1.5 * group_iqr
    )

    insufficient = group_count < MIN_HISTORY
    df["mad_flag"] = np.where(insufficient, pd.NA, mad_flag_raw)
    df["iqr_flag"] = np.where(insufficient, pd.NA, iqr_flag_raw)
    df["stat_basis"] = np.where(insufficient, "insufficient_history", "per_account_currency")

    return df


# Feature columns fed to the models — continuous/graded signals only.
# Binary/pre-thresholded features (is_new_account, decline_then_approve_flag,
# country_mismatch, etc.) are intentionally excluded — they belong to the
# rules, not the models.
MODEL_FEATURES = [
    "log_amount", "amount_ratio", "tx_count_24h", "hour_deviation",
    "declined_burst_count", "account_age_days", "pass_through_ratio",
    "distinct_senders_24h", "distinct_recipients_24h", "dormancy_days",
]


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-imputed, with a companion `<col>_missing` flag per feature
    that actually has NaNs — so the model can distinguish "we imputed a
    typical value here" from "this row is genuinely typical." Only
    features with real NaNs get a companion column, so this doesn't
    blow up dimensionality unnecessarily.
    """
    X = df[MODEL_FEATURES].copy()

    for col in X.columns:
        if X[col].isna().any():
            X[f"{col}_missing"] = X[col].isna().astype(int)
            X[col] = X[col].fillna(X[col].median())

    return X


def rank_normalize(raw_scores: np.ndarray) -> np.ndarray:
    """High raw score = weirder. Returns 0-1, ties handled via argsort-argsort."""
    raw_scores = np.asarray(raw_scores)
    if len(raw_scores) <= 1:
        return np.zeros(len(raw_scores))
    ranks = raw_scores.argsort().argsort()
    return ranks / (len(ranks) - 1)


def fit_isolation_forest(X: pd.DataFrame, contamination=None, random_state=None) -> np.ndarray:
    # Real fraud rates run ~0.1-0.5%, NOT sklearn's ~10% default.
    contamination = SETTINGS["isolation_forest"]["contamination"] if contamination is None else contamination
    random_state = SETTINGS["isolation_forest"]["random_state"] if random_state is None else random_state
    n_estimators = SETTINGS["isolation_forest"]["n_estimators"]

    assert not X.isna().any().any(), "NaNs in model matrix — fix in build_model_matrix, not here"
    if len(X) < 2:
        return np.zeros(len(X))
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iso.fit(X)
    raw = -iso.decision_function(X)  # sign flip: high = weird
    return rank_normalize(raw)


def fit_lof(X: pd.DataFrame, n_neighbors=None) -> np.ndarray:
    n_neighbors = SETTINGS["lof"]["neighbors"] if n_neighbors is None else n_neighbors

    assert not X.isna().any().any(), "NaNs in model matrix — fix in build_model_matrix, not here"
    if len(X) < 2:
        # can't compute a *local* density with fewer than 2 points to compare against
        return np.zeros(len(X))
    k = max(1, min(n_neighbors, len(X) - 1))  # corner case: small fresh-data file
    X_scaled = StandardScaler().fit_transform(X)  # LOF is distance-based, needs scaling — IsoForest doesn't
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit_predict(X_scaled)
    raw = -lof.negative_outlier_factor_
    return rank_normalize(raw)


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    """Adds iso_score, lof_score, combined_score. Exposed separately so
    the app can show either model alone or the combined view."""
    df = df.copy()
    X = build_model_matrix(df)
    df["iso_score"] = fit_isolation_forest(X)
    df["lof_score"] = fit_lof(X)
    # max, not average: either model alone is enough to push a
    # transaction up the queue — deliberately more sensitive than
    # requiring both to agree.
    df["combined_score"] = np.maximum(df["iso_score"], df["lof_score"])
    return df


# Currency conversion
def _amount_in_egp(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rate = df["Currency"].map(FX_TO_EGP).astype(float)
    unresolved = rate.isna()
    amount_egp = df["Transaction Amount"] * rate
    return amount_egp, unresolved


# Structuring helper
def _structuring_burst_count(
    df: pd.DataFrame,
    amount_egp: pd.Series,
    window: str = "24h",
    band=(0.80, 1.00),
):

    unit = EGP_THRESHOLDS["p2p_transfer"]

    near = (
        df["Transaction Type"].eq("Transfer")
        & amount_egp.between(unit * band[0], unit * band[1])
    )

    result = np.zeros(len(df), dtype=int)

    for _, g in (
        df.assign(_near=near, _amt=amount_egp)
        .sort_values("timestamp")
        .groupby("IBAN", sort=False)
    ):

        idx = g.index.to_numpy()
        ts = g["timestamp"].to_numpy()
        flags = g["_near"].to_numpy()

        left = 0

        for i in range(len(g)):

            while left < i and ts[left] < ts[i] - pd.Timedelta(window):
                left += 1

            result[idx[i]] = flags[left:i+1].sum()

    return (
        pd.Series(result, index=df.index),
        pd.Series(near, index=df.index),
    )


# Impossible travel
def _impossible_travel(
    df: pd.DataFrame,
    hours: int = 6,
):

    prev_country = (
        df.groupby("IBAN")["Transaction Country"]
        .shift()
    )

    delta = df["hours_since_last_tx"]

    return (
        prev_country.notna()
        & (prev_country != df["Transaction Country"])
        & (delta < hours)
    )


# Extreme amount
def _extreme_amount(df):

    mad = df["mad_flag"].fillna(False)
    iqr = df["iqr_flag"].fillna(False)

    return (
        mad
        | iqr
        | (df["amount_ratio"] >= 8)
    )


# Foreign currency spike
def _foreign_currency_spike(
    df,
    amount_egp,
):

    prev_currency = (
        df.groupby("IBAN")["Currency"]
        .shift()
    )

    return (
        prev_currency.notna()
        & (prev_currency != df["Currency"])
        & (amount_egp > 30000)
    )

# Layering
def _layering(df):
    prev_amount = (
        df.groupby("IBAN")["Transaction Amount"]
        .shift()
    )

    prev_dc = (
        df.groupby("IBAN")["Debit/Credit"]
        .shift()
    )

    prev_time = (
        df.groupby("IBAN")["timestamp"]
        .shift()
    )

    delta = (
        df["timestamp"] - prev_time
    ).dt.total_seconds() / 60

    return (
        (prev_dc == "Credit")
        &
        (df["Debit/Credit"] == "Debit")
        &
        (delta <= 60)
        &
        (
            abs(
                df["Transaction Amount"] -
                prev_amount
            )
            <= prev_amount * 0.20
        )
    )


def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amount_egp, unresolved_currency_mask = _amount_in_egp(df)

    # Existing Rules (improved)
    rule_decline_probing = (
        df["decline_then_approve_flag"]
        .fillna(False)
        .astype(bool)
    )

    rule_new_p2p_transfer = (
        (df["Transaction Type"] == "Transfer")
        &
        (df["Beneficiary Type"].isin(
            ["Bank", "Digital Wallet"]
        ))
        &
        df["new_merchant_flag"].fillna(False)
        &
        (amount_egp > EGP_THRESHOLDS["p2p_transfer"])
        &
        (df["combined_score"] > 0.75)
    )

    rule_mule_pass_through = (
        df["pass_through_ratio"].between(0.90, 1.10)
        &
        (
            (df["distinct_senders_24h"] >= 4)
            |
            (df["distinct_recipients_24h"] >= 4)
        )
    )

    rule_dormant_reactivation = (
        (df["dormancy_days"] >= 45)
        &
        (df["Debit/Credit"] == "Debit")
        &
        (amount_egp > EGP_THRESHOLDS["dormant_reactivation"])
    )

    rule_country_mismatch = (
        df["new_country_flag"].fillna(False)
        &
        (df["country_signal_strength"] == "solid")
        &
        (df["Transaction Country"] != df["Nationality"])
    )

    rule_new_account_large_amount = (
        df["is_new_account"]
        &
        (amount_egp > EGP_THRESHOLDS["new_account"])
    )

    device_age = df["device_age_days"]

    rule_device_added_same_day = (
        device_age.notna()
        &
        (device_age >= 0)
        &
        (device_age < 1)
        &
        (amount_egp > 15000)
    )

    data_issue_negative_device_age = (
        device_age.notna()
        &
        (device_age < 0)
    )

    # Improved Structuring
    burst_count, near_threshold = _structuring_burst_count(
        df,
        amount_egp,
    )

    rule_structuring = (
        near_threshold
        &
        (burst_count >= 3)
        &
        (df["combined_score"] > 0.70)
    )

    # Improved Velocity Burst
    median_tx = (
        df.groupby("IBAN")["tx_count_24h"]
        .transform("median")
    )

    rule_velocity_burst = (
        (df["tx_count_24h"] >= 8)
        &
        (df["tx_count_24h"] >= median_tx * 2.5)
    )

    # NEW RULES
    rule_impossible_travel = _impossible_travel(df)

    rule_extreme_amount = (
        _extreme_amount(df)
        &
        (df["combined_score"] > 0.80)
    )

    rule_foreign_currency_spike = (
        _foreign_currency_spike(
            df,
            amount_egp,
        )
        &
        (df["combined_score"] > 0.70)
    )

    rule_layering = (
        _layering(df)
        &
        (df["combined_score"] > 0.75)
    )

    # Register Rules
    rule_defs = {

        "decline_probing": rule_decline_probing,

        "new_p2p_transfer": rule_new_p2p_transfer,

        "mule_pass_through": rule_mule_pass_through,

        "dormant_reactivation": rule_dormant_reactivation,

        "country_mismatch_solid": rule_country_mismatch,

        "new_account_large_amount": rule_new_account_large_amount,

        "device_added_same_day": rule_device_added_same_day,

        "structuring": rule_structuring,

        "velocity_burst": rule_velocity_burst,

        "impossible_travel": rule_impossible_travel,

        "extreme_amount": rule_extreme_amount,

        "foreign_currency_spike": rule_foreign_currency_spike,

        "layering": rule_layering,
    }

    for name, mask in rule_defs.items():
        df[f"rule_{name}"] = mask.fillna(False)

    rule_cols = [
        f"rule_{n}"
        for n in rule_defs
    ]

    df["rule_flags"] = df[rule_cols].apply(
        lambda row: [
            name
            for name, value
            in zip(rule_defs.keys(), row)
            if value
        ],
        axis=1,
    )

    quality = pd.Series(
        [[] for _ in range(len(df))],
        index=df.index,
    )

    quality = quality.mask(
        data_issue_negative_device_age,
        quality.apply(
            lambda x: x + ["device_added_after_transaction"]
        ),
    )

    quality = quality.mask(
        unresolved_currency_mask,
        quality.apply(
            lambda x: x + ["currency_threshold_missing"]
        ),

    )

    df["data_quality_flags"] = quality

    # Better Final Score
    has_rule = (
        df["rule_flags"]
        .apply(len)
        > 0
    )

    agreement = (
        has_rule
        &
        (df["combined_score"] >= 0.80)

    )

    df["combined_score"] = np.where(
        agreement,
        1.0,
        np.where(
            has_rule,
            np.maximum(
                df["combined_score"],
                0.95,
            ),
            df["combined_score"],
        ),
    )
    df["case_type"] = df["rule_flags"].apply(classify_case)

    def confidence(score):
        if score >= 0.95:
            return "High"

        if score >= 0.80:
            return "Medium"
        return "Low"
    df["confidence"] = df["combined_score"].apply(confidence)
    return df

FRAUD_RULES = {
    "decline_probing",
    "new_p2p_transfer",
    "dormant_reactivation",
    "country_mismatch_solid",
    "new_account_large_amount",
    "device_added_same_day",
    "structuring",
    "velocity_burst",
    "impossible_travel",
    "extreme_amount",
    "foreign_currency_spike",
}

MULE_RULES = {
    "mule_pass_through",
    "layering",
}


def classify_case(rule_flags):
    flags = set(rule_flags)

    fraud = len(flags & FRAUD_RULES)
    mule = len(flags & MULE_RULES)

    if fraud and mule:
        return "Mixed"

    if fraud:
        return "Fraud"

    if mule:
        return "Mule"

    if len(flags) > 0:
        return "Suspicious"

    return "Normal"

def score(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three layers. Input must be Task 2's feature-engineered output."""
    df = compute_stat_flags(df)
    df = score_models(df)
    df = compute_rule_flags(df)
    return df


if __name__ == "__main__":
    import data_layer
    import features

    raw = data_layer.validate("data/Sample_Data.xlsx")
    featured = features.build(raw)
    result = score(featured)

    print("Shape:", result.shape)
    print()
    print("Stat basis distribution:", result["stat_basis"].value_counts().to_dict())
    print()
    print("rule_flags: rows with >=1 rule fired:", (result["rule_flags"].apply(len) > 0).sum(), "/", len(result))
    print()
    print("Score columns present:", [c for c in ["iso_score", "lof_score", "combined_score"] if c in result.columns])
