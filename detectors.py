"""
Task 3 — Detection Engine: manual rules + unsupervised models.

Branch: task3-detectors

Three independent detection layers, combined into one ranked, comparable
score, plus a per-row rule_flags list. See SCHEMA.md for the exact
output contract.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


MIN_HISTORY = 5  # minimum txns in (IBAN, Currency) group before stats are trusted

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


def fit_isolation_forest(X: pd.DataFrame, contamination=0.005, random_state=42) -> np.ndarray:
    # Real fraud rates run ~0.1-0.5%, NOT sklearn's ~10% default.
    assert not X.isna().any().any(), "NaNs in model matrix — fix in build_model_matrix, not here"
    if len(X) < 2:
        return np.zeros(len(X))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso.fit(X)
    raw = -iso.decision_function(X)  # sign flip: high = weird
    return rank_normalize(raw)


def fit_lof(X: pd.DataFrame, n_neighbors=20) -> np.ndarray:
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


# Rules (9 total, each mapped to a specific corner case)
#
# Per-currency thresholds, NOT a conversion rate. We don't have a real
# exchange rate in the data, and hardcoding one would go stale silently.
# Each rule gets its own number per currency instead. `None` means
# "no threshold set for this currency yet" — rows in that currency
# CANNOT trigger the rule, and get flagged via `currency_unhandled`
# below so that's visible, not confused with "checked and clean."
AMOUNT_THRESHOLDS = {
    "p2p_transfer":          {"EGP": 10_000, "USD": 200},
    "dormant_reactivation":  {"EGP": 10_000, "USD": 200},
    "new_account":           {"EGP":  5_000, "USD": 100},
    "structuring":           {"EGP": 25_000, "USD": 500},
}


def _threshold_and_gap(df: pd.DataFrame, rule_name: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns (threshold per row, unresolved mask). `unresolved` is True
    where this row's currency has no threshold defined for this rule —
    those rows are excluded from the rule (can't evaluate) rather than
    silently passing it.
    """
    table = AMOUNT_THRESHOLDS[rule_name]
    threshold = df["Currency"].map(table).astype(float)  # NaN where currency missing or value is None
    unresolved = threshold.isna()
    return threshold, unresolved


def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one boolean column per rule (rule_<name>), a comma-joined
    `rule_flags` summary column, and a separate `data_quality_flags`
    column for issues that are about the DATA being wrong, not a
    transaction being suspicious (kept separate so they don't inflate
    the fraud/mule alert queue).
    """
    df = df.copy()

    unresolved_currency_mask = pd.Series(False, index=df.index)

    # --- Rule 1: decline-then-approve probing
    rule_decline_probing = df["decline_then_approve_flag"].fillna(False).astype(bool)

    # --- Rule 2: transfer to a person ---
    threshold, unresolved = _threshold_and_gap(df, "p2p_transfer")
    unresolved_currency_mask |= unresolved
    rule_new_p2p_transfer = (
        (df["Transaction Type"] == "Transfer")
        & (df["Beneficiary Type"].isin(["Bank", "Digital Wallet"]))
        & (df["new_merchant_flag"].fillna(False).astype(bool))
        & (~unresolved)
        & (df["Transaction Amount"] > threshold)
    )

    # --- Rule 3: mule — pass-through --- (currency-independent: ratio + count, no amount)
    rule_mule_pass_through = df["pass_through_ratio"].between(0.85, 1.15) & (
        (df["distinct_senders_24h"] >= 3) | (df["distinct_recipients_24h"] >= 3)
    )

    # --- Rule 4: fraud — dormant reactivation ---
    threshold, unresolved = _threshold_and_gap(df, "dormant_reactivation")
    unresolved_currency_mask |= unresolved
    rule_dormant_reactivation = (
        (df["dormancy_days"] >= 30)
        & (df["Debit/Credit"] == "Debit")
        & (~unresolved)
        & (df["Transaction Amount"] > threshold)
    )

    # --- Rule 5: country mismatch, solid channel ---
    # Uses new_country_flag (first time THIS account used this country —
    # a behavioral change) rather than country_mismatch (vs birth
    # Nationality — permanent for any expat/foreign-national customer,
    # confirmed via testing: one real account flagged on ALL 9 of its
    # solid-channel transactions purely for having a non-Egyptian
    # nationality, which isn't a fraud signal at all).
    rule_country_mismatch = df["new_country_flag"].fillna(False).astype(bool) & (
        df["country_signal_strength"] == "solid"
    )

    # --- Rule 6: new account + large transaction ---
    threshold, unresolved = _threshold_and_gap(df, "new_account")
    unresolved_currency_mask |= unresolved
    rule_new_account_large_amount = df["is_new_account"].fillna(False).astype(bool) & (
        ~unresolved
    ) & (df["Transaction Amount"] > threshold)

    # --- Rule 7: device added shortly before use ---
    device_age = df["device_age_days"]
    rule_device_added_same_day = device_age.notna() & (device_age >= 0) & (device_age < 1)
    data_issue_negative_device_age = device_age.notna() & (device_age < 0)

    # --- Rule 8: structuring ---
    # NOTE: sum_48h_window is computed upstream in Task 2 by summing raw
    # Transaction Amount with no currency split — on an account that
    # mixes EGP and USD, this sum silently mixes units. Flag to Task 2;
    # not fixable from here since we only receive the already-summed column.
    threshold, unresolved = _threshold_and_gap(df, "structuring")
    unresolved_currency_mask |= unresolved
    rule_structuring = (
        (~unresolved) & (df["sum_48h_window"] > threshold) & (df["tx_count_24h"] >= 3)
    )

    # --- Rule 9: velocity burst ---
    rule_velocity_burst = df["tx_count_24h"] >= 5

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
    }

    for name, mask in rule_defs.items():
        df[f"rule_{name}"] = mask.fillna(False)

    # list[str] per SCHEMA.md — not a comma-joined string, since Task 4
    # needs to iterate individual rule names per row.
    rule_cols = [f"rule_{n}" for n in rule_defs]
    df["rule_flags"] = df[rule_cols].apply(
        lambda r: [n for n, v in zip(rule_defs.keys(), r) if v], axis=1
    )

    quality_issues = pd.Series([[] for _ in range(len(df))], index=df.index)
    quality_issues = quality_issues.mask(
        data_issue_negative_device_age.fillna(False),
        quality_issues.apply(lambda lst: lst + ["device_added_after_transaction"]),
    )
    quality_issues = quality_issues.mask(
        unresolved_currency_mask.fillna(False),
        quality_issues.apply(lambda lst: lst + ["currency_unhandled_for_amount_rule"]),
    )
    df["data_quality_flags"] = quality_issues
    # A fired rule is deterministic certainty (per the deck: "rules win on
    # known, explainable patterns") — it should never be outranked by a
    # model's opinion. Any row with >=1 rule fired gets pushed to at
    # least 0.95, but keeps its model score if that's already higher.
    has_rule = df["rule_flags"].apply(len) > 0
    if "combined_score" in df.columns:
        df["combined_score"] = np.where(has_rule, np.maximum(df["combined_score"], 0.95), df["combined_score"])

    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three layers. Input must be Task 2's feature-engineered output."""
    df = compute_stat_flags(df)
    df = score_models(df)
    df = compute_rule_flags(df)
    return df


if __name__ == "__main__":
    import data_layer
    import features

    raw = data_layer.validate("/mnt/user-data/uploads/Sample_Data.xlsx")
    featured = features.build(raw)
    result = score(featured)

    print("Shape:", result.shape)
    print()
    print("Stat basis distribution:", result["stat_basis"].value_counts().to_dict())
    print("mad_flag counts:", result["mad_flag"].value_counts(dropna=False).to_dict())
    print("iqr_flag counts:", result["iqr_flag"].value_counts(dropna=False).to_dict())
    print()
    print("rule_flags: rows with >=1 rule fired:", (result["rule_flags"].str.len() > 0).sum(), "/", len(result))
    fired = result[result["rule_flags"].str.len() > 0]
    if not fired.empty:
        print(fired[["Transaction ID", "rule_flags"]].to_string())
    print()
    dq_nonempty = result[result["data_quality_flags"].str.len() > 0]
    print("data_quality_flags: rows with >=1 issue:", len(dq_nonempty), "/", len(result))
    if not dq_nonempty.empty:
        print(dq_nonempty[["Transaction ID", "Currency", "data_quality_flags"]].to_string())
    print()
    print("Score columns present:", [c for c in ["iso_score", "lof_score", "combined_score"] if c in result.columns])
    print(result[["Transaction ID", "iso_score", "lof_score", "combined_score"]].sort_values("combined_score", ascending=False).head(10).to_string())