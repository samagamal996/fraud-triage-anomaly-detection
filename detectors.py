"""
Task 3 — Detection Engine: manual rules + unsupervised models.

Branch: task3-detectors

Three independent detection layers, combined into one ranked, comparable
score, plus a per-row rule_flags string. See SCHEMA.md for the exact
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
    Grouping by (IBAN, Currency) means an account that uses both EGP and USD gets separate baselines per currency, so a
    normal-sized USD transaction never gets compared against an EGP
    baseline.

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
FX_TO_EGP = {"EGP": 1.0, "USD": 48.0}
DEFAULT_FX_RATE = 1.0  # unknown currency falls back to 1:1 — documented assumption
 
# EGP-equivalent thresholds, one per rule that needs one.
THRESHOLD_P2P_TRANSFER = 10_000
THRESHOLD_DORMANT_REACTIVATION = 10_000
THRESHOLD_NEW_ACCOUNT = 5_000
THRESHOLD_STRUCTURING = 25_000
 
 
def compute_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one boolean column per rule (rule_<name>), a comma-joined
    `rule_flags` summary column, and a separate `data_quality_flags`
    column for issues that are about the DATA being wrong, not a
    transaction being suspicious (kept separate so they don't inflate
    the fraud/mule alert queue).
 
    Input must be Task 2's feature-engineered output — needs
    decline_then_approve_flag, new_merchant_flag, pass_through_ratio,
    distinct_senders_24h/distinct_recipients_24h, dormancy_days,
    country_mismatch/country_mismatch_strength, is_new_account,
    device_age_days, sum_48h_window, tx_count_24h.
    """
    df = df.copy()
 
    # EGP-equivalent amount, for threshold comparisons only.
    fx_rate = df["Currency"].map(FX_TO_EGP).fillna(DEFAULT_FX_RATE)
    amount_egp = df["Transaction Amount"] * fx_rate
 
    # --- Rule 1: decline-then-approve probing
    rule_decline_probing = df["decline_then_approve_flag"].fillna(False).astype(bool)
 
    # --- Rule 2: transfer to a person ---
    # First-time transfer to an individual (not a merchant), above a
    # floor amount — a first-time small P2P transfer is usually nothing.
    rule_new_p2p_transfer = (
        (df["Transaction Type"] == "Transfer")
        & (df["Beneficiary Type"].isin(["Bank", "Digital Wallet"]))
        & (df["new_merchant_flag"].fillna(0).astype(bool))
        & (amount_egp > THRESHOLD_P2P_TRANSFER)
    )
 
    # --- Rule 3: mule — pass-through ---
    # Currency-independent by design — it's a ratio + counterparty count,
    # not an amount. .between() returns False on NaN automatically, so
    # accounts with no transfer activity in the window are excluded
    # without an extra null check.
    rule_mule_pass_through = df["pass_through_ratio"].between(0.85, 1.15) & (
        (df["distinct_senders_24h"] >= 3) | (df["distinct_recipients_24h"] >= 3)
    )
 
    # --- Rule 4: fraud — dormant reactivation ---
    # A long-quiet account suddenly sending a large amount out. NaN
    # dormancy_days (a first-ever transaction) evaluates to False here,
    # correctly — a first transaction isn't a "reactivation."
    rule_dormant_reactivation = (
        (df["dormancy_days"] >= 30)
        & (df["Debit/Credit"] == "Debit")
        & (amount_egp > THRESHOLD_DORMANT_REACTIVATION)
    )
 
    # --- Rule 5: country mismatch, solid channel ---
    # Only trusted when the location evidence is strong (card-present /
    # terminal-based). Weak/online mismatches are left to the models —
    # too spoofable to hard-flag on.
    rule_country_mismatch = df["country_mismatch"].fillna(False).astype(bool) & (
        df["country_mismatch_strength"] == "solid"
    )
 
    # --- Rule 6: new account + large transaction ---
    # Direction-agnostic, flat threshold — simpler and easier to defend
    # in one sentence than a ratio-based version.
    rule_new_account_large_amount = df["is_new_account"].fillna(False).astype(bool) & (
        amount_egp > THRESHOLD_NEW_ACCOUNT
    )
 
    # --- Rule 7: device added shortly before use (the "most important column") ---
    # Same-calendar-day device + transaction. Negative device_age_days
    # (device "added" AFTER the transaction) is a DATA problem, not a
    # fraud signal 
    device_age = df["device_age_days"]
    rule_device_added_same_day = device_age.notna() & (device_age >= 0) & (device_age < 1)
    data_issue_negative_device_age = device_age.notna() & (device_age < 0)
 
    # --- Rule 8: structuring ---
    rule_structuring = (df["sum_48h_window"] > THRESHOLD_STRUCTURING) & (df["tx_count_24h"] >= 3)
 
    # --- Rule 9: velocity burst ---
    # Flat threshold, no per-account baseline — documented limitation,
    # not a bug. Will often co-fire with Rule 8 since both use
    # tx_count_24h; that's expected, not redundant.
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
 
    df["rule_flags"] = df[[f"rule_{n}" for n in rule_defs]].apply(
        lambda r: ",".join(n for n, v in zip(rule_defs.keys(), r) if v), axis=1
    )
 
    df["data_quality_flags"] = np.where(
        data_issue_negative_device_age.fillna(False), "device_added_after_transaction", ""
    )
 
    return df
 
 
def score(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three layers. Input must be Task 2's feature-engineered output."""
    df = compute_stat_flags(df)
    df = score_models(df)
    df = compute_rule_flags(df)
    return df
 
 
if __name__ == "__main__":
    import features
 
    raw = pd.read_excel("data/Sample_Data.xlsx")
    raw["timestamp"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw["Account Open Date"] = pd.to_datetime(raw["Account Open Date"])
    raw["Device Add Date"] = pd.to_datetime(raw["Device Add Date"], errors="coerce")
 
    featured = features.build(raw)
    result = score(featured)
 
    print("Shape:", result.shape)
    print()
    print("Stat basis distribution:", result["stat_basis"].value_counts().to_dict())
    print("mad_flag counts:", result["mad_flag"].value_counts(dropna=False).to_dict())
    print()
    print("rule_flags distribution:")
    print(result["rule_flags"].value_counts())
    print()
    print("data_quality_flags distribution:")
    print(result["data_quality_flags"].value_counts())
    print()
    print("Score columns present:", [c for c in ["iso_score", "lof_score", "combined_score"] if c in result.columns])


