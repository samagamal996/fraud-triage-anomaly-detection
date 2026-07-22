"""
Task 4 — Alert Queue + Explanation Layer.

Turns detectors.score()'s output into a ranked, human-readable alert
queue. See SCHEMA.md for the exact Stage 2->3 input contract this module
consumes.

Public contract (used by app.py):

    explain_alert(row: pd.Series) -> list[str]
        2-5 short justification strings for one scored+featured row.

    explain_queue(df, top_k=None, threshold=None, score_col="combined_score")
        -> pd.DataFrame with `reasons` (list[str]) and `reason_text`
        (str, "; "-joined) columns attached, sorted score descending.
"""

from __future__ import annotations

import pandas as pd

from config_loader import load_config

MESSAGES = load_config("messages.json")

MODEL_ONLY_PREFIX = "Flagged by the anomaly model (no rule fired) — "
RULE_PREFIX = "Rule fired — "
DATA_NOTE_PREFIX = "Data note — "

# (feature, notable_threshold, extreme_threshold, sentence template)
# Sentence templates take the row's own value; "extreme" rows are listed
# before "notable" ones so the sharpest signal leads.
_NUMERIC_SIGNALS = [
    ("amount_ratio", 3.0, 8.0, lambda v: f"amount is {v:.1f}x this account's rolling average"),
    ("hour_deviation", 3.0, 6.0, lambda v: f"occurs {v:.1f}h from this account's usual time of day"),
    ("dormancy_days", 14.0, 90.0, lambda v: f"{v:.0f} days since this account's previous transaction"),
    ("tx_count_24h", 4.0, 8.0, lambda v: f"{int(v)} transactions from this account in the last 24h"),
    ("declined_burst_count", 1.0, 3.0, lambda v: f"{int(v)} recent declines before this approval"),
    ("distinct_senders_24h", 3.0, 6.0, lambda v: f"money received from {int(v)} distinct senders in 24h"),
    ("distinct_recipients_24h", 3.0, 6.0, lambda v: f"money sent to {int(v)} distinct recipients in 24h"),
]

_FLAG_SIGNALS = [
    ("new_country_flag", lambda row: (
        f"first-ever transaction from this account in country: {row.get('Transaction Country')} "
        f"({row.get('country_signal_strength')} signal)"
    )),
    ("new_device_flag", lambda row: "device never seen on this account before"),
    ("new_merchant_flag", lambda row: f"first-ever transaction with merchant: {row.get('Beneficiary Name')}"),
]


def _severity(feature: str, value: float, notable: float, extreme: float) -> int:
    """2 = extreme, 1 = notable, 0 = below threshold — used only to order signals."""
    if value >= extreme:
        return 2
    if value >= notable:
        return 1
    return 0


def _rule_reasons(row: pd.Series) -> list[str]:
    return [
        f"{RULE_PREFIX}{MESSAGES.get(name, name.replace('_', ' '))}"
        for name in (row.get("rule_flags") or [])
    ]


def _data_quality_reasons(row: pd.Series) -> list[str]:
    return [
        f"{DATA_NOTE_PREFIX}{MESSAGES.get(issue, issue.replace('_', ' '))}"
        for issue in (row.get("data_quality_flags") or [])
    ]


def _feature_reasons(row: pd.Series, *, limit: int) -> list[str]:
    """Rank this row's own relative-feature values by severity and return
    sentences for the strongest ones — the model-only explanation path."""
    candidates: list[tuple[int, str]] = []

    for feature, notable, extreme, template in _NUMERIC_SIGNALS:
        value = row.get(feature)
        if pd.isna(value):
            continue
        sev = _severity(feature, float(value), notable, extreme)
        if sev > 0:
            candidates.append((sev, template(float(value))))

    for flag, template in _FLAG_SIGNALS:
        if row.get(flag) is True:
            candidates.append((1, template(row)))

    if row.get("mad_flag") is True or row.get("iqr_flag") is True:
        candidates.append((1, "flagged as a robust statistical outlier for this account/currency"))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [text for _, text in candidates[:limit]]


def explain_alert(row: pd.Series) -> list[str]:
    """
    2-5 human-readable justification strings for one alert row.

    Rule-fired reasons always lead (deterministic, per the deck: "rules
    win on known, explainable patterns"), followed by data-quality notes,
    then the strongest feature deviations. Rows with no rule and no
    feature clearing the "notable" bar (a model-only flag with nothing
    single-feature-obvious behind it — the hardest case to defend) fall
    back to naming which detector fired and the raw score, so the queue
    never shows a blank reason.
    """
    reasons = _rule_reasons(row) + _data_quality_reasons(row)

    remaining_slots = max(0, 5 - len(reasons))
    if remaining_slots:
        feature_reasons = _feature_reasons(row, limit=remaining_slots)
        if reasons:
            reasons.extend(feature_reasons)
        else:
            reasons.extend(f"{MODEL_ONLY_PREFIX}{text}" for text in feature_reasons)

    if not reasons:
        iso = row.get("iso_score")
        lof = row.get("lof_score")
        driver = "Isolation Forest" if (iso or 0) >= (lof or 0) else "Local Outlier Factor"
        combined = row.get("combined_score", 0.0)
        reasons.append(
            f"{MODEL_ONLY_PREFIX}no single feature stands out; {driver} ranked this row's overall "
            f"pattern as unusual for the dataset (combined_score={combined:.2f}). Needs analyst judgment."
        )

    return reasons[:5]


def build_queue(
    df: pd.DataFrame,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    score_col: str = "combined_score",
) -> pd.DataFrame:
    """
    Select and rank the alert queue. Exactly one of top_k / threshold
    should be given — matches the app's Top-K vs threshold toggle.
    """
    if (top_k is None) == (threshold is None):
        raise ValueError("Pass exactly one of top_k or threshold")

    queue = df.sort_values(score_col, ascending=False)
    if top_k is not None:
        queue = queue.head(top_k)
    else:
        queue = queue[queue[score_col] >= threshold]
    return queue


def explain_queue(
    df: pd.DataFrame,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    score_col: str = "combined_score",
) -> pd.DataFrame:
    """build_queue(), with `reasons` (list[str]) and `reason_text` (str)
    columns attached — the "ranked list of alerts, each with a
    human-readable reason string" hand-off from the team plan."""
    queue = build_queue(df, top_k=top_k, threshold=threshold, score_col=score_col).copy()
    queue["reasons"] = queue.apply(explain_alert, axis=1)
    queue["reason_text"] = queue["reasons"].apply(lambda r: "; ".join(r))
    return queue


if __name__ == "__main__":
    import data_layer
    import features
    import detectors

    raw = data_layer.validate("data/Sample_Data.xlsx")
    scored = detectors.score(features.build(raw))
    queue = explain_queue(scored, top_k=10)

    for _, alert in queue.iterrows():
        print(f"\n{alert['Transaction ID']}  score={alert['combined_score']:.2f}  rules={alert['rule_flags']}")
        for reason in alert["reasons"]:
            print(f"  - {reason}")