"""
Task 5 — App Shell + Analyst Feedback Loop (Streamlit).

Pipeline: raw file -> data_layer.validate -> features.build -> detectors.score
                    -> (explain.explain_alert | _fallback_explain) -> queue + feedback

------------------------------------------------------------------------------
CONTRACT FOR TASK 4 (explain.py) — nothing else in this file needs to change
once this lands:

    def explain_alert(row: pd.Series) -> list[str]:
        '''Return 2-5 short, human-readable justification strings for one
        scored+featured transaction row (a row from detectors.score()'s
        output). E.g.:
            ["amount is 8.2x this account's 30-day average",
             "first ever transaction in country: GB",
             "device never seen on this account"]
        '''

The app already imports `explain` if the module exists and calls
`explain.explain_alert(row)` for every alert. Until then it falls back to
a simple built-in explainer (`_fallback_explain`, below) so the queue is
never blank on Demo Day. Ping the team if `explain.py` needs a different
row shape than what `detectors.score()` currently outputs.
------------------------------------------------------------------------------

FRAUD LIKELIHOOD %: shown per-alert, derived directly from the selected
score (combined_score / iso_score / lof_score depending on the sidebar
choice), rescaled to 0-100%. This is NOT a statistically calibrated
probability — we have no labeled historical data to calibrate against.
It's a direct rescaling of the same evidence already driving the ranking
and the reasons shown underneath it: when a rule fires, combined_score
is floored at 0.95 (detectors.py), so a rule-fired alert always reads
as >=95%; otherwise it reflects the models' relative anomaly ranking.
Worth saying exactly this if asked in the defense — it's an honest
reflection of the same signal, not an independently trained classifier.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import data_layer
import features
import detectors
from config_loader import load_config

try:
    import explain as _explain_module  # Task 4, teammate's module
except ImportError:
    _explain_module = None

MESSAGES = load_config("messages.json")
FEEDBACK_PATH = Path("data/analyst_feedback.csv")
FEEDBACK_COLUMNS = ["session_id", "file_name", "transaction_id", "verdict", "recorded_at"]

DISPLAY_COLUMNS = [
    "Transaction ID", "IBAN", "timestamp", "Transaction Amount", "Currency",
    "Channel", "Transaction Type", "Transaction Country", "Status",
]

STAT_COLUMNS = ["stat_basis", "mad_flag", "iqr_flag"]


# ----------------------------------------------------------------------------
# Fraud likelihood % — direct rescaling of the ranking score (see module
# docstring above for the honesty caveat on what this number does/doesn't mean)
# ----------------------------------------------------------------------------

def fraud_likelihood_pct(row: pd.Series, score_col: str) -> float:
    value = row.get(score_col)
    if pd.isna(value):
        return 0.0
    return round(float(value) * 100, 1)


# ----------------------------------------------------------------------------
# Explanation layer (real module if present, otherwise a built-in fallback)
# ----------------------------------------------------------------------------

def _fmt_pct_x(ratio: float) -> str:
    return f"{ratio:.1f}x"


def _fallback_explain(row: pd.Series) -> list[str]:
    """Feature-driven justification, used only until explain.py exists."""
    reasons: list[str] = []

    for rule_name in row.get("rule_flags", []) or []:
        text = MESSAGES.get(rule_name, rule_name.replace("_", " "))
        reasons.append(f"Rule fired — {text}")

    for issue in row.get("data_quality_flags", []) or []:
        text = MESSAGES.get(issue, issue.replace("_", " "))
        reasons.append(f"Data note — {text}")

    if not reasons or len(reasons) < 4:
        amount_ratio = row.get("amount_ratio")
        if pd.notna(amount_ratio) and amount_ratio >= 2:
            reasons.append(
                f"amount is {_fmt_pct_x(amount_ratio)} this account's rolling average"
            )
        if row.get("new_country_flag") is True:
            reasons.append(
                f"first-ever transaction from this account in country: {row.get('Transaction Country')} "
                f"({row.get('country_signal_strength')} signal)"
            )
        if row.get("new_device_flag") is True:
            reasons.append("device never seen on this account before")
        if row.get("new_merchant_flag") is True:
            reasons.append(f"first-ever transaction with merchant: {row.get('Beneficiary Name')}")
        hour_dev = row.get("hour_deviation")
        if pd.notna(hour_dev) and hour_dev >= 4:
            reasons.append(f"occurs {hour_dev:.1f}h from this account's usual time of day")
        dormancy = row.get("dormancy_days")
        if pd.notna(dormancy) and dormancy >= 30:
            reasons.append(f"{dormancy:.0f} days since this account's previous transaction")
        tx_count = row.get("tx_count_24h")
        if pd.notna(tx_count) and tx_count >= 4:
            reasons.append(f"{int(tx_count)} transactions from this account in the last 24h")
        if row.get("stat_basis") == "insufficient_history":
            reasons.append("note: fewer than 5 prior transactions — statistical baseline not yet established")

    if not reasons:
        score = row.get("combined_score", 0.0)
        reasons.append(f"statistical outlier relative to typical account behavior (score={score:.2f})")

    return reasons[:5]


def explain_row(row: pd.Series) -> list[str]:
    if _explain_module is not None and hasattr(_explain_module, "explain_alert"):
        try:
            result = _explain_module.explain_alert(row)
            if result:
                return result
        except Exception as exc:  # noqa: BLE001 — never let a broken explainer blank the queue
            return [f"(explain.py raised {type(exc).__name__}: {exc}) — showing fallback below", *_fallback_explain(row)]
    return _fallback_explain(row)


# ----------------------------------------------------------------------------
# Feedback persistence — keyed by (session_id, transaction_id) so a second
# fresh file mid-demo can't collide with or overwrite the first file's verdicts.
# ----------------------------------------------------------------------------

def _ensure_feedback_file() -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(FEEDBACK_COLUMNS)


def load_feedback() -> pd.DataFrame:
    _ensure_feedback_file()
    return pd.read_csv(FEEDBACK_PATH, dtype=str)


def save_feedback(session_id: str, file_name: str, transaction_id: str, verdict: str) -> None:
    _ensure_feedback_file()
    existing = load_feedback()
    mask = (existing["session_id"] == session_id) & (existing["transaction_id"] == transaction_id)
    existing = existing[~mask]  # upsert: drop any prior verdict for this key, then append the new one
    new_row = pd.DataFrame([{
        "session_id": session_id,
        "file_name": file_name,
        "transaction_id": transaction_id,
        "verdict": verdict,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }])
    pd.concat([existing, new_row], ignore_index=True).to_csv(FEEDBACK_PATH, index=False)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def _session_id_for(file_bytes: bytes, file_name: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    return f"{file_name}-{digest}"


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    buffer.name = file_name  # data_layer._load() branches on .name
    validated = data_layer.validate(buffer)
    featured = features.build(validated)
    scored = detectors.score(featured)
    return scored


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Fraud Triage", layout="wide")
st.title("Fraud Triage — Alert Queue")

with st.sidebar:
    st.header("1. Upload transactions")
    with st.expander("Expected file schema (20 columns)", expanded=False):
        st.write(
            "CSV or XLSX with exactly these columns. Dates are `DD/MM/YYYY`, "
            "Time is `HH:MM:SS`. Amount is always positive — direction lives "
            "in Debit/Credit. Beneficiary block is empty for cash "
            "(withdrawal/deposit); Device ID/Device Add Date are only "
            "present for Mobile/Web transfers and Apple Pay/Google Pay "
            "POS & E-Commerce. Supported currencies: EGP, USD."
        )
        st.code("\n".join(data_layer.REQUIRED_COLUMNS), language=None)

    uploaded = st.file_uploader("Transactions file", type=["csv", "xlsx"])

    scored_df = None
    session_id = None
    file_name = None

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        session_id = _session_id_for(file_bytes, file_name)
        try:
            scored_df = run_pipeline(file_bytes, file_name)
        except data_layer.ValidationError as exc:
            st.error(f"File rejected — {exc}")
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the app
            st.error(f"Unexpected error while processing this file: {exc}")

    if scored_df is not None:
        st.success(f"{len(scored_df)} transactions loaded and scored.")
        st.header("2. Detector settings")

        score_source = st.radio(
            "Score to rank by",
            ["combined_score", "iso_score", "lof_score"],
            format_func=lambda s: {
                "combined_score": "Combined (rules + models)",
                "iso_score": "Isolation Forest only",
                "lof_score": "LOF only",
            }[s],
        )

        selection_mode = st.radio("Queue selection", ["Top-K", "Score threshold"])
        if selection_mode == "Top-K":
            top_k = st.number_input("K (alerts to review)", min_value=1, max_value=len(scored_df), value=min(50, len(scored_df)))
            threshold = None
        else:
            threshold = st.slider("Minimum score", 0.0, 1.0, 0.5, 0.01)
            top_k = None

        rules_only = st.checkbox("Show only rows where a rule fired", value=False)

st.divider()

if scored_df is None:
    st.info("Upload a transactions file in the sidebar to build the alert queue.")
    st.stop()

# ---- Build the queue -------------------------------------------------------

queue = scored_df.copy()
if rules_only:
    queue = queue[queue["rule_flags"].apply(len) > 0]

queue = queue.sort_values(score_source, ascending=False)
if selection_mode == "Top-K":
    queue = queue.head(int(top_k))
else:
    queue = queue[queue[score_source] >= threshold]

# ---- KPI row ----------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions scored", len(scored_df))
k2.metric("In queue", len(queue))
k3.metric("Rule-fired (whole file)", int((scored_df["rule_flags"].apply(len) > 0).sum()))
k4.metric("Insufficient history rows", int((scored_df["stat_basis"] == "insufficient_history").sum()))

# ---- Queue table --------------------------------------------------------

st.subheader(f"Ranked alert queue ({len(queue)})")
table = queue[DISPLAY_COLUMNS + [score_source, "rule_flags"] + STAT_COLUMNS].copy()
table.insert(
    table.columns.get_loc(score_source) + 1,
    "fraud_likelihood_%",
    queue.apply(lambda r: fraud_likelihood_pct(r, score_source), axis=1),
)
table["rule_flags"] = table["rule_flags"].apply(lambda flags: ", ".join(flags) if flags else "—")
table["mad_flag"] = table["mad_flag"].map({True: "outlier", False: "normal", pd.NA: "n/a"})
table["iqr_flag"] = table["iqr_flag"].map({True: "outlier", False: "normal", pd.NA: "n/a"})
table = table.rename(columns={score_source: "score", "mad_flag": "MAD", "iqr_flag": "IQR", "stat_basis": "stat basis"})
st.dataframe(table, use_container_width=True, hide_index=True)

# ---- Alert detail -----------------------------------------------------------

st.subheader("Alert detail")
if queue.empty:
    st.warning("No transactions match the current queue settings.")
    st.stop()

selected_id = st.selectbox("Select a Transaction ID to inspect", queue["Transaction ID"].tolist())
row = scored_df.loc[scored_df["Transaction ID"] == selected_id].iloc[0]

left, right = st.columns([1, 1])

with left:
    st.markdown(f"**{row['Transaction ID']}** — score `{row[score_source]:.2f}`")
    st.write(
        f"{row['Transaction Amount']:.2f} {row['Currency']} · {row['Transaction Type']} "
        f"via {row['Channel']} · {row['timestamp']}"
    )
    st.write(f"Account `{row['IBAN']}` ({row['Account Type']}, opened {row['Account Open Date'].date()})")

    pct = fraud_likelihood_pct(row, score_source)
    st.metric("Estimated fraud likelihood", f"{pct:.0f}%")
    st.caption(
        "Derived directly from the score and reasons below — not an independently "
        "calibrated probability (no labeled historical data to calibrate against). "
        "Rule-fired alerts are floored at 95%."
    )

    if row["stat_basis"] == "insufficient_history":
        stat_line = "insufficient history for this account/currency — no statistical baseline yet"
    else:
        mad = "outlier" if row["mad_flag"] else "normal"
        iqr = "outlier" if row["iqr_flag"] else "normal"
        stat_line = f"MAD: **{mad}** · IQR: **{iqr}** (per-account, per-currency baseline)"
    st.caption(f"Day 1 statistical check — {stat_line}")

    st.markdown("**Why this is in the queue:**")
    for reason in explain_row(row):
        st.markdown(f"- {reason}")

    st.markdown("**Analyst verdict**")
    fb1, fb2, fb3 = st.columns(3)
    if fb1.button("🚩 Fraud", use_container_width=True):
        save_feedback(session_id, file_name, selected_id, "fraud")
        st.toast("Saved: fraud")
    if fb2.button("✅ Benign", use_container_width=True):
        save_feedback(session_id, file_name, selected_id, "benign")
        st.toast("Saved: benign")
    if fb3.button("❔ Unsure", use_container_width=True):
        save_feedback(session_id, file_name, selected_id, "unsure")
        st.toast("Saved: unsure")

    existing_feedback = load_feedback()
    prior = existing_feedback[
        (existing_feedback["session_id"] == session_id)
        & (existing_feedback["transaction_id"] == selected_id)
    ]
    if not prior.empty:
        st.caption(f"Current saved verdict: **{prior.iloc[-1]['verdict']}**")

with right:
    st.markdown(f"**Recent history — account `{row['IBAN']}`**")
    history = scored_df[scored_df["IBAN"] == row["IBAN"]].sort_values("timestamp")
    chart_data = history.set_index("timestamp")[["Transaction Amount"]]
    st.line_chart(chart_data)
    st.dataframe(
        history[DISPLAY_COLUMNS + [score_source]].rename(columns={score_source: "score"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
with st.expander("Analyst feedback log (this session's file)"):
    fb = load_feedback()
    fb = fb[fb["session_id"] == session_id] if session_id else fb.iloc[0:0]
    st.dataframe(fb, use_container_width=True, hide_index=True)