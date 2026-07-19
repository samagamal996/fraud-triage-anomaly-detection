"""
Task 5 — Streamlit App: file upload, controls, alert queue, feedback loop.

Owner: [fill in name]
Branch: task5-app

The only file that imports Streamlit. Wires together data_layer,
features, detectors, and explain. Can be scaffolded with mock data
before the other modules are finished — swap in real calls as they land.
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import data_layer
import features
import detectors
import explain

FEEDBACK_FILE = "data/feedback.jsonl"

st.set_page_config(page_title="Fraud Alert Triage", layout="wide")
st.title("Fraud Alert Triage")

if "df" not in st.session_state:
    st.session_state.df = None

uploaded = st.file_uploader(
    "Upload transactions file",
    type=["csv", "xlsx"],
    help="Expects the 20-column production schema — see README.md / SCHEMA.md",
)

if uploaded is not None:
    try:
        raw_df = data_layer.validate(uploaded)
        st.session_state.df = run_pipeline(raw_df)
        st.success(f"Loaded {len(st.session_state.df)} transactions.")
    except data_layer.ValidationError as e:
        st.error(f"File rejected: {e}")

if st.session_state.df is not None:
    df = st.session_state.df

    mode = st.selectbox("Detector mode", ["Combined", "Rules only", "Models only"])
    view = st.radio("View", ["Top-K", "Threshold"], horizontal=True)

    if view == "Top-K":
        k = st.slider("K", 10, 200, 50)
        queue = df.nlargest(k, "combined_score")
    else:
        thresh = st.slider("Score threshold", 0.0, 1.0, 0.8)
        queue = df[df["combined_score"] >= thresh].sort_values(
            "combined_score", ascending=False
        )

    st.caption(f"{len(queue)} alerts in queue")

    for _, row in queue.iterrows():
        with st.expander(
            f"{row['Transaction ID']} — score {row['combined_score']:.2f} "
            f"({row['flag_source']})"
        ):
            st.write(row["alert_reason"])

            c1, c2, c3 = st.columns(3)
            tx_id = row["Transaction ID"]
            if c1.button("Fraud", key=f"fraud_{tx_id}"):
                _save_feedback(tx_id, "fraud")
                st.toast(f"Saved: {tx_id} -> fraud")
            if c2.button("Benign", key=f"benign_{tx_id}"):
                _save_feedback(tx_id, "benign")
                st.toast(f"Saved: {tx_id} -> benign")
            if c3.button("Unsure", key=f"unsure_{tx_id}"):
                _save_feedback(tx_id, "unsure")
                st.toast(f"Saved: {tx_id} -> unsure")

            # TODO: account history chart for context
            # account_history = df[df["IBAN"] == row["IBAN"]]
            # st.line_chart(account_history.set_index("timestamp")["Transaction Amount"])


@st.cache_data
def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run features -> detectors -> explain. Cached so re-renders don't refit models."""
    df = features.build(raw_df)
    df = detectors.score(df)
    df = explain.justify(df)
    return df


def _save_feedback(transaction_id: str, label: str) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    record = {
        "transaction_id": transaction_id,
        "label": label,
        "timestamp": datetime.now().isoformat(),
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
