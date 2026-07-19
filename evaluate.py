"""
Task 6 — Evaluation: answer key, precision@k, fresh-data testing.

Owner: [fill in name]
Branch: task6-evaluation

Not part of the live pipeline — run separately to check the pipeline's
quality before Demo Day.
"""

import pandas as pd


def precision_at_k(df: pd.DataFrame, answer_key: dict, k: int = 50) -> float:
    """
    Args:
        df: fully scored dataframe (output of explain.justify())
        answer_key: dict mapping Transaction ID -> True (fraud) / False (benign)
        k: queue size to evaluate against

    Returns:
        precision@k as a float (real fraud found / k)
    """
    top_k = df.nlargest(k, "combined_score")
    hits = top_k["Transaction ID"].map(answer_key).fillna(False)
    return hits.mean()


def build_answer_key(df: pd.DataFrame) -> dict:
    """
    TODO (Task 6): manually review a sample of transactions and label a
    handful as known-fraud and known-benign-but-weird, mirroring the
    deck's "14 planted cases" pattern. Store as a dict or CSV of
    Transaction ID -> True/False.
    """
    raise NotImplementedError


def run_fresh_data_test(pipeline_fn, unseen_file_path: str) -> pd.DataFrame:
    """
    Run the full pipeline on a file it wasn't built/tuned against, to
    simulate the mentor's Demo Day fresh-data test.
    """
    raise NotImplementedError
