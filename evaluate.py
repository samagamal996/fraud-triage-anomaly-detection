"""
Task 6 — Evaluation, Fresh-Data Test & Defense Prep.

Three things a fraud-ops manager will actually ask for on Demo Day:
  1. Precision@k against your own planted answer key, before the mentor's.
  2. A "fresh, unseen data" run — same pipeline, a file it wasn't tuned on.
  3. A grilling rehearsal — pick N random alerts, force yourselves to
     answer "why is this in my queue?" using explain.py's own output.

Answer key format (CSV or XLSX), one row per transaction you're planting
a verdict for — not every row in the file, just the ones you've judged:

    Transaction ID,is_fraud,notes
    TX0000061,1,planted: payroll-account beneficiary swap
    TX0000024,0,benign but weird: regular taxi rider, just a new merchant

`is_fraud` accepts 1/0, true/false, yes/no (case-insensitive).
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

import data_layer
import features
import detectors
import explain


def run_fresh_data_test(file_path: str) -> pd.DataFrame:
    """Full pipeline on a file the team hasn't tuned against — this is
    the exact call the app makes, so a result here is a genuine preview
    of what Demo Day's live run will produce."""
    validated = data_layer.validate(file_path)
    featured = features.build(validated)
    return detectors.score(featured)


def load_answer_key(path: str) -> pd.DataFrame:
    """Returns a dataframe indexed by Transaction ID with an `is_fraud`
    bool column (and `notes` if present)."""
    if str(path).lower().endswith(".csv"):
        key = pd.read_csv(path)
    else:
        key = pd.read_excel(path)

    missing = {"Transaction ID", "is_fraud"} - set(key.columns)
    if missing:
        raise ValueError(f"Answer key is missing columns: {sorted(missing)}")

    truthy = {"1", "true", "yes", "y", "fraud"}
    key["is_fraud"] = (
        key["is_fraud"].astype(str).str.strip().str.casefold().isin(truthy)
    )
    return key.set_index("Transaction ID")


def precision_at_k(
    scored_df: pd.DataFrame,
    answer_key: pd.DataFrame,
    k: int = 50,
    score_col: str = "combined_score",
) -> dict:
    """
    Precision@k against the planted answer key. Rows not present in the
    answer key are excluded from both the numerator and denominator
    (we haven't judged them, so they can't count as hits or misses) —
    reported separately as `unlabeled_in_top_k` so it's visible how much
    of the queue this key actually covers.
    """
    queue = scored_df.sort_values(score_col, ascending=False).head(k)
    labeled = queue.set_index("Transaction ID").join(answer_key[["is_fraud"]], how="left")

    covered = labeled["is_fraud"].notna()
    hits = int((labeled["is_fraud"] == True).sum())  # noqa: E712
    covered_count = int(covered.sum())
    unlabeled = int((~covered).sum())

    return {
        "k": k,
        "hits": hits,
        "covered_in_top_k": covered_count,
        "unlabeled_in_top_k": unlabeled,
        "precision_at_k_of_covered": hits / covered_count if covered_count else float("nan"),
        "precision_at_k_of_k": hits / k,
        "total_planted_frauds": int(answer_key["is_fraud"].sum()),
        "planted_frauds_caught": hits,
        "recall_of_planted": hits / int(answer_key["is_fraud"].sum()) if answer_key["is_fraud"].sum() else float("nan"),
    }


def rule_vs_model_breakdown(
    scored_df: pd.DataFrame, answer_key: pd.DataFrame, score_col: str = "combined_score"
) -> pd.DataFrame:
    """
    For planted frauds only: did a rule catch it, did the model alone
    catch it (high score, no rule), or was it missed entirely? Feeds
    directly into "one rule you'd now rewrite" — the question the deck
    says to have a real answer for.
    """
    fraud_ids = answer_key[answer_key["is_fraud"]].index
    subset = scored_df[scored_df["Transaction ID"].isin(fraud_ids)].copy()
    subset["caught_by_rule"] = subset["rule_flags"].apply(len) > 0
    subset["caught_by_model_only"] = (~subset["caught_by_rule"]) & (subset[score_col] >= 0.5)
    subset["missed"] = (~subset["caught_by_rule"]) & (~subset["caught_by_model_only"])

    cols = ["Transaction ID", score_col, "rule_flags", "caught_by_rule", "caught_by_model_only", "missed"]
    return subset[cols].sort_values(score_col, ascending=False)


def grilling_rehearsal(scored_df: pd.DataFrame, n: int = 5, seed: int | None = None, top_k: int = 50) -> None:
    """
    Prints n random alerts from the top_k queue with their explain.py
    reasons, formatted as the mentor will ask it: "why is this in my
    queue?" Rehearse out loud before Demo Day, not just read silently.
    """
    rng = random.Random(seed)
    queue = explain.explain_queue(scored_df, top_k=top_k)
    sample_size = min(n, len(queue))
    sample = queue.sample(n=sample_size, random_state=rng.randint(0, 2**31))

    for _, alert in sample.iterrows():
        print(f"\n{'=' * 60}")
        print(f"Why is {alert['Transaction ID']} in my queue?")
        print(f"{'=' * 60}")
        print(f"  Account:      {alert['IBAN']}")
        print(f"  Amount:       {alert['Transaction Amount']:.2f} {alert['Currency']}")
        print(f"  Score:        {alert['combined_score']:.2f}")
        for reason in alert["reasons"]:
            print(f"  - {reason}")


def _print_report(metrics: dict) -> None:
    print("\n--- Precision@k ---")
    print(f"  k = {metrics['k']}")
    print(f"  hits = {metrics['hits']} / {metrics['covered_in_top_k']} labeled rows in top-{metrics['k']}"
          f" ({metrics['unlabeled_in_top_k']} rows in top-{metrics['k']} have no planted label)")
    print(f"  precision@k (of labeled rows only) = {metrics['precision_at_k_of_covered']:.1%}")
    print(f"  precision@k (of all k, unlabeled counted as miss) = {metrics['precision_at_k_of_k']:.1%}")
    print(f"  recall of planted frauds = {metrics['planted_frauds_caught']}/{metrics['total_planted_frauds']}"
          f" ({metrics['recall_of_planted']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6 — evaluation & defense prep")
    parser.add_argument("file", help="Transactions file to score (CSV/XLSX)")
    parser.add_argument("--answer-key", help="CSV/XLSX with Transaction ID, is_fraud columns")
    parser.add_argument("--k", type=int, default=50, help="Precision@k (default 50)")
    parser.add_argument("--score-col", default="combined_score")
    parser.add_argument("--grill", type=int, default=0, help="Also print N random alerts for rehearsal")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Running fresh-data test on {args.file} ...")
    scored = run_fresh_data_test(args.file)
    print(f"Scored {len(scored)} transactions.")

    if args.answer_key:
        key = load_answer_key(args.answer_key)
        metrics = precision_at_k(scored, key, k=args.k, score_col=args.score_col)
        _print_report(metrics)

        breakdown = rule_vs_model_breakdown(scored, key, score_col=args.score_col)
        if not breakdown.empty:
            print("\n--- Planted frauds: rule vs model breakdown ---")
            print(breakdown.to_string(index=False))
            missed = breakdown[breakdown["missed"]]
            if not missed.empty:
                print(f"\n{len(missed)} planted fraud(s) missed entirely — start 'one rule you'd rewrite' here:")
                print(missed[["Transaction ID"]].to_string(index=False))
    else:
        print("No --answer-key given — skipping precision@k. Pass one to score against your planted cases.")

    if args.grill:
        grilling_rehearsal(scored, n=args.grill, seed=args.seed, top_k=args.k)


if __name__ == "__main__":
    main()