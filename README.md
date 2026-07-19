# Fraud Triage — Anomaly Detection Project

A ranked, explainable fraud alert triage app. Raw transactions CSV in →
validated data → behavioral features → anomaly scores (rules + models) →
ranked queue with plain-English explanations → analyst feedback loop.

## Pipeline

```
Raw CSV → data_layer.py → features.py → detectors.py → explain.py → app.py (Streamlit)
```

Each stage is a pure-Python module with no UI code. `app.py` is the only
file that imports Streamlit — it wires the other four together.

## Module ownership

| File | Owns | Branch |
|---|---|---|
| `data_layer.py` | Task 1 — schema validation, cleaning | `task1-data-layer` |
| `features.py` | Task 2 — per-account behavioral features | `task2-features` |
| `detectors.py` | Task 3 — rules + IsoForest + LOF | `task3-detectors` |
| `explain.py` | Task 4 — alert justification text | `task4-explain` |
| `app.py` | Task 5 — Streamlit app shell + feedback loop | `task5-app` |
| `evaluate.py`, `tests/` | Task 6 — answer key, precision@k, fresh-data test | `task6-evaluation` |

## Branch workflow

1. Branch off `main`: `git checkout -b task2-features`
2. Work only in your owned file(s) — don't edit someone else's module
3. Commit early and often
4. When your module runs cleanly against the schema below (no errors,
   basic smoke test passes), open a PR into `main`
5. Self-merge once it's green — don't block on waiting for review given
   the timeline, but leave the PR open so others can see the diff
6. Pull `main` before you branch, so you're building on the latest
   merged work from upstream modules

## Data contract (locked — do not change column names without telling the group)

Source columns (from `Sample_Data.xlsx`, confirmed against the mentor's schema slide):

```
IBAN, Account Open Date, Account Type, Nationality, Transaction ID,
Date, Time, Channel, Transaction Type, Debit/Credit, Transaction Amount,
Currency, Status, Transaction Country, Beneficiary Type, Beneficiary Name,
Beneficiary IBAN/Wallet, Beneficiary Country Code, Device ID, Device Add Date
```

**Open question — not yet confirmed with mentor:** `Date` column looks like
DD/MM/YYYY (e.g. `01/06/2026`) based on the sample. Confirm before Task 1
parses it, since getting it backwards silently corrupts every time-based
feature.

### Stage-by-stage output schema

See `SCHEMA.md` for the full column-by-column contract at each pipeline stage.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest tests/
```
