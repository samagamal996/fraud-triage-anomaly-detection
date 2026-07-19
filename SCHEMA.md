# Data Contract — Stage by Stage

This is the exact contract each module hands to the next. Column names
and types should not change without the whole team agreeing.

## Stage 0 → 1: `data_layer.validate()` output

Cleaned source columns + one derived column. Raw `Date`/`Time` strings
are dropped once `timestamp` exists.

| Column | Type | Notes |
|---|---|---|
| IBAN | string | account key |
| Account Open Date | date | |
| Account Type | category | retail / corporate |
| Nationality | string | |
| Transaction ID | string | unique |
| timestamp | datetime | **derived** — merged Date + Time |
| Channel | category | |
| Transaction Type | category | |
| Debit/Credit | category | |
| Transaction Amount | float | always positive — assert, don't abs() |
| Currency | category | |
| Status | category | Approved / Declined — kept, not filtered |
| Transaction Country | string | |
| Beneficiary Type | category, nullable | null only for cash transactions |
| Beneficiary Name | string, nullable | |
| Beneficiary IBAN/Wallet | string, nullable | |
| Beneficiary Country Code | string, nullable | |
| Device ID | string, nullable | null unless Mobile/Web/ApplePay/GooglePay |
| Device Add Date | date, nullable | |

## Stage 1 → 2: `features.build()` output

Everything above, plus:

| Column | Type | Notes |
|---|---|---|
| log_amount | float | log-transformed amount |
| amount_ratio | float | amount ÷ account's own rolling average |
| tx_count_24h | int | velocity, trailing 24h |
| sum_48h_window | float | rolling sum — structuring signal |
| new_country_flag | bool | first time this account used this country |
| country_signal_strength | category | solid / weak / weakest, by channel trust |
| new_device_flag | bool, nullable | null where device concept doesn't apply |
| new_merchant_flag | bool, nullable | |
| hour_deviation | float | deviation from account's usual hour pattern |
| declined_burst_count | int | recent declines before this transaction |

## Stage 2 → 3: `detectors.score()` output

Everything above, plus:

| Column | Type | Notes |
|---|---|---|
| iso_score | float | Isolation Forest, rank-normalized 0-1 |
| lof_score | float | LOF, rank-normalized 0-1 |
| combined_score | float | 0-1, final ranking field |
| rule_flags | string | which of the 5 manual rules fired, if any |

## Stage 3 → 4: `explain.justify()` output

Everything above, plus:

| Column | Type | Notes |
|---|---|---|
| alert_reason | string | human-readable justification |
| flag_source | category | rule / model / both |

This final table is what `app.py` renders as the ranked queue.
