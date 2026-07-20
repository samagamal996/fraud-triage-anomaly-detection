# # Data Contract — Stage by Stage

# 

# This is the exact contract each module hands to the next. Column names

# and types should not change without the whole team agreeing.

# 

# ## Stage 0 → 1: `data_layer.validate()` output

# 

# Cleaned source columns + one derived column. Raw `Date`/`Time` strings

# are dropped once `timestamp` exists.

# 

# | Column | Type | Notes |

# |---|---|---|

# | IBAN | string | account key |

# | Account Open Date | date | |

# | Account Type | category | retail / corporate |

# | Nationality | string | |

# | Transaction ID | string | unique |

# | timestamp | datetime | **derived** — merged Date + Time |

# | Channel | category | |

# | Transaction Type | category | |

# | Debit/Credit | category | |

# | Transaction Amount | float | always positive — assert, don't abs() |

# | Currency | category | |

# | Status | category | Approved / Declined — kept, not filtered |

# | Transaction Country | string | |

# | Beneficiary Type | category, nullable | null only for cash transactions |

# | Beneficiary Name | string, nullable | |

# | Beneficiary IBAN/Wallet | string, nullable | |

# | Beneficiary Country Code | string, nullable | |

# | Device ID | string, nullable | null unless Mobile/Web/ApplePay/GooglePay |

# | Device Add Date | date, nullable | |

# 

# ## Stage 1 → 2: `features.build()` output

# 

# Everything above, plus:

# 

# | Column | Type | Notes |

# |---|---|---|

# | log_amount | float | log-transformed amount |

# | amount_ratio | float | amount ÷ account's own rolling average |

# | tx_count_24h | int | velocity, trailing 24h |

# | sum_48h_window | float | rolling sum — structuring signal |

# | new_country_flag | bool | first time this account used this country |

# | country_signal_strength | category | solid / weak / weakest, by channel trust |

# | new_device_flag | bool, nullable | null where device concept doesn't apply |

# | new_merchant_flag | bool, nullable | |

# | hour_deviation | float | deviation from account's usual hour pattern |

# | declined_burst_count | int | recent declines before this transaction |

# 

# ## Stage 2 → 3: `detectors.score()` output

# 

# Everything above, plus:

# 

# | Column | Type | Notes |

# |---|---|---|

# | iso_score | float | Isolation Forest, rank-normalized 0-1 |

# | lof_score | float | LOF, rank-normalized 0-1 |

# | combined_score | float | 0-1, final ranking field |

# | rule_flags | string | which of the 5 manual rules fired, if any |

# 

# ## Stage 3 → 4: `explain.justify()` output

# 

# Everything above, plus:

# 

# | Column | Type | Notes |

# |---|---|---|

# | alert_reason | string | human-readable justification |

# | flag_source | category | rule / model / both |

# 

# This final table is what `app.py` renders as the ranked queue.

# ==========================================

## Stage 1 → 2: `features.build()` output

Everything from Stage 0→1, plus:

### Group A — core baseline

| Column                  | Type           | Notes                                                           |
| ----------------------- | -------------- | --------------------------------------------------------------- |
| log_amount              | float          | log-transformed amount                                          |
| amount_ratio            | float          | amount ÷ account's own prior rolling avg (excludes current row) |
| tx_count_24h            | int            | trailing 24h count                                              |
| sum_48h_window          | float          | rolling 48h sum — structuring signal                            |
| new_country_flag        | bool           | first time this account used this country                       |
| country_signal_strength | category       | solid / weak / weakest, by channel trust                        |
| new_device_flag         | bool, nullable | null where device concept doesn't apply                         |
| new_merchant_flag       | bool, nullable | null for cash                                                   |
| hour_deviation          | float          | deviation from account's usual hour pattern                     |

### Group B — decline-then-approve probing

| Column                    | Type  | Notes                                       |
| ------------------------- | ----- | ------------------------------------------- |
| declined_burst_count      | int   | recent declines before this row             |
| recent_decline_density    | float | declines ÷ total txns in window             |
| decline_then_approve_flag | bool  | approved row preceded by 2+ recent declines |

### Group C — new account + big transaction

| Column           | Type | Notes                         |
| ---------------- | ---- | ----------------------------- |
| account_age_days | int  | timestamp − Account Open Date |
| is_new_account   | bool | account_age_days < 30         |

### Group D — mule detection

| Column                  | Type  | Notes                                     |
| ----------------------- | ----- | ----------------------------------------- |
| inflow_sum_24h          | float | rolling sum of incoming Transfer amounts  |
| outflow_sum_24h         | float | rolling sum of outgoing Transfer amounts  |
| pass_through_ratio      | float | outflow ÷ inflow                          |
| distinct_senders_24h    | int   | fan-in                                    |
| distinct_recipients_24h | int   | fan-out                                   |
| dormancy_days           | int   | days since account's previous transaction |

### Group E — country mismatch

| Column                    | Type     | Notes                                 |
| ------------------------- | -------- | ------------------------------------- |
| country_mismatch          | bool     | Transaction Country ≠ Nationality     |
| country_mismatch_strength | category | reuses solid/weak/weakest trust label |

### Group F — device age

| Column          | Type  | Notes                       |
| --------------- | ----- | --------------------------- |
| device_age_days | float | timestamp − Device Add Date |

## Stage 2 → 3: `detectors.score()` output

Everything above, plus:

| Column         | Type           | Notes                                                                            |
| -------------- | -------------- | -------------------------------------------------------------------------------- |
| mad_flag       | bool, nullable | robust statistical outlier (per account+currency)                                |
| iqr_flag       | bool, nullable | second statistical opinion                                                       |
| stat_basis     | category       | per_account_currency / insufficient_history                                      |
| iso_score      | float          | Isolation Forest, rank-normalized 0-1                                            |
| lof_score      | float          | LOF, rank-normalized 0-1                                                         |
| combined_score | float          | 0-1, final ranking field                                                         |
| rule_flags     | list[string]   | which manual rules fired, if any — changed from single string, multiple can fire |
