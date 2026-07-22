from data_layer import validate
from features import build
from detectors import score

df = validate(r"data\Sample_Data.xlsx")
# df = validate(r"data\Synthetic_Fraud_Test_Data.xlsx")
df = build(df)
df = score(df)

rules = [
    "decline_probing",
    "new_p2p_transfer",
    "mule_pass_through",
    "dormant_reactivation",
    "country_mismatch_solid",
    "new_account_large_amount",
    "device_added_same_day",
    "structuring",
    "velocity_burst",
]

print("\nRule counts")
print("-" * 40)

for rule in rules:
    print(f"{rule:30} {df[f'rule_{rule}'].sum()}")

print(df.shape)
print()
print(df[[
    "Transaction ID",
    "combined_score",
    "rule_flags",
    "iso_score",
    "lof_score"
]].sort_values("combined_score", ascending=False).head(10))