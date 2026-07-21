from data_layer import validate
from features import build
from detectors import score

df = validate(r"data\Sample_Data.xlsx")
# df = validate(r"data\Synthetic_Fraud_Test_Data.xlsx")
df = build(df)
df = score(df)

print(df.shape)
print()
print(df[[
    "Transaction ID",
    "combined_score",
    "rule_flags",
    "iso_score",
    "lof_score"
]].sort_values("combined_score", ascending=False).head(10))