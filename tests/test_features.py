from data_layer import validate
from features import build

df = validate(r"data\Sample_Data.xlsx")
df = validate(r"data\fraud-triage-anomaly-detection/data/Synthetic_Fraud_Test_Data.xlsx")
features = build(df)

print(features.shape)
print(features.columns)
print(features.head())
