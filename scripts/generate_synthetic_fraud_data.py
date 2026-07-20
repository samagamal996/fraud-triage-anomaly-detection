"""Generate a deterministic, schema-compatible fraud evaluation workbook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260720
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "Synthetic_Fraud_Test_Data.xlsx"
ANSWER_KEY_PATH = DATA_DIR / "Synthetic_Fraud_Test_Answer_Key.csv"
SAMPLE_PATH = DATA_DIR / "Sample_Data.xlsx"
SAMPLE_ANSWER_KEY_PATH = DATA_DIR / "Sample_Data_Answer_Key.csv"

SOURCE_COLUMNS = [
    "IBAN",
    "Account Open Date",
    "Account Type",
    "Nationality",
    "Transaction ID",
    "Date",
    "Time",
    "Channel",
    "Transaction Type",
    "Debit/Credit",
    "Transaction Amount",
    "Currency",
    "Status",
    "Transaction Country",
    "Beneficiary Type",
    "Beneficiary Name",
    "Beneficiary IBAN/Wallet",
    "Beneficiary Country Code",
    "Device ID",
    "Device Add Date",
]

CASH_TYPES = {"Cash Withdrawal", "Cash Deposit"}
DEVICE_TYPES = {
    "POS (Apple Pay)",
    "POS (Google Pay)",
    "E-Commerce (Apple Pay)",
    "E-Commerce (Google Pay)",
}
DEVICE_CHANNELS = {"Mobile App", "Web App"}

VALID_COMBINATIONS = {
    ("Cash Deposit", "ATM", "Credit"),
    ("Cash Deposit", "Branch", "Credit"),
    ("Cash Withdrawal", "ATM", "Debit"),
    ("E-Commerce", "Debit Card", "Debit"),
    ("E-Commerce", "VISA Card", "Debit"),
    ("E-Commerce (Apple Pay)", "Debit Card", "Debit"),
    ("E-Commerce (Apple Pay)", "VISA Card", "Debit"),
    ("E-Commerce (Google Pay)", "Debit Card", "Debit"),
    ("E-Commerce (Google Pay)", "VISA Card", "Debit"),
    ("POS", "Debit Card", "Debit"),
    ("POS", "VISA Card", "Debit"),
    ("POS (Apple Pay)", "Debit Card", "Debit"),
    ("POS (Apple Pay)", "VISA Card", "Debit"),
    ("POS (Google Pay)", "Debit Card", "Debit"),
    ("POS (Google Pay)", "VISA Card", "Debit"),
    ("Transfer", "Branch", "Credit"),
    ("Transfer", "Branch", "Debit"),
    ("Transfer", "Mobile App", "Debit"),
    ("Transfer", "Web App", "Debit"),
}

MERCHANTS = [
    ("Fresh Market", "EG-MRC-10001"),
    ("Nile Pharmacy", "EG-MRC-10002"),
    ("Cairo Telecom", "EG-MRC-10003"),
    ("Delta Electronics", "EG-MRC-10004"),
    ("Metro Fuel", "EG-MRC-10005"),
    ("Lotus Fashion", "EG-MRC-10006"),
    ("City Books", "EG-MRC-10007"),
    ("Home Supplies", "EG-MRC-10008"),
]

BANKS = [
    ("National Bank of Egypt", "EG120001000000000000000001"),
    ("Banque Misr", "EG120002000000000000000002"),
    ("Commercial International Bank", "EG120003000000000000000003"),
    ("AlexBank", "EG120004000000000000000004"),
    ("QNB Alahli", "EG120005000000000000000005"),
]


def _account_iban(account_number: int) -> str:
    return f"EG{30 + account_number:02d}{account_number + 1:04d}{account_number + 1:021d}"


def _device_id(account_number: int, suffix: str = "A") -> str:
    return f"DEV-SYN-{account_number + 1:02d}-{suffix}"


def _normal_row(
    rng: np.random.Generator,
    account_number: int,
    account_open: pd.Timestamp,
    timestamp: pd.Timestamp,
) -> dict[str, object]:
    account_type = "corporate" if account_number % 5 == 0 else "retail"
    transaction_kind = rng.choice(
        ["pos", "wallet_pos", "ecommerce", "cash", "transfer"],
        p=[0.34, 0.12, 0.16, 0.18, 0.20],
    )
    merchant_name, merchant_iban = MERCHANTS[int(rng.integers(len(MERCHANTS)))]
    bank_name, bank_iban = BANKS[int(rng.integers(len(BANKS)))]
    device_added = max(account_open + pd.Timedelta(days=14), pd.Timestamp("2025-01-15"))

    row: dict[str, object] = {
        "IBAN": _account_iban(account_number),
        "Account Open Date": account_open.strftime("%Y-%m-%d"),
        "Account Type": account_type,
        "Nationality": "EG",
        "Transaction ID": "",
        "_timestamp": timestamp,
        "Channel": "",
        "Transaction Type": "",
        "Debit/Credit": "Debit",
        "Transaction Amount": 0.0,
        "Currency": "EGP",
        "Status": "Declined" if rng.random() < 0.025 else "Approved",
        "Transaction Country": "EG",
        "Beneficiary Type": None,
        "Beneficiary Name": None,
        "Beneficiary IBAN/Wallet": None,
        "Beneficiary Country Code": None,
        "Device ID": None,
        "Device Add Date": None,
        "_is_fraud": False,
        "_fraud_scenario": "",
    }

    if transaction_kind == "pos":
        row.update(
            {
                "Channel": rng.choice(["Debit Card", "VISA Card"]),
                "Transaction Type": "POS",
                "Transaction Amount": round(float(rng.lognormal(5.5, 0.55)), 2),
                "Beneficiary Type": "Merchant",
                "Beneficiary Name": merchant_name,
                "Beneficiary IBAN/Wallet": merchant_iban,
                "Beneficiary Country Code": "EG",
            }
        )
    elif transaction_kind == "wallet_pos":
        wallet = rng.choice(["Apple Pay", "Google Pay"])
        row.update(
            {
                "Channel": rng.choice(["Debit Card", "VISA Card"]),
                "Transaction Type": f"POS ({wallet})",
                "Transaction Amount": round(float(rng.lognormal(5.35, 0.50)), 2),
                "Beneficiary Type": "Merchant",
                "Beneficiary Name": merchant_name,
                "Beneficiary IBAN/Wallet": merchant_iban,
                "Beneficiary Country Code": "EG",
                "Device ID": _device_id(account_number),
                "Device Add Date": device_added.strftime("%Y-%m-%d"),
            }
        )
    elif transaction_kind == "ecommerce":
        row.update(
            {
                "Channel": rng.choice(["Debit Card", "VISA Card"]),
                "Transaction Type": "E-Commerce",
                "Transaction Amount": round(float(rng.lognormal(5.8, 0.65)), 2),
                "Beneficiary Type": "Merchant",
                "Beneficiary Name": merchant_name,
                "Beneficiary IBAN/Wallet": merchant_iban,
                "Beneficiary Country Code": "EG",
            }
        )
    elif transaction_kind == "cash":
        deposit = bool(rng.random() < 0.25)
        row.update(
            {
                "Channel": rng.choice(["ATM", "Branch"]) if deposit else "ATM",
                "Transaction Type": "Cash Deposit" if deposit else "Cash Withdrawal",
                "Debit/Credit": "Credit" if deposit else "Debit",
                "Transaction Amount": float(rng.choice([500, 1000, 1500, 2000, 3000])),
            }
        )
    else:
        incoming = bool(rng.random() < 0.18)
        row.update(
            {
                "Channel": (
                    "Branch"
                    if incoming
                    else rng.choice(["Branch", "Mobile App", "Web App"], p=[0.2, 0.65, 0.15])
                ),
                "Transaction Type": "Transfer",
                "Debit/Credit": "Credit" if incoming else "Debit",
                "Transaction Amount": round(
                    float(rng.lognormal(8.25 if account_type == "corporate" else 7.65, 0.55)),
                    2,
                ),
                "Beneficiary Type": "Bank",
                "Beneficiary Name": bank_name,
                "Beneficiary IBAN/Wallet": bank_iban,
                "Beneficiary Country Code": "EG",
            }
        )
        if row["Channel"] in DEVICE_CHANNELS:
            row["Device ID"] = _device_id(account_number)
            row["Device Add Date"] = device_added.strftime("%Y-%m-%d")

    return row


def _set_transfer(
    row: dict[str, object],
    *,
    amount: float,
    channel: str,
    direction: str = "Debit",
    country: str = "EG",
    counterparty: str = "EG990009000000000000000009",
    counterparty_name: str = "External Counterparty",
    device_id: str | None = None,
    device_date: str | None = None,
) -> None:
    row.update(
        {
            "Channel": channel,
            "Transaction Type": "Transfer",
            "Debit/Credit": direction,
            "Transaction Amount": amount,
            "Status": "Approved",
            "Transaction Country": country,
            "Beneficiary Type": "Bank",
            "Beneficiary Name": counterparty_name,
            "Beneficiary IBAN/Wallet": counterparty,
            "Beneficiary Country Code": country,
            "Device ID": device_id if channel in DEVICE_CHANNELS else None,
            "Device Add Date": device_date if channel in DEVICE_CHANNELS else None,
        }
    )


def _set_wallet_purchase(
    row: dict[str, object],
    *,
    amount: float,
    wallet: str,
    transaction_type: str,
    country: str,
    merchant_name: str,
    merchant_id: str,
    device_id: str,
    device_date: str,
) -> None:
    row.update(
        {
            "Channel": "VISA Card",
            "Transaction Type": f"{transaction_type} ({wallet})",
            "Debit/Credit": "Debit",
            "Transaction Amount": amount,
            "Status": "Approved",
            "Transaction Country": country,
            "Beneficiary Type": "Merchant",
            "Beneficiary Name": merchant_name,
            "Beneficiary IBAN/Wallet": merchant_id,
            "Beneficiary Country Code": country,
            "Device ID": device_id,
            "Device Add Date": device_date,
        }
    )


def _mark_fraud(row: dict[str, object], scenario: str) -> None:
    row["_is_fraud"] = True
    row["_fraud_scenario"] = scenario


def _plant_fraud_patterns(accounts: list[list[dict[str, object]]]) -> None:
    # 1. High-value foreign transfer from a brand-new device.
    target = accounts[0][-1]
    date = pd.Timestamp(target["_timestamp"]).strftime("%Y-%m-%d")
    _set_transfer(
        target,
        amount=185_000.00,
        channel="Mobile App",
        country="US",
        counterparty="US64SYNTHETIC000000001",
        counterparty_name="Atlantic Holdings",
        device_id="DEV-FRAUD-FOREIGN-01",
        device_date=date,
    )
    _mark_fraud(target, "large_foreign_transfer_new_device")

    # 2. Approved wallet purchase immediately after two declined probes.
    target = accounts[1][-1]
    base_time = pd.Timestamp(target["_timestamp"])
    for offset, previous in zip([18, 7], accounts[1][-3:-1]):
        previous["_timestamp"] = base_time - pd.Timedelta(minutes=offset)
        _set_wallet_purchase(
            previous,
            amount=19.99,
            wallet="Google Pay",
            transaction_type="E-Commerce",
            country="EG",
            merchant_name="Online Verification",
            merchant_id="EG-MRC-PROBE",
            device_id="DEV-PROBE-02",
            device_date=(base_time - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        previous["Status"] = "Declined"
    _set_wallet_purchase(
        target,
        amount=8_750.00,
        wallet="Google Pay",
        transaction_type="E-Commerce",
        country="US",
        merchant_name="Global Digital Store",
        merchant_id="US-MRC-00002",
        device_id="DEV-PROBE-02",
        device_date=(base_time - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    _mark_fraud(target, "decline_then_approve_card_testing")

    # 3. Rapid fan-in followed by near-complete pass-through.
    target = accounts[2][-1]
    base_time = pd.Timestamp(target["_timestamp"])
    for sender, previous in enumerate(accounts[2][-5:-1], start=1):
        previous["_timestamp"] = base_time - pd.Timedelta(hours=10 - sender * 2)
        _set_transfer(
            previous,
            amount=25_000.00,
            channel="Branch",
            direction="Credit",
            counterparty=f"EG77SENDER{sender:014d}",
            counterparty_name=f"Sender {sender}",
        )
    _set_transfer(
        target,
        amount=96_500.00,
        channel="Mobile App",
        counterparty="EG88RECIPIENT00000000001",
        counterparty_name="Rapid Settlement",
        device_id=_device_id(2),
        device_date="2025-03-01",
    )
    _mark_fraud(target, "mule_fan_in_pass_through")

    # 4. Dormant-account takeover.
    target = accounts[3][-1]
    target["_timestamp"] = pd.Timestamp(accounts[3][-2]["_timestamp"]) + pd.Timedelta(days=62)
    date = pd.Timestamp(target["_timestamp"]).strftime("%Y-%m-%d")
    _set_transfer(
        target,
        amount=72_000.00,
        channel="Web App",
        country="GB",
        counterparty="GB29SYNTHETIC000000004",
        counterparty_name="London Trade Services",
        device_id="DEV-DORMANT-04",
        device_date=date,
    )
    _mark_fraud(target, "dormant_account_takeover")

    # 5. Very new account making a large transfer.
    target = accounts[4][-1]
    base_time = pd.Timestamp("2026-06-24 11:18:00")
    account_open = base_time - pd.Timedelta(days=5)
    for index, row in enumerate(accounts[4]):
        row["Account Open Date"] = account_open.strftime("%Y-%m-%d")
        row["_timestamp"] = account_open + pd.Timedelta(hours=8 + index * 7)
    target["_timestamp"] = base_time
    _set_transfer(
        target,
        amount=130_000.00,
        channel="Mobile App",
        counterparty="EG55NEWACCOUNT00000000005",
        counterparty_name="New Beneficiary Five",
        device_id="DEV-NEW-ACCOUNT-05",
        device_date=account_open.strftime("%Y-%m-%d"),
    )
    _mark_fraud(target, "new_account_large_transfer")

    # 6. Foreign e-commerce purchase from a just-added device.
    target = accounts[5][-1]
    date = pd.Timestamp(target["_timestamp"]).strftime("%Y-%m-%d")
    _set_wallet_purchase(
        target,
        amount=24_900.00,
        wallet="Apple Pay",
        transaction_type="E-Commerce",
        country="US",
        merchant_name="Overseas Luxury Goods",
        merchant_id="US-MRC-00006",
        device_id="DEV-FOREIGN-ECOM-06",
        device_date=date,
    )
    _mark_fraud(target, "foreign_ecommerce_new_device")

    # 7. Repeated just-under-threshold transfers within 36 hours.
    target = accounts[6][-1]
    base_time = pd.Timestamp(target["_timestamp"])
    for number, row in enumerate(accounts[6][-6:]):
        row["_timestamp"] = base_time - pd.Timedelta(hours=(5 - number) * 6)
        _set_transfer(
            row,
            amount=9_250.00 if row is not target else 9_850.00,
            channel="Mobile App",
            counterparty=f"EG66STRUCTURE{number:013d}",
            counterparty_name=f"Structured Recipient {number + 1}",
            device_id=_device_id(6),
            device_date="2025-02-01",
        )
    _mark_fraud(target, "structured_transfer_sequence")

    # 8. Large transfer at an hour far outside this account's history.
    target = accounts[7][-1]
    target["_timestamp"] = pd.Timestamp(target["_timestamp"]).normalize() + pd.Timedelta(
        hours=3, minutes=7
    )
    _set_transfer(
        target,
        amount=83_000.00,
        channel="Web App",
        counterparty="EG44NIGHTTRANSFER000000008",
        counterparty_name="Night Settlement",
        device_id=_device_id(7),
        device_date="2025-01-15",
    )
    _mark_fraud(target, "unusual_hour_large_transfer")

    # 9. Sudden fan-out to several new recipients.
    target = accounts[8][-1]
    base_time = pd.Timestamp(target["_timestamp"])
    for recipient, row in enumerate(accounts[8][-5:], start=1):
        row["_timestamp"] = base_time - pd.Timedelta(hours=(5 - recipient) * 3)
        _set_transfer(
            row,
            amount=18_500.00,
            channel="Mobile App",
            counterparty=f"EG33FANOUT{recipient:016d}",
            counterparty_name=f"Fanout Recipient {recipient}",
            device_id=_device_id(8),
            device_date="2025-04-10",
        )
    _mark_fraud(target, "mule_fan_out")

    # 10. High-value wallet purchase on a device added the same day.
    target = accounts[9][-1]
    date = pd.Timestamp(target["_timestamp"]).strftime("%Y-%m-%d")
    _set_wallet_purchase(
        target,
        amount=36_500.00,
        wallet="Google Pay",
        transaction_type="POS",
        country="EG",
        merchant_name="Premium Electronics",
        merchant_id="EG-MRC-FRAUD-10",
        device_id="DEV-SAME-DAY-10",
        device_date=date,
    )
    _mark_fraud(target, "same_day_device_large_purchase")


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    accounts: list[list[dict[str, object]]] = []

    for account_number in range(20):
        account_open = pd.Timestamp("2017-01-10") + pd.Timedelta(
            days=account_number * 117
        )
        usual_hour = 9 + account_number % 10
        account_rows: list[dict[str, object]] = []
        for transaction_number in range(15):
            day = 1 + transaction_number * 2 + account_number % 2
            minute_offset = int(rng.integers(-40, 41))
            timestamp = (
                pd.Timestamp("2026-06-01")
                + pd.Timedelta(days=day - 1, hours=usual_hour, minutes=minute_offset)
            )
            account_rows.append(
                _normal_row(rng, account_number, account_open, timestamp)
            )
        accounts.append(account_rows)

    _plant_fraud_patterns(accounts)
    rows = [row for account in accounts for row in account]
    internal = pd.DataFrame(rows).sort_values(
        ["_timestamp", "IBAN"], kind="mergesort"
    ).reset_index(drop=True)
    internal["Transaction ID"] = [
        f"SYN2026-{number:06d}" for number in range(1, len(internal) + 1)
    ]
    internal["Date"] = internal["_timestamp"].dt.strftime("%d/%m/%Y")
    internal["Time"] = internal["_timestamp"].dt.strftime("%H:%M:%S")

    transactions = internal[SOURCE_COLUMNS].copy()
    answer_key = internal[
        ["Transaction ID", "_is_fraud", "_fraud_scenario"]
    ].rename(
        columns={
            "_is_fraud": "is_fraud",
            "_fraud_scenario": "fraud_scenario",
        }
    )
    return transactions, answer_key


def validate_source(df: pd.DataFrame, *, expected_rows: int) -> None:
    if list(df.columns) != SOURCE_COLUMNS:
        raise AssertionError("Source columns or their order do not match the contract")
    if len(df) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(df)}")
    if df["Transaction ID"].isna().any() or not df["Transaction ID"].is_unique:
        raise AssertionError("Transaction IDs must be present and unique")
    if df["IBAN"].isna().any():
        raise AssertionError("IBAN must not be null")
    if not pd.to_numeric(df["Transaction Amount"], errors="coerce").gt(0).all():
        raise AssertionError("Transaction Amount must always be positive")
    if not df["Account Type"].isin(["retail", "corporate"]).all():
        raise AssertionError("Invalid Account Type")
    if not df["Status"].isin(["Approved", "Declined"]).all():
        raise AssertionError("Invalid Status")

    combinations = set(
        df[["Transaction Type", "Channel", "Debit/Credit"]]
        .itertuples(index=False, name=None)
    )
    invalid_combinations = combinations.difference(VALID_COMBINATIONS)
    if invalid_combinations:
        raise AssertionError(f"Invalid transaction combinations: {invalid_combinations}")

    cash = df["Transaction Type"].isin(CASH_TYPES)
    beneficiary_columns = [
        "Beneficiary Type",
        "Beneficiary Name",
        "Beneficiary IBAN/Wallet",
        "Beneficiary Country Code",
    ]
    if not df.loc[cash, beneficiary_columns].isna().all().all():
        raise AssertionError("Cash rows must not contain beneficiary data")
    if not df.loc[~cash, beneficiary_columns].notna().all().all():
        raise AssertionError("Non-cash rows must contain beneficiary data")

    device_applies = df["Channel"].isin(DEVICE_CHANNELS) | df[
        "Transaction Type"
    ].isin(DEVICE_TYPES)
    if not df.loc[device_applies, ["Device ID", "Device Add Date"]].notna().all().all():
        raise AssertionError("Device-based rows require device ID and add date")
    if not df.loc[~device_applies, ["Device ID", "Device Add Date"]].isna().all().all():
        raise AssertionError("Device data is present where the device concept does not apply")

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    account_open = pd.to_datetime(
        df["Account Open Date"], format="%Y-%m-%d", errors="raise"
    )
    if (timestamp < account_open).any():
        raise AssertionError("A transaction predates its account")
    device_added = pd.to_datetime(
        df["Device Add Date"], format="%Y-%m-%d", errors="coerce"
    )
    if (device_added[device_applies] > timestamp[device_applies]).any():
        raise AssertionError("A transaction predates its device")


def main() -> None:
    sample = pd.read_excel(SAMPLE_PATH)
    validate_source(sample, expected_rows=77)
    sample_answer_key = pd.DataFrame(
        {
            "Transaction ID": sample["Transaction ID"],
            "is_fraud": False,
            "fraud_scenario": "",
        }
    )

    transactions, answer_key = generate()
    validate_source(transactions, expected_rows=300)
    if int(answer_key["is_fraud"].sum()) != 10:
        raise AssertionError("The answer key must contain exactly 10 fraud rows")

    transactions.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
    answer_key.to_csv(ANSWER_KEY_PATH, index=False)
    sample_answer_key.to_csv(SAMPLE_ANSWER_KEY_PATH, index=False)

    # Read the written artifacts back; this catches Excel type/format drift.
    validate_source(pd.read_excel(OUTPUT_PATH), expected_rows=300)
    persisted_key = pd.read_csv(ANSWER_KEY_PATH)
    if len(persisted_key) != 300 or int(persisted_key["is_fraud"].sum()) != 10:
        raise AssertionError("Persisted answer key does not contain 300 rows / 10 frauds")
    persisted_sample_key = pd.read_csv(SAMPLE_ANSWER_KEY_PATH)
    if (
        len(persisted_sample_key) != 77
        or int(persisted_sample_key["is_fraud"].sum()) != 0
    ):
        raise AssertionError("Sample answer key does not contain 77 rows / 0 frauds")

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}: 300 rows")
    print(f"Wrote {ANSWER_KEY_PATH.relative_to(ROOT)}: exactly 10 fraud labels")
    print(f"Wrote {SAMPLE_ANSWER_KEY_PATH.relative_to(ROOT)}: exactly 0 fraud labels")


if __name__ == "__main__":
    main()
