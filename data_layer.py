"""
Task 1 — Data Layer: schema validation and cleaning.

Branch: task1-data-layer

Turns a raw transactions CSV/XLSX into a clean, validated dataframe that
every downstream module (features.py, detectors.py) can trust. See
SCHEMA.md for the exact Stage 0 -> 1 output contract.
"""


from __future__ import annotations

import pandas as pd
from config_loader import load_config
SCHEMA = load_config("schema.json")

REQUIRED_COLUMNS = [
    "IBAN", "Account Open Date", "Account Type", "Nationality",
    "Transaction ID", "Date", "Time", "Channel", "Transaction Type",
    "Debit/Credit", "Transaction Amount", "Currency", "Status",
    "Transaction Country", "Beneficiary Type", "Beneficiary Name",
    "Beneficiary IBAN/Wallet", "Beneficiary Country Code",
    "Device ID", "Device Add Date",
]

# Confirmed against the sample file itself (day values up to 30 appear
# with month always <=12) -- DD/MM/YYYY, not inferred from convention.
DATE_FORMAT = "%d/%m/%Y"
TIME_FORMAT = "%H:%M:%S"

VALID_ACCOUNT_TYPES = set(SCHEMA["account_types"])
VALID_CURRENCIES = set(SCHEMA["currencies"])
VALID_STATUSES = set(SCHEMA["statuses"])
VALID_DEBIT_CREDIT = set(SCHEMA["debit_credit"])

# The 9 valid (Transaction Type -> Channel, Beneficiary Type, Device ID
# requirement, Debit/Credit) combinations from the mentor's schema slide.
# device_required: True = must have Device ID, False = must NOT have one.
# beneficiary_types: None = must be null (cash has no counterparty).
VALID_COMBINATIONS = {
    "Transfer": {
        "channels": {"Mobile App", "Web App", "Branch"},
        "beneficiary_types": {"Bank", "Digital Wallet"},
        "device_required": None,  # Mobile/Web: yes, Branch: no -- checked separately below
    },
    "POS": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": False,
    },
    "POS (Apple Pay)": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": True,
    },
    "POS (Google Pay)": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": True,
    },
    "E-Commerce": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": False,
    },
    "E-Commerce (Apple Pay)": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": True,
    },
    "E-Commerce (Google Pay)": {
        "channels": {"Debit Card", "VISA Card"},
        "beneficiary_types": {"Merchant"},
        "device_required": True,
    },
    "Cash Withdrawal": {
        "channels": {"ATM", "Branch"},
        "beneficiary_types": None,
        "device_required": False,
    },
    "Cash Deposit": {
        "channels": {"ATM", "Branch"},
        "beneficiary_types": None,
        "device_required": False,
    },
}


class ValidationError(Exception):
    """Raised when a file or row fails schema validation."""
    pass

def _check_duplicate_transaction_ids(df: pd.DataFrame) -> None:
    duplicates = df[df["Transaction ID"].duplicated(keep=False)]

    if not duplicates.empty:
        ids = duplicates["Transaction ID"].unique()

        raise ValidationError(
            "Duplicate Transaction IDs found:\n"
            + "\n".join(map(str, ids[:20]))
        )
    
def validate(file) -> pd.DataFrame:
    """
    Load and validate a raw transactions file.

    Args:
        file: path or file-like object (CSV or XLSX)

    Returns:
        Cleaned dataframe matching the Stage 0->1 contract in SCHEMA.md
        (timestamp merged, Date/Time dropped, dtypes fixed, rows sorted).

    Raises:
        ValidationError: with a specific, human-readable reason if the
        file or any row doesn't match the expected schema.
    """
    df = _load(file)
    _check_required_columns(df)
    _check_duplicate_transaction_ids(df)
    df = _merge_timestamp(df)
    df = _fix_dtypes(df)
    _check_enums(df)
    _check_row_combinations(df)
    _check_null_patterns(df)
    _check_amount(df)

    df = df.sort_values(["IBAN", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def _load(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")


def _merge_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Date + Time into a single `timestamp` datetime column."""
    df = df.copy()
    combined = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    timestamp = pd.to_datetime(
        combined, format=f"{DATE_FORMAT} {TIME_FORMAT}", errors="coerce"
    )
    if timestamp.isna().any():
        bad_rows = df.index[timestamp.isna()].tolist()
        raise ValidationError(
            f"Could not parse Date/Time as {DATE_FORMAT} {TIME_FORMAT} on rows: {bad_rows[:10]}"
        )
    df["timestamp"] = timestamp
    return df.drop(columns=["Date", "Time"])


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Account Open Date"] = pd.to_datetime(
    df["Account Open Date"],
    format="%Y-%m-%d",
    errors="coerce"
    )

    if df["Account Open Date"].isna().any():
        raise ValidationError("Account Open Date contains unparseable values")

    # Device Add Date is genuinely nullable (only applies to some channels) --
    # NaT here is a correct, expected value, not an error.
    df["Device Add Date"] = pd.to_datetime(
        df["Device Add Date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    today = pd.Timestamp.now().normalize()

    future_open = df["Account Open Date"] > today

    if future_open.any():
        raise ValidationError(
            f"Future Account Open Date found on rows "
            f"{df.index[future_open].tolist()[:10]}"
        )

    future_device = (
        df["Device Add Date"].notna()
        & (df["Device Add Date"] > today)
    )

    if future_device.any():
        raise ValidationError(
            f"Future Device Add Date found on rows "
            f"{df.index[future_device].tolist()[:10]}"
        )

    for col in [
        "Account Type", "Channel", "Transaction Type", "Debit/Credit",
        "Currency", "Status", "Beneficiary Type",
    ]:
        df[col] = df[col].astype("string").str.strip()

    return df


def _check_enums(df: pd.DataFrame) -> None:

    checks = {
        "Account Type": VALID_ACCOUNT_TYPES,
        "Currency": VALID_CURRENCIES,
        "Status": VALID_STATUSES,
        "Debit/Credit": VALID_DEBIT_CREDIT,
    }

    for column, allowed in checks.items():

        invalid = ~df[column].isin(allowed)

        if invalid.any():

            values = (
                df.loc[invalid, column]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValidationError(
                f"{column} contains unsupported values: "
                f"{values}\n"
                f"Allowed: {sorted(allowed)}"
            )


def _check_row_combinations(df: pd.DataFrame) -> None:
    """Validate each row against the 9-combination matrix."""
    bad_type = ~df["Transaction Type"].isin(VALID_COMBINATIONS.keys())
    if bad_type.any():
        raise ValidationError(
            f"Unknown Transaction Type on rows: {df.index[bad_type].tolist()[:10]} "
            f"-> {df.loc[bad_type, 'Transaction Type'].unique().tolist()}"
        )

    errors = []
    for txn_type, spec in VALID_COMBINATIONS.items():
        subset = df[df["Transaction Type"] == txn_type]
        if subset.empty:
            continue

        bad_channel = ~subset["Channel"].isin(spec["channels"])
        if bad_channel.any():
            errors.append(
                f"{txn_type}: invalid Channel on rows {subset.index[bad_channel].tolist()[:5]} "
                f"-> {subset.loc[bad_channel, 'Channel'].unique().tolist()}"
            )

        if spec["beneficiary_types"] is None:
            bad_bene = subset["Beneficiary Type"].notna()
            if bad_bene.any():
                errors.append(
                    f"{txn_type}: Beneficiary Type must be null (cash has no counterparty), "
                    f"rows {subset.index[bad_bene].tolist()[:5]}"
                )
        else:
            bad_bene = ~subset["Beneficiary Type"].isin(spec["beneficiary_types"])
            if bad_bene.any():
                errors.append(
                    f"{txn_type}: invalid Beneficiary Type on rows {subset.index[bad_bene].tolist()[:5]} "
                    f"-> {subset.loc[bad_bene, 'Beneficiary Type'].unique().tolist()}"
                )

        if spec["device_required"] is True:
            bad_device = subset["Device ID"].isna()
            if bad_device.any():
                errors.append(
                    f"{txn_type}: Device ID required (the phone IS the device), "
                    f"rows {subset.index[bad_device].tolist()[:5]}"
                )
        elif spec["device_required"] is False:
            bad_device = subset["Device ID"].notna()
            if bad_device.any():
                errors.append(
                    f"{txn_type}: Device ID must be null (card-only, not phone-based), "
                    f"rows {subset.index[bad_device].tolist()[:5]}"
                )
        else:
            # Transfer: Mobile/Web -> device required, Branch -> device must be null
            remote = subset["Channel"].isin(["Mobile App", "Web App"])
            bad_remote = remote & subset["Device ID"].isna()
            bad_branch = (~remote) & subset["Device ID"].notna()
            if bad_remote.any():
                errors.append(
                    f"Transfer: Device ID required for Mobile/Web App, "
                    f"rows {subset.index[bad_remote].tolist()[:5]}"
                )
            if bad_branch.any():
                errors.append(
                    f"Transfer: Device ID must be null for Branch, "
                    f"rows {subset.index[bad_branch].tolist()[:5]}"
                )

    if errors:
        raise ValidationError("Row combination validation failed:\n" + "\n".join(errors))


def _check_null_patterns(df: pd.DataFrame) -> None:
    """
    Confirm null patterns match the schema's expectations (documentation
    check -- the combination check above already enforces the specifics;
    this catches Beneficiary Name/IBAN drifting independently of Type).
    """
    cash = df["Transaction Type"].isin(["Cash Withdrawal", "Cash Deposit"])
    leaked_name = cash & df["Beneficiary Name"].notna()
    leaked_iban = cash & df["Beneficiary IBAN/Wallet"].notna()
    if leaked_name.any() or leaked_iban.any():
        raise ValidationError(
            "Cash transactions must not carry Beneficiary Name/IBAN "
            f"(rows: {df.index[leaked_name | leaked_iban].tolist()[:10]})"
        )


def _check_amount(df: pd.DataFrame) -> None:
    amount = pd.to_numeric(df["Transaction Amount"], errors="coerce")
    if amount.isna().any() or (amount <= 0).any():
        bad = df.index[amount.isna() | (amount <= 0)].tolist()
        raise ValidationError(
            f"Transaction Amount must be positive numeric on all rows (bad rows: {bad[:10]})"
        )
    # Status == "Declined" is intentionally left in place here -- not
    # filtered -- it's a real signal (Rule 1: decline-then-approve probing).


if __name__ == "__main__":
    df = validate(r"data\Sample_Data.xlsx")    
    print("Validated shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Status counts:", df["Status"].value_counts().to_dict())
    print(df.dtypes)