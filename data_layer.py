"""
Task 1 — Data Layer: schema validation and cleaning.

Owner: [fill in name]
Branch: task1-data-layer

Turns a raw transactions CSV/XLSX into a clean, validated dataframe that
every downstream module can trust. See SCHEMA.md for the exact output
contract.
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "IBAN", "Account Open Date", "Account Type", "Nationality",
    "Transaction ID", "Date", "Time", "Channel", "Transaction Type",
    "Debit/Credit", "Transaction Amount", "Currency", "Status",
    "Transaction Country", "Beneficiary Type", "Beneficiary Name",
    "Beneficiary IBAN/Wallet", "Beneficiary Country Code",
    "Device ID", "Device Add Date",
]

# The 9 valid combinations from the mentor's schema slide.
# TODO: encode as actual validation logic, not just a placeholder.
VALID_TRANSACTION_TYPES = [
    "Transfer", "POS", "POS (Apple Pay)", "POS (Google Pay)",
    "E-Commerce", "E-Commerce (Apple Pay)", "E-Commerce (Google Pay)",
    "Cash Withdrawal", "Cash Deposit",
]


class ValidationError(Exception):
    """Raised when a file or row fails schema validation."""
    pass


def validate(file) -> pd.DataFrame:
    """
    Load and validate a raw transactions file.

    Args:
        file: path or file-like object (CSV or XLSX)

    Returns:
        Cleaned dataframe matching the Stage 1 contract in SCHEMA.md

    Raises:
        ValidationError: with a specific, human-readable reason if the
        file or any row doesn't match the expected schema.
    """
    # TODO:
    # 1. Load file (handle both .csv and .xlsx)
    # 2. Check all REQUIRED_COLUMNS are present
    # 3. Merge Date + Time into `timestamp` (confirm DD/MM/YYYY vs MM/DD/YYYY
    #    with mentor before parsing — see README open question)
    # 4. Validate each row against the 9-combination matrix
    #    (Transaction Type x Channel x Beneficiary Type x Device ID x Debit/Credit)
    # 5. Validate null patterns are as expected (not flagged as errors):
    #    - Beneficiary fields null only for Cash Withdrawal/Deposit
    #    - Device ID null unless Mobile App/Web App/Apple Pay/Google Pay
    # 6. Assert Transaction Amount > 0 for all rows
    # 7. Keep Status == "Declined" rows — do not filter them out
    raise NotImplementedError


def _merge_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Date + Time columns into a single `timestamp` datetime column."""
    raise NotImplementedError


def _check_valid_combination(row) -> bool:
    """Check a single row against the 9 valid Transaction Type combinations."""
    raise NotImplementedError
