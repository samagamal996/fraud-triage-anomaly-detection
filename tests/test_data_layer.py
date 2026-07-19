"""
Smoke tests for data_layer.py. Add real assertions once validate() is implemented.

Run: pytest tests/
"""

import pandas as pd
import pytest

import data_layer


def test_rejects_missing_column():
    """A file missing a required column should raise ValidationError, not crash."""
    # TODO
    pass


def test_accepts_valid_sample():
    """The real Sample_Data.xlsx should validate cleanly with 77 rows."""
    # TODO: df = data_layer.validate("data/Sample_Data.xlsx")
    # assert len(df) == 77
    pass


def test_beneficiary_null_only_for_cash():
    """Beneficiary fields should be null iff Transaction Type is Cash Withdrawal/Deposit."""
    # TODO
    pass


def test_device_id_null_pattern():
    """Device ID should be null unless Channel/Transaction Type implies a device."""
    # TODO
    pass


def test_amount_always_positive():
    """Transaction Amount should never be negative or zero after validation."""
    # TODO
    pass
