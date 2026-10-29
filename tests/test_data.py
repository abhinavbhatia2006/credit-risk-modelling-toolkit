from itertools import pairwise

import numpy as np
import pytest

from creditrisk.data import TARGET, features, synthetic, validate
from creditrisk.modeling import load_config, split_data


def test_generation_reproducible():
    assert synthetic(500, 42).equals(synthetic(500, 42))
    assert validate(synthetic(500))["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "zero_limit",
        "bad_status",
        "bad_label",
        "infinite",
        "missing_column",
        "all_missing",
    ],
)
def test_contract_rejects_bad_inputs(change):
    df = synthetic(500)
    if change == "duplicate":
        df.loc[1, "account_id"] = df.loc[0, "account_id"]
    elif change == "zero_limit":
        df.loc[0, "credit_limit"] = 0
    elif change == "bad_status":
        df.loc[0, "status_1"] = 10
    elif change == "bad_label":
        df.loc[0, TARGET] = 2
    elif change == "infinite":
        df.loc[0, "bill_1"] = np.inf
    elif change == "missing_column":
        df = df.drop(columns="payment_6")
    else:
        df["payment_6"] = np.nan
    assert not validate(df)["passed"]


def test_outcome_and_extra_columns_cannot_enter_features():
    df = synthetic(500)
    original = features(df)
    df[TARGET] = 1 - df[TARGET]
    df["future_collection_amount"] = 999999
    df["SEX"] = 1
    assert original.equals(features(df))


def test_negative_credit_balance_allowed_and_missing_reported():
    df = synthetic(500)
    df.loc[0, "bill_1"] = -100
    df.loc[1, "payment_2"] = np.nan
    result = validate(df)
    assert result["passed"] and result["warnings"]
    assert features(df).loc[0, "utilization"] == 0


def test_partition_disjointness_and_reproducibility():
    df = synthetic(1000)
    first = split_data(df, load_config())
    second = split_data(df, load_config())
    seen = set()
    for name, frame in first.items():
        ids = set(frame.account_id)
        assert not ids & seen
        assert frame.equals(second[name])
        seen |= ids
    assert len(seen) == len(df)


def test_temporal_purge_preserves_label_boundaries():
    import pandas as pd

    df = synthetic(2000)
    df["date"] = pd.date_range("2010-01-01", periods=len(df), freq="D")
    config = load_config()
    config.update(split="temporal", date_column="date", label_horizon_days=30)
    parts = split_data(df, config)
    frames = list(parts.values())
    for earlier, later in pairwise(frames):
        assert earlier.date.max() + pd.Timedelta(days=30) < later.date.min()
    assert sum(len(p) for p in frames) < len(df)


def test_temporal_rejects_single_cohort():
    df = synthetic(1000)
    df["date"] = "2025-01-01"
    config = load_config()
    config.update(split="temporal", date_column="date")
    with pytest.raises(ValueError, match="distinct dates"):
        split_data(df, config)
