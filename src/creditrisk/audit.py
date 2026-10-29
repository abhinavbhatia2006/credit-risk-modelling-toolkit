"""Recompute report totals independently using SQLite and pandas."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID, TARGET
from .io import write_json

SQL = """SELECT COUNT(*) AS accounts,
COUNT(DISTINCT account_id) AS unique_accounts,
SUM(probability_of_default) AS expected_defaults,
SUM(exposure_proxy) AS total_exposure,
SUM(probability_of_default * exposure_proxy) AS pd_weighted_exposure,
SUM(flag_for_review) AS review_flags
FROM predictions"""


def reconcile(predictions, out):
    df = pd.read_csv(predictions, dtype={ID: "string"})
    required = [ID, "probability_of_default", "exposure_proxy", "flag_for_review"]
    if set(required) - set(df):
        raise ValueError("Prediction file is missing required audit columns")
    if df.empty or df[required].isna().any().any() or df[ID].duplicated().any():
        raise ValueError("Predictions must be nonempty, complete and unique by account")
    if (
        not df.probability_of_default.between(0, 1).all()
        or not np.isfinite(df.exposure_proxy).all()
        or (df.exposure_proxy < 0).any()
    ):
        raise ValueError("Invalid probabilities or exposures")
    if not df.flag_for_review.isin([0, 1]).all():
        raise ValueError("Review flags must be binary")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out / "portfolio.sqlite") as db:
        df.to_sql("predictions", db, if_exists="replace", index=False)
        sql = pd.read_sql_query(SQL, db).iloc[0].to_dict()
        if TARGET in df:
            sql["observed_defaults"] = db.execute(
                f'SELECT SUM("{TARGET}") FROM predictions'
            ).fetchone()[0]
    independent = {
        "accounts": len(df),
        "unique_accounts": df[ID].nunique(),
        "expected_defaults": float(df.probability_of_default.sum()),
        "total_exposure": float(df.exposure_proxy.sum()),
        "pd_weighted_exposure": float((df.probability_of_default * df.exposure_proxy).sum()),
        "review_flags": int(df.flag_for_review.sum()),
    }
    if TARGET in df:
        if not df[TARGET].isin([0, 1]).all():
            raise ValueError("Observed outcomes must be binary")
        independent["observed_defaults"] = int(df[TARGET].sum())
    checks = [
        {
            "metric": key,
            "sql": float(sql[key]),
            "pandas": float(value),
            "passed": bool(np.isclose(sql[key], value, rtol=1e-10, atol=1e-8)),
        }
        for key, value in independent.items()
    ]
    result = {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "scope": "Independent aggregation of the same scored file; not source-system verification",
    }
    (out / "reconciliation.sql").write_text(SQL + ";\n", encoding="utf-8")
    write_json(out / "reconciliation.json", result)
    return result
