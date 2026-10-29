"""Canonical account snapshots, synthetic demo and attributed UCI import."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io import new_directory, now, sha256, write_json

ID = "account_id"
TARGET = "default_next_month"
STATUS = [f"status_{i}" for i in range(1, 7)]
BILLS = [f"bill_{i}" for i in range(1, 7)]
PAYMENTS = [f"payment_{i}" for i in range(1, 7)]
RAW_FEATURES = ["credit_limit", *STATUS, *BILLS, *PAYMENTS]
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_CITATION = (
    "Yeh, I. (2009). Default of Credit Card Clients. UCI. doi:10.24432/C55S3H. CC BY 4.0."
)


def read_accounts(path, labeled=True):
    df = pd.read_csv(path, dtype={ID: "string"})
    result = validate(df, labeled=labeled)
    if result["errors"]:
        raise ValueError("Data validation failed: " + "; ".join(result["errors"]))
    return df


def validate(df, labeled=True):
    errors, warnings = [], []
    required = [ID, *RAW_FEATURES] + ([TARGET] if labeled else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        errors.append("Missing columns: " + ", ".join(missing))
    if df.empty:
        errors.append("Dataset has no rows")
    if ID in df:
        if df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any():
            errors.append("account_id must not be empty")
        if df[ID].duplicated().any():
            errors.append("account_id must be unique (one snapshot per customer)")
    for col in RAW_FEATURES:
        if col not in df:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & numeric.isna()).any() or np.isinf(numeric).any():
            errors.append(f"{col} contains nonnumeric or infinite values")
        if numeric.isna().all():
            errors.append(f"{col} is entirely missing")
        elif numeric.isna().any():
            warnings.append(
                f"{col}: {int(numeric.isna().sum())} missing; training-only median imputation"
            )
        if col == "credit_limit" and (numeric.isna().any() or (numeric <= 0).any()):
            errors.append("credit_limit must be present and positive")
        if col in PAYMENTS and (numeric < 0).any():
            errors.append(f"{col} cannot be negative")
        if col in STATUS and (
            (numeric.dropna() < -2).any()
            or (numeric.dropna() > 9).any()
            or (numeric.dropna() % 1 != 0).any()
        ):
            errors.append(f"{col} must be an integer from -2 to 9")
    if TARGET in df:
        if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
            errors.append(f"{TARGET} must contain only 0 and 1 with no missing values")
        elif df[TARGET].nunique() < 2:
            warnings.append("Only one outcome class; AUC and KS cannot be estimated")
    extras = sorted(set(df.columns) - {ID, *RAW_FEATURES, TARGET})
    if extras:
        warnings.append("Excluded from modeling by allowlist: " + ", ".join(extras))
    return {
        "rows": len(df),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
        "missing_by_column": df.isna().sum().to_dict(),
    }


def features(df):
    """Use only observed financial history; never infer predictors from extra columns."""
    x = df[RAW_FEATURES].astype(float).copy()
    limit = x["credit_limit"]
    x["utilization"] = x["bill_1"].clip(lower=0) / limit
    x["mean_utilization"] = x[BILLS].clip(lower=0).mean(axis=1) / limit
    x["payment_to_bill"] = x["payment_1"] / x["bill_1"].abs().clip(lower=1)
    x["mean_payment_to_limit"] = x[PAYMENTS].mean(axis=1) / limit
    x["delinquent_months"] = x[STATUS].gt(0).sum(axis=1)
    x["max_delinquency"] = x[STATUS].max(axis=1)
    x["balance_change_to_limit"] = (x["bill_1"] - x["bill_6"]) / limit
    return x.replace([np.inf, -np.inf], np.nan)


def synthetic(n=6000, seed=42, shift=False, prefix="demo"):
    if n < 100:
        raise ValueError("Generate at least 100 rows")
    rng = np.random.default_rng(seed)
    limit = rng.choice([20000, 50000, 100000, 200000, 400000], n)
    latent = rng.normal(0.45 if shift else 0, 1, n)
    utilization = np.clip(rng.beta(2, 3, n) + 0.12 * latent, 0, 1.6)
    df = pd.DataFrame({ID: [f"{prefix}-{i:07d}" for i in range(n)], "credit_limit": limit})
    for i in range(1, 7):
        status = np.clip(np.floor(latent + rng.normal(0, 0.8, n)), -2, 5).astype(int)
        bill = np.maximum(0, limit * (utilization + rng.normal(0, 0.08, n)))
        df[f"status_{i}"] = status
        df[f"bill_{i}"] = bill.round(2)
        df[f"payment_{i}"] = (bill * np.clip(rng.beta(1, 5, n) - latent * 0.02, 0, 1)).round(2)
    logit = -2.4 + 1.1 * np.maximum(df.status_1, 0) + 1.6 * utilization + 0.25 * latent
    df[TARGET] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def generate_demo(out, rows=6000, seed=42):
    out = new_directory(out)
    synthetic(rows, seed).to_csv(out / "development.csv", index=False)
    synthetic(max(rows // 3, 100), seed + 1, True, "shifted").to_csv(
        out / "current.csv", index=False
    )
    write_json(
        out / "provenance.json",
        {
            "source": "Synthetic educational simulation; not bank/customer data",
            "seed": seed,
            "rows": rows,
            "created_at": now(),
            "current": "Independent simulated cohort with deliberately shifted risk distribution",
            "currency": "Illustrative units",
            "horizon": "simulated next-month default",
        },
    )
    return out


def import_uci(out, archive=None):
    out = new_directory(out)
    raw = out / "source.zip"
    if archive:
        raw.write_bytes(Path(archive).read_bytes())
    else:
        request = urllib.request.Request(
            UCI_URL, headers={"User-Agent": "CreditRisk-Educational/1.0"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            raw.write_bytes(response.read())
    with zipfile.ZipFile(raw) as z:
        names = [name for name in z.namelist() if name.lower().endswith(".xls")]
        if len(names) != 1:
            raise ValueError("Expected exactly one XLS file in the UCI archive")
        source = pd.read_excel(io.BytesIO(z.read(names[0])), header=1, engine="xlrd")
    source.columns = source.columns.astype(str).str.strip()
    rename = {
        "ID": ID,
        "LIMIT_BAL": "credit_limit",
        "default payment next month": TARGET,
        "PAY_0": "status_1",
    }
    rename.update({f"PAY_{i}": f"status_{i}" for i in range(2, 7)})
    rename.update({f"BILL_AMT{i}": f"bill_{i}" for i in range(1, 7)})
    rename.update({f"PAY_AMT{i}": f"payment_{i}" for i in range(1, 7)})
    canonical = source.rename(columns=rename)[[ID, *RAW_FEATURES, TARGET]]
    result = validate(canonical)
    if result["errors"]:
        raise ValueError(str(result["errors"]))
    canonical.to_csv(out / "accounts.csv", index=False)
    write_json(
        out / "provenance.json",
        {
            "source": UCI_URL,
            "citation": UCI_CITATION,
            "downloaded_at": now(),
            "archive_sha256": sha256(raw),
            "rows": len(canonical),
            "population": "Taiwan credit-card customers, 2005; not US or Citi data",
            "currency": "NT dollars",
            "horizon": "next-month payment default",
            "adaptations": "Renamed columns; excluded sex, education, marital status and age",
            "split_limitation": "Single cohort: stratified random split, not temporal validation",
        },
    )
    return out
