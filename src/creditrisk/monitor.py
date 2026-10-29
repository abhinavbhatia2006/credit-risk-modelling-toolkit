"""Drift monitoring supports labeled and unlabeled portfolios."""

from pathlib import Path

import pandas as pd

from .data import ID, TARGET, features, read_accounts
from .io import new_directory, now, sha256, write_json
from .metrics import evaluate, psi, segment_metrics
from .modeling import load_model, predict, scored_frame


def monitor(model_path, reference_path, current_path, out):
    model = load_model(model_path)
    reference = read_accounts(reference_path, labeled=False)
    current = read_accounts(current_path, labeled=False)
    if set(reference[ID].astype(str)) & set(current[ID].astype(str)):
        raise ValueError("Monitoring demo requires disjoint reference/current customer cohorts")
    out = new_directory(out)
    xp, xc = features(reference), features(current)
    rp, cp = predict(model, reference), predict(model, current)
    xp["model_probability"], xc["model_probability"] = rp, cp
    warning, alert = model["config"]["psi_warning"], model["config"]["psi_alert"]
    rows = []
    for feature in xp.columns:
        value = psi(xp[feature], xc[feature])
        rows.append(
            {
                "feature": feature,
                "psi": value,
                "status": "alert" if value >= alert else "watch" if value >= warning else "stable",
                "reference_missing_rate": float(xp[feature].isna().mean()),
                "current_missing_rate": float(xc[feature].isna().mean()),
                "reference_mean": float(xp[feature].mean()),
                "current_mean": float(xc[feature].mean()),
            }
        )
    drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
    drift.to_csv(out / "drift.csv", index=False)
    result = {
        "created_at": now(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_mean_pd": float(rp.mean()),
        "current_mean_pd": float(cp.mean()),
        "alerts": drift.loc[drift.status == "alert", "feature"].tolist(),
        "watch": drift.loc[drift.status == "watch", "feature"].tolist(),
        "thresholds": {"watch": warning, "alert": alert},
        "interpretation": "PSI is a heuristic drift indicator, not a significance test or proof of causality",
        "performance": None,
        "input_hashes": {
            "reference": sha256(reference_path),
            "current": sha256(current_path),
            "model": sha256(model_path),
        },
    }
    if TARGET in current:
        result["performance"] = evaluate(current[TARGET], cp, model["threshold"])
        result["performance_note"] = (
            "Descriptive cohort performance; check label maturity and independence"
        )
        if set(current[ID].astype(str)) & set(model["development_ids"]):
            result["performance_note"] += (
                "; includes development accounts, not independent validation"
            )
        segment_metrics(current, cp).to_csv(out / "segments.csv", index=False)
    else:
        result["performance_note"] = (
            "No observed outcomes: discrimination and calibration cannot be measured"
        )
    scored_frame(current, cp, model["threshold"]).to_csv(out / "predictions.csv", index=False)
    write_json(out / "monitoring.json", result)
    from .report import monitoring_report

    monitoring_report(out)
    return result


def scenario(model_path, input_path, out, lgd=0.6, pd_multiplier=1.5):
    if not 0 <= lgd <= 1 or pd_multiplier < 1:
        raise ValueError("LGD must be in [0,1]; adverse PD multiplier must be >= 1")
    df = read_accounts(input_path, labeled=False)
    model = load_model(model_path)
    p = predict(model, df)
    exposure = df.bill_1.clip(lower=0).to_numpy()
    stressed = (p * pd_multiplier).clip(0, 1)
    base = float((p * exposure * lgd).sum())
    adverse = float((stressed * exposure * lgd).sum())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "accounts": len(df),
        "lgd_assumption": lgd,
        "adverse_pd_multiplier": pd_multiplier,
        "base_loss_proxy": base,
        "adverse_loss_proxy": adverse,
        "incremental_loss_proxy": adverse - base,
        "exposure_proxy_total": float(exposure.sum()),
        "method": "Sum(PD * nonnegative latest bill balance * assumed LGD)",
        "limitations": "One-month sensitivity illustration; balances are exposure proxies; LGD is assumed. "
        "Not a lifetime CECL reserve, regulatory stress test, or empirically estimated macroeconomic scenario.",
        "input_sha256": sha256(input_path),
        "model_sha256": sha256(model_path),
    }
    write_json(out / "scenario.json", result)
    return result
