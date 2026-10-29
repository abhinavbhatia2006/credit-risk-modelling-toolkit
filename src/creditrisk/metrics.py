"""Discrimination, calibration and monitoring statistics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def evaluate(y, probability, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Outcomes and predictions must have equal, nonzero length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes must be binary")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    fpr, tpr, _ = roc_curve(y, p) if both else (None, None, None)
    return {
        "rows": len(y),
        "defaults": int(y.sum()),
        "default_rate": float(y.mean()),
        "mean_pd": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "ks": float(np.max(tpr - fpr)) if both else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": threshold,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def auc_interval(y, p, seed=42, repetitions=200):
    """Percentile bootstrap interval; descriptive, not model-selection evidence."""
    y, p = np.asarray(y), np.asarray(p)
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None


def calibration_table(y, p):
    frame = pd.DataFrame({"observed": y, "predicted": p})
    frame["bin"] = pd.cut(frame.predicted, np.linspace(0, 1, 11), include_lowest=True)
    return (
        frame.groupby("bin", observed=True)
        .agg(
            accounts=("observed", "size"),
            predicted_pd=("predicted", "mean"),
            observed_rate=("observed", "mean"),
        )
        .reset_index()
        .assign(bin=lambda x: x["bin"].astype(str))
    )


def psi(reference, current, bins=10):
    """Reference-fitted quantile bins, explicit missing bin, epsilon smoothing."""
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if not len(a) or not len(b):
        raise ValueError("PSI requires nonempty populations")
    clean = a[np.isfinite(a)]
    if not len(clean):
        edges = np.array([-np.inf, np.inf])
    elif np.unique(clean).size <= bins:
        values = np.unique(clean)
        # A constant baseline still detects departures in either direction.
        edges = np.r_[
            -np.inf, values[0] - 1e-9, (values[:-1] + values[1:]) / 2, values[-1] + 1e-9, np.inf
        ]
    else:
        edges = np.r_[
            -np.inf, np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)[1:-1])), np.inf
        ]
    ac = np.r_[np.histogram(a[np.isfinite(a)], edges)[0], (~np.isfinite(a)).sum()]
    bc = np.r_[np.histogram(b[np.isfinite(b)], edges)[0], (~np.isfinite(b)).sum()]
    ap, bp = ac / len(a), bc / len(b)
    ap, bp = np.maximum(ap, 1e-6), np.maximum(bp, 1e-6)
    ap, bp = ap / ap.sum(), bp / bp.sum()
    return float(np.sum((bp - ap) * np.log(bp / ap)))


def segment_metrics(df, p):
    from .data import TARGET

    frame = df.copy()
    frame["pd"] = p
    util = frame.bill_1.clip(lower=0) / frame.credit_limit
    frame["utilization_band"] = pd.cut(
        util, [-np.inf, 0.3, 0.7, 1, np.inf], labels=["<=30%", "30-70%", "70-100%", ">100%"]
    )
    frame["delinquency_band"] = np.where(frame.status_1 > 0, "delinquent", "not delinquent")
    results = []
    for column in ["utilization_band", "delinquency_band"]:
        for name, group in frame.groupby(column, observed=True):
            result = evaluate(group[TARGET], group.pd)
            results.append(
                {
                    "dimension": column,
                    "segment": str(name),
                    **result,
                    "small_sample": len(group) < 100,
                }
            )
    return pd.DataFrame(results)
