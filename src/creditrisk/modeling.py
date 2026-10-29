"""Four-way separation of fitting, calibration, selection and final evaluation."""

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ID, TARGET, features, read_accounts, validate
from .io import new_directory, now, read_json, sha256, write_json
from .metrics import auc_interval, calibration_table, evaluate, segment_metrics

DEFAULT_CONFIG = {
    "seed": 42,
    "split": "stratified",
    "date_column": None,
    "label_horizon_days": 30,
    "fractions": {"train": 0.55, "calibration": 0.15, "validation": 0.15, "test": 0.15},
    "logistic_c": 0.1,
    "boosting_iterations": 150,
    "boosting_max_leaf_nodes": 15,
    "false_negative_cost": 5.0,
    "false_positive_cost": 1.0,
    "bootstrap_repetitions": 200,
    "psi_warning": 0.1,
    "psi_alert": 0.25,
}


def load_config(path=None):
    config = DEFAULT_CONFIG.copy()
    if path:
        custom = read_json(path)
        unknown = set(custom) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(custom)
    f = config["fractions"]
    if set(f) != {"train", "calibration", "validation", "test"} or any(v <= 0 for v in f.values()):
        raise ValueError(
            "fractions must contain positive train/calibration/validation/test entries"
        )
    if not np.isclose(sum(f.values()), 1):
        raise ValueError("Split fractions must sum to 1")
    if config["split"] not in {"stratified", "temporal"}:
        raise ValueError("split must be stratified or temporal")
    if config["split"] == "temporal" and not config["date_column"]:
        raise ValueError("Temporal splitting requires date_column")
    for name in ["logistic_c", "boosting_iterations", "false_negative_cost", "false_positive_cost"]:
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["bootstrap_repetitions"] < 20:
        raise ValueError("Use at least 20 bootstrap repetitions")
    if config["label_horizon_days"] < 0 or not 0 <= config["psi_warning"] < config["psi_alert"]:
        raise ValueError("Invalid label horizon or PSI thresholds")
    return config


def split_data(df, config):
    if len(df) < 200:
        raise ValueError("Training requires at least 200 rows")
    f, seed = config["fractions"], config["seed"]
    if config["split"] == "stratified":
        train, rest = train_test_split(
            df, train_size=f["train"], random_state=seed, stratify=df[TARGET]
        )
        cal, rest = train_test_split(
            rest,
            train_size=f["calibration"] / (1 - f["train"]),
            random_state=seed,
            stratify=rest[TARGET],
        )
        val, test = train_test_split(
            rest,
            train_size=f["validation"] / (f["validation"] + f["test"]),
            random_state=seed,
            stratify=rest[TARGET],
        )
        parts = {"train": train, "calibration": cal, "validation": val, "test": test}
    else:
        col = config["date_column"]
        if col not in df:
            raise ValueError(f"Missing date column: {col}")
        dates = pd.to_datetime(df[col], errors="raise", utc=True)
        if dates.isna().any():
            raise ValueError("Temporal dates cannot be missing")
        ordered = dates.sort_values()
        boundaries = [
            ordered.iloc[min(int(len(df) * q), len(df) - 1)]
            for q in np.cumsum([f["train"], f["calibration"], f["validation"]])
        ]
        if len(set(boundaries)) != 3:
            raise ValueError("Not enough distinct dates for temporal splitting")
        # Purge earlier snapshots whose future label window overlaps the next partition.
        gap = pd.Timedelta(days=config["label_horizon_days"])
        a, b, c = boundaries
        parts = {
            "train": df[(dates < a) & (dates + gap < a)],
            "calibration": df[(dates >= a) & (dates < b) & (dates + gap < b)],
            "validation": df[(dates >= b) & (dates < c) & (dates + gap < c)],
            "test": df[dates >= c],
        }
    for name, part in parts.items():
        if len(part) < 30 or part[TARGET].nunique() != 2:
            raise ValueError(
                f"{name} needs at least 30 rows and both classes; adjust data or splits"
            )
    return parts


def log_odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def predict(bundle, df):
    p = bundle["estimator"].predict_proba(features(df))[:, 1]
    if bundle["calibrator"] is not None:
        p = bundle["calibrator"].predict_proba(log_odds(p))[:, 1]
    return p


def load_model(path):
    # joblib is pickle-based. The CLI documents that only locally trusted artifacts are safe.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Unsupported model artifact")
    return bundle


def scored_frame(df, p, threshold):
    out = pd.DataFrame(
        {
            ID: df[ID].to_numpy(),
            "probability_of_default": p,
            "flag_for_review": (p >= threshold).astype(int),
            "exposure_proxy": df.bill_1.clip(lower=0).to_numpy(),
        }
    )
    if TARGET in df:
        out[TARGET] = df[TARGET].to_numpy()
    return out


def train(input_path, out, config_path=None):
    config = load_config(config_path)
    df = read_accounts(input_path)
    parts = split_data(df, config)
    out = new_directory(out)
    write_json(out / "config.json", config)
    write_json(out / "data_validation.json", validate(df))
    for name, part in parts.items():
        part.to_csv(out / f"{name}.csv", index=False)
    xtrain, ytrain = features(parts["train"]), parts["train"][TARGET]
    estimators = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=config["logistic_c"], max_iter=2000, random_state=config["seed"]),
        ),
        "boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=config["boosting_iterations"],
                max_leaf_nodes=config["boosting_max_leaf_nodes"],
                l2_regularization=2,
                early_stopping=False,
                random_state=config["seed"],
            ),
        ),
    }
    candidates, selection = {}, []
    for name, estimator in estimators.items():
        estimator.fit(xtrain, ytrain)
        pcal = estimator.predict_proba(features(parts["calibration"]))[:, 1]
        calibrator = LogisticRegression(C=1000, max_iter=1000).fit(
            log_odds(pcal), parts["calibration"][TARGET]
        )
        for variant, calibration in [(name, None), (name + "_calibrated", calibrator)]:
            bundle = {"format_version": 1, "estimator": estimator, "calibrator": calibration}
            candidates[variant] = bundle
            metrics = evaluate(parts["validation"][TARGET], predict(bundle, parts["validation"]))
            selection.append({"model": variant, **metrics})
    selection.sort(key=lambda row: (row["brier_score"], row["log_loss"]))
    champion_name = selection[0]["model"]
    champion = candidates[champion_name]
    pval = predict(champion, parts["validation"])
    # Select review threshold using validation only and illustrative error costs.
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for threshold in thresholds:
        m = evaluate(parts["validation"][TARGET], pval, float(threshold))
        cost = (
            m["false_negative"] * config["false_negative_cost"]
            + m["false_positive"] * config["false_positive_cost"]
        )
        costs.append({"threshold": float(threshold), "total_cost_units": float(cost)})
    threshold = min(costs, key=lambda row: row["total_cost_units"])["threshold"]
    used_ids = (
        pd.concat([parts[k][ID] for k in ["train", "calibration", "validation"]])
        .astype(str)
        .tolist()
    )
    champion.update(
        {
            "name": champion_name,
            "threshold": threshold,
            "development_ids": used_ids,
            "feature_names": list(xtrain.columns),
            "created_at": now(),
            "sklearn_version": sklearn.__version__,
            "config": config,
        }
    )
    joblib.dump(champion, out / "model.joblib")
    pd.DataFrame(selection).to_csv(out / "validation_comparison.csv", index=False)
    pd.DataFrame(costs).to_csv(out / "threshold_selection.csv", index=False)
    # Importance is measured on validation, not used to search against the final test set.
    importance = permutation_importance(
        champion["estimator"],
        features(parts["validation"]),
        parts["validation"][TARGET],
        scoring="roc_auc",
        n_repeats=3,
        random_state=config["seed"],
        n_jobs=1,
    )
    pd.DataFrame(
        {
            "feature": xtrain.columns,
            "auc_drop_mean": importance.importances_mean,
            "auc_drop_std": importance.importances_std,
        }
    ).sort_values("auc_drop_mean", ascending=False).to_csv(
        out / "feature_importance.csv", index=False
    )
    logistic = estimators["logistic"].named_steps["logisticregression"]
    pd.DataFrame({"feature": xtrain.columns, "standardized_coefficient": logistic.coef_[0]}).to_csv(
        out / "logistic_coefficients.csv", index=False
    )
    baseline = float(ytrain.mean())
    ptest = predict(champion, parts["test"])
    test_metrics = evaluate(parts["test"][TARGET], ptest, threshold)
    test_metrics["roc_auc_95pct_bootstrap"] = auc_interval(
        parts["test"][TARGET], ptest, config["seed"], config["bootstrap_repetitions"]
    )
    test_metrics["constant_baseline_brier"] = evaluate(
        parts["test"][TARGET], np.full(len(ptest), baseline)
    )["brier_score"]
    write_json(out / "test_metrics.json", test_metrics)
    scored_frame(parts["test"], ptest, threshold).to_csv(out / "test_predictions.csv", index=False)
    calibration_table(parts["test"][TARGET].to_numpy(), ptest).to_csv(
        out / "calibration.csv", index=False
    )
    segment_metrics(parts["test"], ptest).to_csv(out / "segments.csv", index=False)
    source_provenance = Path(input_path).parent / "provenance.json"
    manifest = {
        "created_at": now(),
        "input": str(Path(input_path).resolve()),
        "input_sha256": sha256(input_path),
        "model_sha256": sha256(out / "model.joblib"),
        "champion": champion_name,
        "selection_rule": "Lowest validation Brier score; log loss breaks ties",
        "split_method": config["split"],
        "input_rows": len(df),
        "purged_rows": len(df) - sum(len(p) for p in parts.values()),
        "partitions": {
            name: {"rows": len(part), "default_rate": float(part[TARGET].mean())}
            for name, part in parts.items()
        },
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_provenance": read_json(source_provenance)
        if source_provenance.exists()
        else {"source": "User-supplied; verify lineage and observation cutoff"},
    }
    write_json(out / "manifest.json", manifest)
    from .audit import reconcile

    reconcile(out / "test_predictions.csv", out / "audit")
    from .report import report

    report(out)
    return {
        "run": str(out.resolve()),
        "champion": champion_name,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_brier": test_metrics["brier_score"],
        "report": str((out / "report.html").resolve()),
    }


def evaluate_file(model_path, input_path, out):
    bundle = load_model(model_path)
    df = read_accounts(input_path)
    overlap = set(df[ID].astype(str)) & set(bundle["development_ids"])
    if overlap:
        raise ValueError(
            f"Evaluation contains {len(overlap)} training/calibration/selection accounts"
        )
    out = new_directory(out)
    p = predict(bundle, df)
    metrics = evaluate(df[TARGET], p, bundle["threshold"])
    write_json(out / "metrics.json", metrics)
    scored_frame(df, p, bundle["threshold"]).to_csv(out / "predictions.csv", index=False)
    segment_metrics(df, p).to_csv(out / "segments.csv", index=False)
    calibration_table(df[TARGET].to_numpy(), p).to_csv(out / "calibration.csv", index=False)
    write_json(
        out / "provenance.json",
        {
            "input_sha256": sha256(input_path),
            "model_sha256": sha256(model_path),
            "created_at": now(),
        },
    )
    return metrics
