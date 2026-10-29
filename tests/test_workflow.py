import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from creditrisk.audit import reconcile
from creditrisk.data import TARGET, features, generate_demo
from creditrisk.io import read_json, write_json
from creditrisk.modeling import evaluate_file, load_model, predict, train
from creditrisk.monitor import monitor, scenario


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    base = tmp_path_factory.mktemp("workflow")
    generate_demo(base / "data", rows=1000)
    write_json(base / "config.json", {"boosting_iterations": 15, "bootstrap_repetitions": 20})
    with threadpool_limits(limits=2):
        train(base / "data/development.csv", base / "run", base / "config.json")
    return base


def test_end_to_end_artifacts_and_independent_audit(workflow):
    run = workflow / "run"
    metrics = read_json(run / "test_metrics.json")
    assert 0 <= metrics["roc_auc"] <= 1
    assert read_json(run / "audit/reconciliation.json")["passed"]
    assert "data:image/png;base64," in (run / "report.html").read_text(encoding="utf-8")
    assert (run / "MODEL_CARD.md").exists()
    assert read_json(run / "manifest.json")["input_rows"] == 1000


def test_serialized_scoring_matches_report(workflow):
    run = workflow / "run"
    model = load_model(run / "model.joblib")
    test = pd.read_csv(run / "test.csv")
    with threadpool_limits(limits=2):
        p = predict(model, test)
    saved = pd.read_csv(run / "test_predictions.csv")
    np.testing.assert_allclose(p, saved.probability_of_default)


def test_preprocessing_fit_only_on_train(workflow):
    run = workflow / "run"
    model = load_model(run / "model.joblib")
    train_df = pd.read_csv(run / "train.csv")
    imputer = model["estimator"].named_steps["simpleimputer"]
    np.testing.assert_allclose(imputer.statistics_, features(train_df).median())


def test_evaluate_rejects_development_overlap(workflow):
    with pytest.raises(ValueError, match="training/calibration/selection"):
        evaluate_file(
            workflow / "run/model.joblib", workflow / "run/train.csv", workflow / "bad-eval"
        )


def test_unlabeled_monitoring_omits_performance(workflow):
    df = pd.read_csv(workflow / "data/current.csv").drop(columns=TARGET)
    df.to_csv(workflow / "unlabeled.csv", index=False)
    with threadpool_limits(limits=2):
        result = monitor(
            workflow / "run/model.joblib",
            workflow / "run/train.csv",
            workflow / "unlabeled.csv",
            workflow / "monitor",
        )
    assert result["performance"] is None
    assert "No observed outcomes" in result["performance_note"]
    assert (workflow / "monitor/report.html").exists()


def test_scenario_monotone_and_explicit(workflow):
    with threadpool_limits(limits=2):
        result = scenario(
            workflow / "run/model.joblib", workflow / "run/test.csv", workflow / "scenario"
        )
    assert result["adverse_loss_proxy"] >= result["base_loss_proxy"]
    assert "Not a lifetime CECL" in result["limitations"]


def test_audit_known_totals(tmp_path):
    frame = pd.DataFrame(
        {
            "account_id": ["a", "b"],
            "probability_of_default": [0.1, 0.2],
            "exposure_proxy": [100, 200],
            "flag_for_review": [0, 1],
        }
    )
    file = tmp_path / "predictions.csv"
    frame.to_csv(file, index=False)
    result = reconcile(file, tmp_path / "audit")
    values = {item["metric"]: item["sql"] for item in result["checks"]}
    assert result["passed"]
    assert values["expected_defaults"] == pytest.approx(0.3)
    assert values["pd_weighted_exposure"] == pytest.approx(50)
    assert values["total_exposure"] == 300


def test_cli_help_and_bad_data(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-m", "creditrisk", "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0 and "monitor" in help_result.stdout
    file = tmp_path / "bad.csv"
    file.write_text("account_id\na\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "creditrisk", "validate-data", "--input", str(file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["passed"] is False


def test_no_silent_run_overwrite(workflow):
    with pytest.raises(ValueError, match="not empty"):
        train(workflow / "data/development.csv", workflow / "run")
