"""Run the remaining CLI paths and record reproducible delivery evidence."""

import base64
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    commands = [
        ["validate-data", "--input", "data/uci/accounts.csv", "--out", "data/uci/validation.json"],
        [
            "evaluate",
            "--model",
            "runs/uci/model.joblib",
            "--input",
            "runs/uci/test.csv",
            "--out",
            "runs/uci-recheck",
        ],
        [
            "score",
            "--model",
            "runs/uci/model.joblib",
            "--input",
            "examples/accounts_unlabeled.csv",
            "--out",
            "runs/example-scored",
        ],
        [
            "audit",
            "--input",
            "runs/example-scored/predictions.csv",
            "--out",
            "runs/example-scored/audit",
        ],
        [
            "scenario",
            "--model",
            "runs/uci/model.joblib",
            "--input",
            "runs/uci/test.csv",
            "--out",
            "runs/uci-scenario",
        ],
        ["report", "--run", "runs/uci"],
    ]
    for command in commands:
        print("Verifying " + command[0], flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "creditrisk", *command],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        json.loads(result.stdout)
    metrics = json.loads((root / "runs/uci/test_metrics.json").read_text())
    repeated = json.loads((root / "runs/uci-recheck/metrics.json").read_text())
    assert metrics["roc_auc"] == repeated["roc_auc"]
    assert metrics["brier_score"] == repeated["brier_score"]
    for run in ["runs/uci", "runs/demo/model", "runs/demo/monitoring"]:
        path = root / run
        report = (path / "report.html").read_text(encoding="utf-8")
        assert "<html lang=" in report and "</html>" in report
        images = re.findall(r"base64,([^\"]+)", report)
        assert images
        for index, encoded in enumerate(images):
            content = base64.b64decode(encoded, validate=True)
            assert content.startswith(b"\x89PNG")
            (path / f"figure-{index}.png").write_bytes(content)
    print("All CLI paths passed; reloaded model metrics match; embedded report images verified.")


if __name__ == "__main__":
    main()
