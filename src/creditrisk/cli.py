"""Command line interface. Machine-readable JSON on stdout, errors on stderr."""

import argparse
import json
import sys
from urllib.error import URLError

import pandas as pd
from threadpoolctl import threadpool_limits

from . import __version__
from .data import ID, generate_demo, import_uci, read_accounts, validate
from .io import json_default, new_directory, write_json


def parser():
    root = argparse.ArgumentParser(
        prog="creditrisk",
        description="Credit-card risk modeling, validation and monitoring.",
        epilog="Only load trusted model.joblib files. Run 'creditrisk demo' for an offline end-to-end example.",
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--debug", action="store_true", help="Show traceback on failure")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the complete offline synthetic workflow")
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--rows", type=int, default=6000)
    demo.add_argument("--seed", type=int, default=42)
    generate = commands.add_parser("generate-data", help="Generate reproducible synthetic datasets")
    generate.add_argument("--out", default="data/synthetic")
    generate.add_argument("--rows", type=int, default=6000)
    generate.add_argument("--seed", type=int, default=42)
    fetch = commands.add_parser("fetch-data", help="Download and canonicalize attributed UCI data")
    fetch.add_argument("--out", default="data/uci")
    fetch.add_argument("--archive", help="Use a local UCI ZIP instead of downloading")
    check = commands.add_parser("validate-data", help="Check the canonical account schema")
    check.add_argument("--input", required=True)
    check.add_argument("--out", help="Optional JSON result path")
    check.add_argument("--unlabeled", action="store_true")
    train = commands.add_parser("train", help="Train, select, evaluate and report")
    train.add_argument("--input", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--config", help="JSON settings; see configs/default.json")
    for name in ["evaluate", "score"]:
        cmd = commands.add_parser(
            name,
            help="Independent labeled evaluation"
            if name == "evaluate"
            else "Score labeled or unlabeled accounts",
        )
        cmd.add_argument("--model", required=True, help="Trusted local model.joblib")
        cmd.add_argument("--input", required=True)
        cmd.add_argument("--out", required=True)
    monitoring = commands.add_parser("monitor", help="Compare reference and current cohorts")
    monitoring.add_argument("--model", required=True)
    monitoring.add_argument("--reference", required=True)
    monitoring.add_argument("--current", required=True)
    monitoring.add_argument("--out", required=True)
    audit = commands.add_parser("audit", help="Reconcile predictions using SQLite and pandas")
    audit.add_argument("--input", required=True, help="Scored predictions CSV")
    audit.add_argument("--out", required=True)
    scenario = commands.add_parser("scenario", help="Illustrative PD/LGD/exposure sensitivity")
    scenario.add_argument("--model", required=True)
    scenario.add_argument("--input", required=True)
    scenario.add_argument("--out", required=True)
    scenario.add_argument("--lgd", type=float, default=0.6)
    scenario.add_argument("--pd-multiplier", type=float, default=1.5)
    report = commands.add_parser("report", help="Regenerate a run's portable HTML report")
    report.add_argument("--run", required=True)
    return root


def dispatch(args):
    # Lazy imports keep help and validation lightweight.
    if args.command == "generate-data":
        return {"data": str(generate_demo(args.out, args.rows, args.seed).resolve())}
    if args.command == "fetch-data":
        return {"data": str(import_uci(args.out, args.archive).resolve())}
    if args.command == "validate-data":
        result = validate(pd.read_csv(args.input, dtype={ID: "string"}), labeled=not args.unlabeled)
        if args.out:
            write_json(args.out, result)
        return result
    if args.command == "train":
        from .modeling import train

        return train(args.input, args.out, args.config)
    if args.command == "evaluate":
        from .modeling import evaluate_file

        return evaluate_file(args.model, args.input, args.out)
    if args.command == "score":
        from .modeling import load_model, predict, scored_frame

        df = read_accounts(args.input, labeled=False)
        model = load_model(args.model)
        out = new_directory(args.out)
        scored_frame(df, predict(model, df), model["threshold"]).to_csv(
            out / "predictions.csv", index=False
        )
        from .io import now, sha256

        write_json(
            out / "provenance.json",
            {
                "input_sha256": sha256(args.input),
                "model_sha256": sha256(args.model),
                "created_at": now(),
            },
        )
        return {"rows": len(df), "predictions": str((out / "predictions.csv").resolve())}
    if args.command == "monitor":
        from .monitor import monitor

        return monitor(args.model, args.reference, args.current, args.out)
    if args.command == "audit":
        from .audit import reconcile

        return reconcile(args.input, args.out)
    if args.command == "scenario":
        from .monitor import scenario

        return scenario(args.model, args.input, args.out, args.lgd, args.pd_multiplier)
    if args.command == "report":
        from .report import report

        return {"report": report(args.run)}
    if args.command == "demo":
        from .modeling import train
        from .monitor import monitor, scenario

        out = new_directory(args.out)
        print("[1/4] Generating synthetic development and shifted current cohorts", file=sys.stderr)
        generate_demo(out / "data", args.rows, args.seed)
        print("[2/4] Fitting, calibrating, selecting and validating models", file=sys.stderr)
        config_path = out / "demo_config.json"
        write_json(config_path, {"seed": args.seed})
        result = train(out / "data/development.csv", out / "model", config_path)
        print("[3/4] Monitoring shifted portfolio", file=sys.stderr)
        monitor(
            out / "model/model.joblib",
            out / "model/train.csv",
            out / "data/current.csv",
            out / "monitoring",
        )
        print("[4/4] Computing illustrative loss sensitivity", file=sys.stderr)
        scenario(out / "model/model.joblib", out / "data/current.csv", out / "scenario")
        return {
            **result,
            "monitoring_report": str((out / "monitoring/report.html").resolve()),
            "note": "Synthetic demo results; not empirical bank performance",
        }
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        # Bound CPU use and avoid oversubscription on high-core Windows machines.
        with threadpool_limits(limits=2):
            result = dispatch(args)
        print(json.dumps(result, indent=2, default=json_default, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("passed") is False else 0
    except (ValueError, OSError, KeyError, TypeError, URLError) as exc:
        if args.debug:
            raise
        print(f"creditrisk: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
