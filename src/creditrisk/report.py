"""Portable HTML reports with embedded figures; no web server required."""

import base64
import html
import io
import os
import tempfile
from pathlib import Path

# Keep the plotting cache writable without touching the user's profile settings.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "creditrisk-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .data import TARGET
from .io import read_json

CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#17283c;
font:15px/1.65 'Segoe UI',Arial,sans-serif}main{max-width:1160px;margin:auto;padding:48px 28px}
header{border-top:5px solid #0b8b83;padding:25px 0}h1{font-size:38px;line-height:1.15;letter-spacing:-1px;margin:12px 0}
h2{font-size:23px;margin:0 0 15px}h3{font-size:17px}.eyebrow{color:#087a73;font-weight:700;
letter-spacing:2px;font-size:12px;text-transform:uppercase}.subtle{color:#586b80}.cards{display:grid;
grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}.card,section{background:white;
border:1px solid #dce3eb;border-radius:10px;padding:24px}.card strong{display:block;font-size:29px;
color:#076d69}.card span{font-size:13px;color:#526479}section{margin:20px 0;overflow:auto}
.notice{padding:15px 18px;border-left:4px solid #d49626;background:#fff5df;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:#476076;
background:#edf4f7}td,th{padding:10px;border-bottom:1px solid #e0e8ee;white-space:nowrap}
img{max-width:100%;height:auto}code{background:#eaf0f5;padding:2px 5px}li{margin-bottom:7px}
footer{font-size:12px;color:#617286;padding:20px 0}a{color:#087a73}
@media(max-width:720px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:20px 14px}h1{font-size:29px}}
@media print{body{background:white}main{padding:0}section,.card{break-inside:avoid}header{padding:10px 0}}
"""


def esc(value):
    return html.escape(str(value))


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:,.4f}", escape=True)


def figure_image(fig, description):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img alt="{esc(description)}" src="data:image/png;base64,{base64.b64encode(stream.getvalue()).decode()}">'


def shell(title, subtitle, content):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>
<style>{CSS}</style></head><body><main><header><div class="eyebrow">CreditRisk / Research Workbench</div>
<h1>{esc(title)}</h1><p class="subtle">{esc(subtitle)}</p></header>{content}
<footer>CreditRisk CLI · Educational model development and validation · Generated locally</footer>
</main></body></html>"""


def report(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    metrics = read_json(run / "test_metrics.json")
    config = read_json(run / "config.json")
    audit = read_json(run / "audit/reconciliation.json")
    comparison = pd.read_csv(run / "validation_comparison.csv")
    predictions = pd.read_csv(run / "test_predictions.csv")
    calibration = pd.read_csv(run / "calibration.csv")
    segments = pd.read_csv(run / "segments.csv")
    importance = pd.read_csv(run / "feature_importance.csv").head(10)
    y, p = predictions[TARGET].to_numpy(), predictions.probability_of_default.to_numpy()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#33485e",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color="#087f7a", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#9aa9b8")
    axes[0].set(
        xlabel="False positive rate", ylabel="True positive rate", title="ROC discrimination"
    )
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, color="#3265a8", lw=2)
    axes[1].axhline(y.mean(), color="#9aa9b8", ls="--")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    axes[2].plot(calibration.predicted_pd, calibration.observed_rate, "o-", color="#087f7a")
    axes[2].plot([0, 1], [0, 1], "--", color="#9aa9b8")
    axes[2].set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title="Calibration · fixed-width bins",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    fig.tight_layout()
    plots = figure_image(fig, "Held-out ROC, precision-recall and calibration curves")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ordered = importance.iloc[::-1]
    ax.barh(
        ordered.feature,
        ordered.auc_drop_mean,
        xerr=ordered.auc_drop_std,
        color="#168e87",
        alpha=0.9,
    )
    ax.set(
        xlabel="Validation ROC-AUC decrease after shuffling (3 repeats)", title="Predictive drivers"
    )
    fig.tight_layout()
    drivers = figure_image(fig, "Permutation importance on validation accounts")
    interval = metrics["roc_auc_95pct_bootstrap"]
    limitations = [
        "Educational analysis; not approved for underwriting or production use.",
        "One snapshot per account. Feature allowlisting prevents accidental inclusion of outcomes, but source-time leakage still requires a data-owner review.",
        "Protected demographic fields are excluded. This does not establish fairness or remove proxy discrimination; no fairness certification is claimed.",
        "Default labels must be fully matured before evaluation. Repayment-status codes <= 0 are retained as recorded; interpret them with source documentation.",
        "Review thresholds use illustrative error costs, not validated lending economics.",
        "Permutation importance describes predictive associations, not causal effects; correlated inputs can share importance.",
    ]
    if manifest["split_method"] == "stratified":
        limitations.append(
            "Stratified random holdout from one population; no evidence of future-period or US-portfolio generalization."
        )
    else:
        limitations.append(
            "Temporal splits purge overlapping label windows; horizon and dates must match the source label definition."
        )
    source = manifest["source_provenance"]
    provenance = esc(source.get("population", source.get("source", "User supplied")))
    baseline = metrics["constant_baseline_brier"]
    verdict = "outperforms" if metrics["brier_score"] < baseline else "does not outperform"
    cards = "".join(
        f'<div class="card"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in [
            (f"{metrics['roc_auc']:.3f}", "Test ROC-AUC"),
            (f"{metrics['brier_score']:.4f}", "Test Brier · lower is better"),
            (f"{metrics['ks']:.3f}", "Test KS"),
            (f"{metrics['rows']:,}", "Untouched test accounts"),
        ]
    )
    partitions = pd.DataFrame([{"partition": k, **v} for k, v in manifest["partitions"].items()])
    content = f"""<div class="notice"><strong>Research scope:</strong> {provenance}. Results apply only to this dataset and split.</div>
<div class="cards">{cards}</div><section><h2>Validation conclusion</h2>
<p><strong>{esc(manifest["champion"])}</strong> was selected by lowest validation Brier score before opening the test set.
On the test set it {verdict} the constant training-default-rate baseline (Brier {baseline:.4f}).</p>
<p>Mean predicted default probability: <strong>{metrics["mean_pd"]:.2%}</strong>; observed rate:
<strong>{metrics["default_rate"]:.2%}</strong>. ROC-AUC 95% percentile-bootstrap interval:
<strong>{interval[0]:.3f}–{interval[1]:.3f}</strong> ({config["bootstrap_repetitions"]} resamples).</p>
<p>This is a research candidate. Independent future-cohort validation and business review remain necessary.</p></section>
<section><h2>Data separation & reproducibility</h2><p>Split: {esc(manifest["split_method"])}.
Purged rows: {manifest["purged_rows"]:,}. Preprocessing fits on training rows only; sigmoid calibration uses calibration rows;
model and threshold selection use validation rows; reported final performance uses test rows.</p>{table(partitions)}
<p class="subtle">Input SHA-256: <code>{esc(manifest["input_sha256"])}</code></p></section>
<section><h2>Model comparison · validation only</h2>{table(comparison[["model", "roc_auc", "average_precision", "brier_score", "log_loss"]])}</section>
<section><h2>Held-out performance</h2>{plots}<h3>Calibration detail</h3>{table(calibration)}</section>
<section><h2>Review threshold</h2><p>Selected threshold: <strong>{metrics["threshold"]:.2f}</strong>.
Assumed false-negative cost: {config["false_negative_cost"]}; false-positive cost: {config["false_positive_cost"]}.
These are illustrative cost units. A flag means review, not automatic credit rejection.</p>
{table(pd.DataFrame([{"true_negative": metrics["true_negative"], "false_positive": metrics["false_positive"], "false_negative": metrics["false_negative"], "true_positive": metrics["true_positive"], "precision": metrics["precision"], "recall": metrics["recall"]}]))}</section>
<section><h2>Segment diagnostics</h2><p>Positive calibration gap means overprediction; negative means underprediction.
Small segments are flagged and should not support strong conclusions.</p>
{table(segments[["dimension", "segment", "rows", "default_rate", "mean_pd", "calibration_gap", "roc_auc", "small_sample"]])}</section>
<section><h2>Model drivers</h2>{drivers}<p>Importance is measured on the underlying selected estimator; the calibration layer is not shuffled.</p></section>
<section><h2>Independent report reconciliation</h2><p>Status: <strong>{"PASS" if audit["passed"] else "FAIL"}</strong>.
SQLite aggregates are compared with separately calculated pandas totals. This checks reporting arithmetic, not upstream source correctness.</p>
{table(pd.DataFrame(audit["checks"]))}</section><section><h2>Limitations & next validation steps</h2><ul>
{"".join(f"<li>{esc(item)}</li>" for item in limitations)}</ul>
<p>Source attribution: {esc(source.get("citation", source.get("source", "User-supplied data")))}</p></section>"""
    (run / "report.html").write_text(
        shell(
            "Credit risk · Model validation",
            "One-month default probability / baseline, challenger and independent test",
            content,
        ),
        encoding="utf-8",
    )
    (run / "MODEL_CARD.md").write_text(
        f"# Model card\n\nSelected model: {manifest['champion']}\n\nSource: {source}\n\n"
        f"Selection: {manifest['selection_rule']}\n\nTest ROC-AUC: {metrics['roc_auc']:.4f}; "
        f"Brier: {metrics['brier_score']:.4f}; KS: {metrics['ks']:.4f}.\n\n"
        + "\n".join(f"- {item}" for item in limitations),
        encoding="utf-8",
    )
    return str((run / "report.html").resolve())


def monitoring_report(out):
    out = Path(out)
    result = read_json(out / "monitoring.json")
    drift = pd.read_csv(out / "drift.csv")
    top = drift.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(
        top.feature, top.psi, color=["#c77b2c" if s == "alert" else "#168e87" for s in top.status]
    )
    ax.axvline(result["thresholds"]["alert"], color="#b46b24", ls="--")
    ax.set(xlabel="Population Stability Index", title="Largest distribution changes")
    fig.tight_layout()
    content = f"""<div class="notice">Drift indicates a distribution change. It does not establish the cause or prove deterioration.</div>
<section><h2>Portfolio movement</h2><p>Reference: {result["reference_rows"]:,} accounts; current: {result["current_rows"]:,}.
Mean PD moved from {result["reference_mean_pd"]:.2%} to {result["current_mean_pd"]:.2%}.</p>
<p>{len(result["alerts"])} alert features and {len(result["watch"])} watch features.
Thresholds are configurable heuristics.</p>{figure_image(fig, "Largest monitored feature PSI values")}</section>
<section><h2>Feature investigation</h2>{table(drift)}<p>Investigate source changes, missingness, portfolio mix,
and segment outcomes before attributing drift to a specific cause.</p></section>
<section><h2>Outcome availability</h2><p>{esc(result["performance_note"])}</p>
{table(pd.DataFrame([result["performance"]])) if result["performance"] else "<p>Performance metrics are intentionally omitted.</p>"}</section>"""
    (out / "report.html").write_text(
        shell(
            "Credit risk · Portfolio monitoring",
            "Reference-fitted drift bins / outcomes when available",
            content,
        ),
        encoding="utf-8",
    )
