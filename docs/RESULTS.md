# Verified results

Run completed 29 October 2026 using the provided default configuration and seed 42. These are measured results, not targets or estimated achievements.

## Public-data model

Source: Yeh, I. (2009), [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H), UCI, CC BY 4.0. Population: Taiwanese credit-card customers from 2005. No Citi data or US portfolio data was used.

- Accounts: **30,000**.
- Split: **16,500 train / 4,500 calibration / 4,500 validation / 4,500 test**.
- Selected candidate: **sigmoid-calibrated histogram gradient boosting**.
- Selection criterion: lowest **validation Brier score**, with log loss as tie-breaker.
- Predictor inputs: financial history only; demographics and account IDs excluded.

| Untouched test metric | Value |
|---|---:|
| ROC-AUC | 0.7708 |
| ROC-AUC bootstrap 95% interval | 0.7552–0.7889 |
| Average precision | 0.5367 |
| KS | 0.4144 |
| Brier score | 0.1376 |
| Constant training-prevalence baseline Brier | 0.1723 |
| Log loss | 0.4369 |
| Observed default rate | 22.13% |
| Mean predicted probability | 22.36% |
| Selected review threshold | 0.17 |
| Recall at that threshold | 74.40% |
| Precision at that threshold | 36.15% |

The review threshold minimizes validation error costs with a false-negative cost of 5 and a false-positive cost of 1. These cost units are assumptions, not bank economics. Test false positives: 1,309; false negatives: 255. The model's recall comes with a substantial review volume; do not report recall without the precision/threshold context.

The raw boosting model's validation Brier was 0.133441; calibration changed it to 0.133429. This improvement is very small: do not claim calibration produced a major improvement. The four-way split was retained rather than refitting on the test cohort.

SQL/pandas reconciliation passed for account counts, distinct accounts, predicted defaults, exposure proxy, PD-weighted exposure, review flags, and observed defaults. These are checks on the same scored dataset, not an external source-system reconciliation.

Full evidence is in `runs/uci/test_metrics.json`, `runs/uci/validation_comparison.csv`, `runs/uci/manifest.json`, and `runs/uci/report.html`.

## Synthetic monitoring demonstration

`runs/demo` uses 6,000 synthetic development accounts plus 2,000 independently generated shifted accounts. The synthetic selected model is logistic regression, with test AUC 0.7620 and Brier 0.1412. These metrics are deliberately separate from the public-data result. The monitoring report demonstrates reference-fitted PSI, missingness tracking, and segment outcomes on a simulated shift.

## Resume bullets supported by this implementation

**Credit Risk Modeling & Validation Toolkit | Python, SQL, scikit-learn**

- Built a reproducible CLI pipeline for default-risk modeling on 30,000 public credit-card records, comparing logistic regression and gradient boosting with separate training, calibration, validation, and test partitions.
- Achieved 0.771 ROC-AUC and 0.138 Brier score on 4,500 held-out accounts; generated calibration, KS, bootstrap uncertainty, and segment-level validation reports.
- Automated SQL/pandas portfolio reconciliation and PSI-based monitoring, with data-integrity checks, source/model provenance, and portable HTML reports.

These bullets describe a local research project. They do not claim deployment, regulatory compliance, US representativeness, or business loss reduction. The public-data holdout is random within one cohort; it is not an out-of-time validation.
