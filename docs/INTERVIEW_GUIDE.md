# Explain and demonstrate the project

## 60-second explanation

"I built a reproducible command-line credit-risk workbench. It compares logistic regression with gradient boosting, calibrates default probabilities on separate data, and selects a model using validation Brier score. An untouched test split measures discrimination and probability quality. The pipeline also reconciles portfolio reports using SQL, diagnoses segment-level calibration, and monitors changes in incoming data. I used a public Taiwanese credit-card dataset and kept its population and temporal limitations explicit."

## Five-minute demo

1. Show `creditrisk --help` and run `validate-data` against the real dataset.
2. Open `runs/uci/report.html`: explain the four partitions and why selection happens before test evaluation.
3. Compare logistic and boosting validation Brier scores; explain why the highest AUC is not necessarily the best probability estimator.
4. Discuss a calibration gap and a segment result. Mention uncertainty and small samples.
5. Show `audit/reconciliation.sql` and its PASS result.
6. Show the synthetic monitoring report and explain that the shift is simulated, not empirical evidence from the UCI cohort.

## Questions to prepare

- **Why Brier score?** It measures squared probability error and captures both calibration and discrimination; it is not a pure calibration metric. Reliability curves provide the calibration view.
- **What is ROC-AUC?** Ranking ability over thresholds. It says nothing by itself about whether a 20% PD corresponds to a 20% default frequency.
- **What is KS?** Maximum difference between true-positive and false-positive rates along the ROC curve in this implementation.
- **Why average precision?** It summarizes the precision–recall trade-off; compare it with the observed default prevalence, and explain prevalence dependence.
- **Why four partitions?** Fit model parameters on train, calibration mapping on calibration, candidate and threshold choices on validation, and final metrics on test.
- **Why no SMOTE?** This baseline prioritizes interpretable probability estimation at the observed prevalence. Rebalancing requires careful correction/calibration and separate validation.
- **What is sigmoid calibration?** A logistic mapping of clipped model log-odds to observed outcomes, fitted on a separate partition.
- **How do you prevent leakage?** Predictor allowlist; train-only preprocessing; unique account IDs; disjoint partitions; evaluation overlap rejection; optional label-window purging. Source-time errors still require data review.
- **Why no temporal result on UCI?** It is a single customer cohort with six historical fields. Assigning artificial dates would create misleading out-of-time claims.
- **Does PSI prove the model failed?** No. It detects distribution differences, is bin-dependent, and uses heuristic alert thresholds. Investigate changes and mature outcomes.
- **What does SQL reconciliation establish?** Independent arithmetic on the same scored dataset; it does not prove upstream source correctness.
- **Is this CECL?** No. The optional PD × assumed LGD × bill-balance sensitivity is one-month and illustrative, not lifetime expected loss accounting.
- **Is the model fair?** Not established. Removing demographic inputs does not eliminate proxies, and separate fairness validation would be required.
- **What would production require?** Mature time-stamped cohorts, independent validation, realistic economics, access controls, secure artifacts, fairness/governance review, lineage, monitoring ownership and controlled deployment.

## Resume wording

Use the measured values in `docs/RESULTS.md`; do not claim production deployment, Citi data, US validation, regulatory approval, business savings or loss reduction. Read and understand the implementation before presenting it as your project.
