# Canonical data contract

CSV, one row per account snapshot. All financial amounts must use the same currency. The six history fields are ordered from most recent (`1`) to oldest (`6`). All predictors must be available at the observation cutoff, before the next-month outcome is known.

| Field | Meaning | Validation |
|---|---|---|
| `account_id` | Unique customer/account key, never a predictor | Nonempty, unique |
| `credit_limit` | Granted limit | Present, finite, > 0 |
| `status_1` … `status_6` | Repayment status codes | Integers -2 through 9; missing allowed |
| `bill_1` … `bill_6` | Statement balance in currency units | Finite when present; negative credit balances allowed |
| `payment_1` … `payment_6` | Recorded payment amounts | Nonnegative; missing allowed |
| `default_next_month` | Next-month payment default outcome | Binary, no missing; optional only for score/monitor/scenario |

An entirely missing financial column is rejected. Partial missingness is reported and imputed with training medians. Credit limits cannot be missing because ratios need a valid denominator. Extra columns are reported and ignored by the feature allowlist; a temporal observation-date column may be used only for splitting.

The UCI adapter maps `PAY_0` to `status_1`, and `PAY_2` ... `PAY_6` to subsequent months. Negative and zero status values are kept as recorded, not redefined as delinquency months; only positive values count toward `delinquent_months`. Consult UCI's source documentation for coding interpretation. ID, sex, education, marital status and age are not model inputs.

Derived predictors: latest and average utilization, latest payment/absolute bill (denominator floor 1), mean payment/limit, count of positive delinquency statuses, maximum recorded delinquency, latest-minus-oldest bill/limit. Ratios may exceed 1; over-limit customers are not discarded.

No target-derived variables, post-outcome collections or future account statuses belong in the predictor fields. Schema checks cannot prove observation-time correctness: this is a source-data responsibility.
