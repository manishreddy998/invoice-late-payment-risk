# Small business late payment risk model with release gates

**Problem.** Wave customers are small businesses, and an invoice paid 60+ days late hurts their cash flow. This prototype scores invoice late payment risk and adds MLOps gates, so a weak model or drifted data cannot ship.

**Architecture.** `risk/data.py` generates seeded invoices from a known logistic process. Because the true probabilities are known, we can compute the best possible AUC. `risk/train.py` trains logistic regression and gradient boosting on 20,000 invoices, tests on 8,000, and applies release gates. The model must be within 0.02 AUC of that best possible AUC, the Brier score must be at most 0.20, and PSI on invoice amount must be at most 0.2. A batch with shifted amounts checks that the drift gate fires.

**Results** (`python -m risk.train`, results.json)

| item | value |
|---|---|
| late rate | 19.7% |
| best possible AUC (true probabilities) | 0.6967 |
| logistic AUC / Brier | 0.6939 / 0.1441, released |
| gradient boosting AUC / Brier | 0.6906 / 0.1450, released |
| drifted batch PSI on amount | 0.625, blocked |

**What I learned.** My first gate was a fixed AUC of 0.70, and it blocked both models. The ceiling was 0.6967, so no model could pass. Gating against the measured ceiling is the honest fix. The simple logistic model wins because the data really is logistic. Real data would need SageMaker training jobs, Airflow scheduling and feature lineage, which are not built here.

Tests: `pytest -q` (4 tests).
