# Model Reliability Lab

An end-to-end classification project focused on the parts that are often missing from portfolio notebooks: isolated validation and test sets, probability calibration, model comparison, cost-aware threshold selection, cohort stability, simulated feature drift, permutation importance, and a generated model card.

The project uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn. It is strictly educational and does not provide medical advice.

## What it demonstrates

- stratified 60/20/20 train, validation, and held-out test partitions;
- calibrated logistic regression compared with a class-weighted random forest;
- model selection using validation Brier score and average precision;
- an explicit 8:1 false-negative to false-positive threshold cost;
- ROC AUC, average precision, Brier score, threshold metrics, and confusion counts;
- measurement-defined cohort checks and simulated drift monitoring;
- permutation importance plus JSON model-card and prediction artifacts.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -m unittest discover -s tests -v
python run_demo.py
```

Generated outputs:

- `demo_summary.json` - compact evidence used by the project page;
- `artifacts/model_card.json` - assumptions, split sizes, metrics, drift, and importance;
- `artifacts/test_predictions.csv` - held-out probabilities and threshold decisions.

## Honest scope

The public dataset is small and well studied. Metrics show this implementation's behavior on one deterministic split, not clinical validity or expected deployment performance. The drift scenario is intentionally simulated to prove monitoring logic.

See [ENGINEERING.md](ENGINEERING.md) for architecture, leakage controls, the verification contract, and change-safety rules.
