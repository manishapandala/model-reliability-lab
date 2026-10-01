# Engineering Review

## Architecture

`reliability.py` owns dataset loading, stratified partitioning, candidate models, calibration-aware selection, cost-based thresholding, evaluation, cohort stability, drift measurement, and permutation importance. `run_demo.py` only orchestrates the experiment and writes artifacts. `index.html` only presents the generated evidence.

The validation set selects the model and decision threshold. The test set remains untouched until final evaluation. Drift uses a separate simulated copy of the test features.

## Verification contract

```bash
python -m ruff check .
python -m compileall -q .
python -m unittest discover -s tests -v
python run_demo.py
```

## Change-safety checklist

- Reject any split or feature transform that uses test labels during training or selection.
- Review false-negative and false-positive costs as explicit product assumptions.
- Keep probability calibration, ranking metrics, threshold metrics, and confusion counts distinct.
- Require a reproducible failing test before changing model-selection logic.
- Inspect performance across measurement-defined cohorts; do not call them demographic fairness groups.
- Keep the medical dataset educational and never present outputs as diagnosis.
- Compare runtime, Brier score, average precision, and drift response after model changes.

## Boundaries

This project demonstrates evaluation discipline on a small public dataset. It is not a medical device, clinical recommendation, external validation study, or evidence of real-world deployment performance.
