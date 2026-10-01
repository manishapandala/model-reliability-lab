"""Reproducible classification, calibration, thresholding, and drift checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 37
POSITIVE_LABEL = 0  # malignant in the sklearn dataset


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    average_precision: float
    brier_score: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the public diagnostic dataset bundled with scikit-learn."""
    bundle = load_breast_cancer(as_frame=True)
    features = bundle.data.copy()
    target = (bundle.target == POSITIVE_LABEL).astype(int)
    target.name = "high_risk"
    return features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified 60/20/20 train, validation, and test partitions."""
    train_x, remainder_x, train_y, remainder_y = train_test_split(
        features,
        target,
        test_size=0.40,
        stratify=target,
        random_state=random_state,
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        remainder_x,
        remainder_y,
        test_size=0.50,
        stratify=remainder_y,
        random_state=random_state,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def build_candidates(random_state: int = RANDOM_STATE) -> dict[str, object]:
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5_000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return {
        "calibrated_logistic": CalibratedClassifierCV(logistic, method="sigmoid", cv=5),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def positive_probability(model: object, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(1)]


def select_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
    false_negative_cost: float = 8.0,
    false_positive_cost: float = 1.0,
) -> tuple[float, float]:
    """Choose a validation threshold using an explicit asymmetric cost."""
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        del tn, tp
        cost = false_negative_cost * fn + false_positive_cost * fp
        if cost < best_cost:
            best_threshold = float(threshold)
            best_cost = float(cost)
    return round(best_threshold, 3), best_cost


def evaluate(labels: pd.Series, probabilities: np.ndarray, threshold: float) -> ClassificationMetrics:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return ClassificationMetrics(
        roc_auc=round(float(roc_auc_score(labels, probabilities)), 4),
        average_precision=round(float(average_precision_score(labels, probabilities)), 4),
        brier_score=round(float(brier_score_loss(labels, probabilities)), 4),
        precision=round(float(precision_score(labels, predictions, zero_division=0)), 4),
        recall=round(float(recall_score(labels, predictions, zero_division=0)), 4),
        f1=round(float(f1_score(labels, predictions, zero_division=0)), 4),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def population_stability_index(reference: pd.Series, observed: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference quantile bins with stable edge handling."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    observed_counts = np.histogram(observed, bins=edges)[0].astype(float)
    reference_share = np.clip(reference_counts / reference_counts.sum(), 1e-6, None)
    observed_share = np.clip(observed_counts / observed_counts.sum(), 1e-6, None)
    psi = np.sum((observed_share - reference_share) * np.log(observed_share / reference_share))
    return round(float(psi), 4)


def cohort_metrics(
    model: object,
    features: pd.DataFrame,
    labels: pd.Series,
    threshold: float,
    split_value: float,
) -> dict[str, dict]:
    """Evaluate measurement-defined cohorts; this is not demographic fairness analysis."""
    groups = {
        "lower_mean_radius": features["mean radius"] <= split_value,
        "higher_mean_radius": features["mean radius"] > split_value,
    }
    output = {}
    for name, mask in groups.items():
        output[name] = {
            "rows": int(mask.sum()),
            **evaluate(labels.loc[mask], positive_probability(model, features.loc[mask]), threshold).to_dict(),
        }
    return output


def run_experiment(random_state: int = RANDOM_STATE) -> tuple[dict, pd.DataFrame]:
    features, target = load_dataset()
    train_x, validation_x, test_x, train_y, validation_y, test_y = split_dataset(
        features, target, random_state
    )

    candidates = build_candidates(random_state)
    validation_scores = {}
    for name, model in candidates.items():
        model.fit(train_x, train_y)
        probabilities = positive_probability(model, validation_x)
        validation_scores[name] = {
            "average_precision": round(float(average_precision_score(validation_y, probabilities)), 4),
            "brier_score": round(float(brier_score_loss(validation_y, probabilities)), 4),
        }

    selected_name = min(
        validation_scores,
        key=lambda name: (
            validation_scores[name]["brier_score"],
            -validation_scores[name]["average_precision"],
        ),
    )
    selected_model = candidates[selected_name]
    validation_probabilities = positive_probability(selected_model, validation_x)
    threshold, validation_cost = select_threshold(validation_y, validation_probabilities)

    test_probabilities = positive_probability(selected_model, test_x)
    test_metrics = evaluate(test_y, test_probabilities, threshold)
    radius_split = float(train_x["mean radius"].median())

    shifted_test = test_x.copy()
    shifted_test["mean radius"] *= 1.18
    shifted_test["mean texture"] *= 1.12
    shifted_probabilities = positive_probability(selected_model, shifted_test)
    drift = {
        "mean_radius_psi": population_stability_index(train_x["mean radius"], shifted_test["mean radius"]),
        "mean_texture_psi": population_stability_index(
            train_x["mean texture"], shifted_test["mean texture"]
        ),
        "mean_probability_change": round(
            float(shifted_probabilities.mean() - test_probabilities.mean()), 4
        ),
    }

    importance = permutation_importance(
        selected_model,
        test_x,
        test_y,
        scoring="average_precision",
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    top_features = sorted(
        zip(test_x.columns, importance.importances_mean, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:8]

    predictions = test_x.copy()
    predictions["actual_high_risk"] = test_y.to_numpy()
    predictions["predicted_probability"] = np.round(test_probabilities, 6)
    predictions["predicted_high_risk"] = (test_probabilities >= threshold).astype(int)

    summary = {
        "scope": "educational model-reliability evaluation on a public diagnostic dataset",
        "dataset_rows": len(features),
        "features": features.shape[1],
        "split_rows": {
            "train": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "candidate_validation": validation_scores,
        "selected_model": selected_name,
        "decision_threshold": threshold,
        "false_negative_cost": 8.0,
        "false_positive_cost": 1.0,
        "validation_cost": validation_cost,
        "test_metrics": test_metrics.to_dict(),
        "cohort_stability": cohort_metrics(
            selected_model, test_x, test_y, threshold, radius_split
        ),
        "simulated_drift": drift,
        "top_permutation_features": [
            {"feature": name, "importance": round(float(value), 5)} for name, value in top_features
        ],
    }
    return summary, predictions
