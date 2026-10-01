"""Tests for data splitting, evaluation, drift, and experiment reproducibility."""

import unittest

import numpy as np
import pandas as pd

from reliability import (
    evaluate,
    load_dataset,
    population_stability_index,
    run_experiment,
    select_threshold,
    split_dataset,
)


class ReliabilityTests(unittest.TestCase):
    def test_public_dataset_contract(self):
        features, target = load_dataset()
        self.assertEqual(features.shape, (569, 30))
        self.assertEqual(set(target.unique()), {0, 1})

    def test_split_is_disjoint_and_complete(self):
        features, target = load_dataset()
        train_x, validation_x, test_x, *_ = split_dataset(features, target)
        self.assertEqual(len(train_x) + len(validation_x) + len(test_x), len(features))
        self.assertFalse(set(train_x.index) & set(test_x.index))
        self.assertFalse(set(validation_x.index) & set(test_x.index))

    def test_split_is_deterministic(self):
        features, target = load_dataset()
        first = split_dataset(features, target)[0].index.tolist()
        second = split_dataset(features, target)[0].index.tolist()
        self.assertEqual(first, second)

    def test_threshold_reflects_false_negative_cost(self):
        labels = pd.Series([0, 0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.4, 0.45, 0.8])
        threshold, _ = select_threshold(labels, probabilities, false_negative_cost=20)
        self.assertLessEqual(threshold, 0.45)

    def test_metrics_include_confusion_counts(self):
        labels = pd.Series([0, 0, 1, 1])
        result = evaluate(labels, np.array([0.1, 0.7, 0.8, 0.9]), 0.5)
        self.assertEqual(result.true_positive, 2)
        self.assertEqual(result.false_positive, 1)

    def test_psi_is_zero_for_identical_samples(self):
        values = pd.Series(np.linspace(0, 1, 100))
        self.assertEqual(population_stability_index(values, values), 0.0)

    def test_invalid_psi_bins_are_rejected(self):
        values = pd.Series([1, 2, 3])
        with self.assertRaises(ValueError):
            population_stability_index(values, values, bins=1)

    def test_experiment_is_reproducible(self):
        first, _ = run_experiment()
        second, _ = run_experiment()
        self.assertEqual(first, second)
        self.assertEqual(first["split_rows"]["test"], 114)


if __name__ == "__main__":
    unittest.main()
