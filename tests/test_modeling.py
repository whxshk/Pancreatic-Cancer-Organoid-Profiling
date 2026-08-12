"""Tests for the modeling feature matrix and the regression metrics.

These tests cover code written in this project. They deliberately do not assert
the behaviour of scikit-learn's PCA, K-Means, or Random Forest implementations.

The last test is not a correctness check but an executable statement of a known
limitation: the supervised target is an exact linear function of one of its own
features. See the warning at the top of ``modeling.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pancreatic_organoid.modeling import (
    build_modeling_features,
    regression_metrics,
    to_feature_matrix,
)

NORM_SCHEMA = "sample_uuid string, gene_id string, count double"


# --- Pure Python: no Spark required -----------------------------------------


def test_regression_metrics_match_hand_computed_values():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])

    metrics = regression_metrics(y_true, y_pred)

    # Every residual is +/-10, so MAE = 10, MSE = 100, RMSE = 10.
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mse"] == pytest.approx(100.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    # SS_res = 400, SS_tot = 50000 -> R^2 = 1 - 400/50000
    assert metrics["r2"] == pytest.approx(0.992)
    # mean(10/100, 10/200, 10/300, 10/400) * 100
    assert metrics["mape"] == pytest.approx(5.2083333, rel=1e-6)


def test_regression_metrics_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y)

    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx(0.0)


def test_regression_metrics_rejects_zero_targets():
    """MAPE divides by the true value, so zero targets are rejected outright."""
    with pytest.raises(ValueError, match="MAPE"):
        regression_metrics(np.array([0.0, 1.0]), np.array([0.5, 1.0]))


def test_to_feature_matrix_selects_features_and_target_in_order():
    pdf = pd.DataFrame(
        {
            "sample_uuid": ["a"],
            "library_size": [1000.0],
            "avg_count": [250.0],
            "num_zero_genes": [1.0],
            "num_detected_genes": [3.0],
            "pct_zero_genes": [0.25],
        }
    )
    X, y = to_feature_matrix(pdf)

    assert X.shape == (1, 4)
    # Column order must match FEATURE_COLUMNS.
    assert list(X[0]) == [250.0, 1.0, 3.0, 0.25]
    assert list(y) == [1000.0]


def test_to_feature_matrix_reports_missing_columns():
    with pytest.raises(KeyError, match="avg_count"):
        to_feature_matrix(pd.DataFrame({"sample_uuid": ["a"]}))


# --- Spark-backed feature construction --------------------------------------


@pytest.mark.spark
def test_build_modeling_features_aggregates_per_sample(spark):
    df = spark.createDataFrame(
        [
            ("sample_a", "ENSG1", 100.0),
            ("sample_a", "ENSG2", 300.0),
            ("sample_a", "ENSG3", 0.0),
            ("sample_a", "ENSG4", 0.0),
        ],
        NORM_SCHEMA,
    )
    row = build_modeling_features(df).collect()[0]

    assert row["library_size"] == pytest.approx(400.0)
    assert row["avg_count"] == pytest.approx(100.0)  # 400 / 4 genes
    assert row["num_zero_genes"] == pytest.approx(2.0)
    assert row["num_detected_genes"] == pytest.approx(2.0)
    assert row["pct_zero_genes"] == pytest.approx(0.5)


@pytest.mark.spark
def test_target_is_a_linear_function_of_avg_count(spark):
    """Documents the known collinearity between the target and a feature.

    Because the Silver unpivot gives every sample a row for every gene, the
    gene count per sample is constant, so ``library_size == avg_count * N``
    exactly. The Random Forest therefore predicts a rescaled copy of one of its
    inputs, and its reported R^2 does not measure genuine predictive power.
    """
    df = spark.createDataFrame(
        [
            ("sample_a", "ENSG1", 100.0),
            ("sample_a", "ENSG2", 300.0),
            ("sample_a", "ENSG3", 0.0),
            ("sample_b", "ENSG1", 50.0),
            ("sample_b", "ENSG2", 25.0),
            ("sample_b", "ENSG3", 0.0),
        ],
        NORM_SCHEMA,
    )
    rows = build_modeling_features(df).collect()
    genes_per_sample = 3

    for row in rows:
        assert row["library_size"] == pytest.approx(row["avg_count"] * genes_per_sample)
        # The gene count is itself recoverable from the other two features.
        assert row["num_zero_genes"] + row["num_detected_genes"] == genes_per_sample
