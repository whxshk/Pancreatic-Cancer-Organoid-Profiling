"""Machine learning on sample-level features: PCA, K-Means, Random Forest.

.. warning::

   **The supervised target is algebraically derivable from its own features.**

   Every sample in the normalized table has a row for every gene, so the number
   of rows per sample, ``N``, is identical across samples. That makes::

       library_size = sum(count)          <- the target
       avg_count    = avg(count)          <- a feature, and exactly library_size / N

   so ``library_size == avg_count * N``. The model is predicting a rescaled
   copy of one of its own inputs, and the reported R^2 of 0.9719 measures that
   relationship rather than any real predictive capability.

   The notebook output confirms it independently: PCA on the four features
   reports explained variance ``[0.842, 0.158]``, which sums to exactly 1.0.
   Two components can only capture all variance if the four features span a
   two-dimensional space, which is what the identity above implies.

   This code is preserved as originally written. The limitation is documented
   rather than corrected, because it reflects the real constraint behind the
   modeling: the open-access tier of the source dataset carries no drug
   response or clinical outcome labels, so the only available targets were
   quality-control metrics derived from the counts themselves. See the
   "Limitations" section of the README.

The unsupervised part (PCA and K-Means) does not have this problem in the same
way. It describes how samples group by sequencing-quality profile, which is a
legitimate descriptive result, though the same collinearity means the structure
it finds is essentially two-dimensional by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SAMPLE_COLUMN = "sample_uuid"
COUNT_COLUMN = "count"

#: Predictors used by the supervised model. See the module warning: these are
#: not independent of the target.
FEATURE_COLUMNS = (
    "avg_count",
    "num_zero_genes",
    "num_detected_genes",
    "pct_zero_genes",
)
TARGET_COLUMN = "library_size"

N_COMPONENTS = 2
N_CLUSTERS = 3
N_ESTIMATORS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class UnsupervisedResult:
    """Output of the PCA and K-Means step."""

    components: np.ndarray
    clusters: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class SupervisedResult:
    """Output of the Random Forest step."""

    model: RandomForestRegressor
    metrics: dict[str, float]
    predictions_all: np.ndarray


def build_modeling_features(
    df_norm: DataFrame,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Aggregates normalized counts into one row per sample for modeling.

    This is deliberately kept separate from
    :func:`pancreatic_organoid.features.build_sample_features`. Both summarise
    samples, but the Gold QC table averages ``log1p_cpm`` while the modeling
    matrix averages the raw ``count``; the notebook computed them separately
    and they are not interchangeable.

    Returns:
        ``sample_uuid``, ``library_size``, ``avg_count``, ``num_zero_genes``,
        ``num_detected_genes``, ``pct_zero_genes``.
    """
    return df_norm.groupBy(sample_column).agg(
        F.sum(count_column).alias(TARGET_COLUMN),
        F.avg(count_column).alias("avg_count"),
        F.sum(F.when(F.col(count_column) == 0, 1.0).otherwise(0.0)).alias("num_zero_genes"),
        F.sum(F.when(F.col(count_column) > 0, 1.0).otherwise(0.0)).alias("num_detected_genes"),
    ).withColumn(
        "pct_zero_genes",
        F.col("num_zero_genes") / (F.col("num_zero_genes") + F.col("num_detected_genes")),
    )


def to_feature_matrix(pdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Splits the sample table into a feature matrix and target vector.

    Raises:
        KeyError: If any expected feature or target column is absent.
    """
    missing = [c for c in (*FEATURE_COLUMNS, TARGET_COLUMN) if c not in pdf.columns]
    if missing:
        raise KeyError(f"Sample feature table is missing column(s): {missing}")
    return pdf[list(FEATURE_COLUMNS)].values, pdf[TARGET_COLUMN].values


def fit_pca_kmeans(
    X: np.ndarray,
    n_components: int = N_COMPONENTS,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> UnsupervisedResult:
    """Standardizes the features, projects to 2 components, and clusters.

    Features are standardized first because they are on wildly different scales
    (``library_size`` runs to millions while ``pct_zero_genes`` is bounded by
    1). Both PCA and K-Means are distance-based, so without scaling the
    large-magnitude features would dominate entirely.

    Note:
        K-Means is fitted on the **standardized features**, not on the PCA
        components. The Phase II write-up describes it as clustering the
        PCA-reduced space; the code has always clustered the scaled features
        directly. Preserved as written.
    """
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    return UnsupervisedResult(
        components=components,
        clusters=clusters,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes R^2, MAE, MSE, RMSE, and MAPE.

    Raises:
        ValueError: If ``y_true`` contains a zero, which would make MAPE
            undefined. In this pipeline that cannot happen, because samples
            with a zero library size are removed during normalization.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when y_true contains zeros.")
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train_library_size_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> SupervisedResult:
    """Fits a Random Forest regressor and evaluates it on a held-out split.

    A Random Forest was chosen because it handles non-linear relationships,
    needs no feature scaling, and is robust to outliers — all reasonable for
    count-derived features. Note that it is fitted on the **unscaled** matrix,
    which is fine for a tree ensemble.

    Metrics are computed on the test split only. ``predictions_all`` covers
    every sample, including training rows, because the notebook wrote
    predictions for all samples into the Gold table; those values are
    in-sample for roughly 80% of rows and should not be read as held-out
    performance.

    See the module-level warning before interpreting the metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    return SupervisedResult(
        model=model, metrics=metrics, predictions_all=model.predict(X)
    )


def build_prediction_table(
    pdf: pd.DataFrame,
    unsupervised: UnsupervisedResult,
    supervised: SupervisedResult,
    y: np.ndarray,
) -> pd.DataFrame:
    """Assembles the Gold ``sample_predictions`` table.

    Model-level metrics are repeated on every row. That is redundant in a
    relational sense but is what makes them usable as Power BI card values.
    """
    result = pdf.copy()
    result["PC1"] = unsupervised.components[:, 0]
    result["PC2"] = unsupervised.components[:, 1]
    result["cluster"] = unsupervised.clusters
    result["pred_library_size"] = supervised.predictions_all
    result["target_library_size"] = y
    result["model_r2"] = supervised.metrics["r2"]
    result["model_rmse"] = supervised.metrics["rmse"]
    return result


def run_modeling(df_norm: DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Runs the full modeling step and returns the table and its metrics.

    The Spark to pandas conversion here is safe: ``build_modeling_features``
    reduces millions of gene/sample rows to one row per sample (110 in the
    analysed dataset), so what reaches the driver is a table of about a hundred
    rows. The heavy aggregation stays distributed.
    """
    pdf = build_modeling_features(df_norm).toPandas()
    X, y = to_feature_matrix(pdf)
    unsupervised = fit_pca_kmeans(X)
    supervised = train_library_size_model(X, y)
    return build_prediction_table(pdf, unsupervised, supervised, y), supervised.metrics
