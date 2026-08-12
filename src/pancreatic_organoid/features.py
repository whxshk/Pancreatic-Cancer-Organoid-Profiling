"""Gold layer feature engineering: per-sample and per-gene summaries.

Two summary tables are derived from the normalized counts.

**Sample features** describe sequencing quality for each organoid: how deeply
it was sequenced and how much of the transcriptome was detected. RNA-seq count
matrices are heavily zero-inflated, so the proportion of zero-count genes is a
useful quality signal — an unusually high value normally means shallow
sequencing or degraded input material.

**Gene features** describe each gene's behaviour across samples: its average
expression, how variable it is, and in how many samples it was detected at all.
Genes detected in very few samples are typically excluded from downstream
analysis.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SAMPLE_COLUMN = "sample_uuid"
COUNT_COLUMN = "count"
LOG_CPM_COLUMN = "log1p_cpm"
LIBRARY_SIZE_COLUMN = "library_size"


def _count_where(condition) -> "F.Column":
    """Counts rows satisfying a condition inside an aggregation.

    ``F.sum(condition)`` fails in Spark because ``SUM`` rejects boolean input,
    so the condition is mapped to 1/0 first.
    """
    return F.sum(F.when(condition, 1).otherwise(0))


def build_sample_features(
    df_norm: DataFrame,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Summarises sequencing quality for each sample.

    Returns:
        One row per sample with ``library_size``, ``avg_log1p_cpm``,
        ``num_zero_genes``, ``num_detected_genes``, and ``pct_zero_genes``.

    Note:
        ``library_size`` is taken with ``F.first`` rather than recomputed,
        because it was already attached to every row during normalization and
        is constant within a sample.
    """
    return df_norm.groupBy(sample_column).agg(
        F.first(LIBRARY_SIZE_COLUMN).alias(LIBRARY_SIZE_COLUMN),
        F.avg(LOG_CPM_COLUMN).alias("avg_log1p_cpm"),
        _count_where(F.col(count_column) == 0).alias("num_zero_genes"),
        _count_where(F.col(count_column) > 0).alias("num_detected_genes"),
    ).withColumn(
        "pct_zero_genes",
        F.col("num_zero_genes") / (F.col("num_zero_genes") + F.col("num_detected_genes")),
    )


def build_gene_features(
    df_norm: DataFrame,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Summarises expression level, variability, and detection for each gene.

    Returns:
        One row per gene with ``mean_log1p_cpm``, ``std_log1p_cpm``,
        ``num_samples_detected``, ``num_samples_total``, and
        ``pct_samples_detected``.

    Note:
        ``std_log1p_cpm`` is null for genes present in a single sample, since
        the sample standard deviation is undefined for one observation.
    """
    return df_norm.groupBy("gene_id", "gene_name").agg(
        F.avg(LOG_CPM_COLUMN).alias("mean_log1p_cpm"),
        F.stddev(LOG_CPM_COLUMN).alias("std_log1p_cpm"),
        _count_where(F.col(count_column) > 0).alias("num_samples_detected"),
        F.count("*").alias("num_samples_total"),
    ).withColumn(
        "pct_samples_detected",
        F.col("num_samples_detected") / F.col("num_samples_total"),
    )
