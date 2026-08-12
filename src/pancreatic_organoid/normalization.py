"""Normalization: convert raw counts to CPM and a log-scaled expression value.

Raw RNA-seq counts are not comparable between samples. A gene with 500 reads in
a sample sequenced to 5 million reads and 500 reads in a sample sequenced to 1
million reads represents very different expression levels. Counts Per Million
divides out that sequencing-depth difference::

    cpm = count * 1e6 / library_size

Expression is also heavily right-skewed: a handful of genes dominate the total
while most sit near zero. A log transform compresses that range so distances
and variances behave sensibly for PCA and clustering. ``log1p`` is used so that
zero counts map to zero instead of negative infinity.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SAMPLE_COLUMN = "sample_uuid"
COUNT_COLUMN = "count"
LIBRARY_SIZE_COLUMN = "library_size"
CPM_COLUMN = "cpm"
LOG_CPM_COLUMN = "log1p_cpm"

_COUNTS_PER_MILLION_SCALE = 1e6


def compute_library_sizes(
    df: DataFrame,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Totals the counts per sample, i.e. its sequencing depth."""
    return df.groupBy(sample_column).agg(F.sum(count_column).alias(LIBRARY_SIZE_COLUMN))


def filter_nonzero_libraries(df_totals: DataFrame) -> DataFrame:
    """Drops samples whose total count is zero.

    These are samples with no usable coverage. They must be removed before CPM
    is computed, because dividing by a zero library size yields null or
    infinity for every gene in the sample.
    """
    return df_totals.filter(F.col(LIBRARY_SIZE_COLUMN) > 0)


def add_cpm(
    df_counts: DataFrame,
    df_totals: DataFrame,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Joins library sizes onto the counts and derives CPM and log1p(CPM).

    The join is an inner join, so samples filtered out of ``df_totals`` by
    :func:`filter_nonzero_libraries` are excluded from the result entirely.
    """
    return (
        df_counts.join(df_totals, on=sample_column, how="inner")
        .withColumn(
            CPM_COLUMN,
            (F.col(count_column) * _COUNTS_PER_MILLION_SCALE) / F.col(LIBRARY_SIZE_COLUMN),
        )
        .withColumn(LOG_CPM_COLUMN, F.log1p(CPM_COLUMN))
    )


def normalize_counts(
    df_counts: DataFrame,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Runs the full normalization step on an annotated counts table.

    Returns:
        The input columns plus ``library_size``, ``cpm``, and ``log1p_cpm``,
        restricted to samples with a non-zero library size.
    """
    totals = filter_nonzero_libraries(
        compute_library_sizes(df_counts, sample_column, count_column)
    )
    return add_cpm(df_counts, totals, sample_column, count_column)
