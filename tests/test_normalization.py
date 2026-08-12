"""Tests for library sizes, CPM, and the log transform."""

from __future__ import annotations

import math

import pytest

from pancreatic_organoid.normalization import (
    compute_library_sizes,
    filter_nonzero_libraries,
    normalize_counts,
)

COUNTS_SCHEMA = "gene_id string, sample_uuid string, count double"


@pytest.mark.spark
def test_library_size_totals_counts_per_sample(spark):
    df = spark.createDataFrame(
        [
            ("ENSG1", "sample_a", 100.0),
            ("ENSG2", "sample_a", 900.0),
            ("ENSG1", "sample_b", 50.0),
        ],
        COUNTS_SCHEMA,
    )
    sizes = {r["sample_uuid"]: r["library_size"] for r in compute_library_sizes(df).collect()}
    assert sizes == {"sample_a": 1000.0, "sample_b": 50.0}


@pytest.mark.spark
def test_filter_nonzero_libraries_drops_uncovered_samples(spark):
    df = spark.createDataFrame(
        [("sample_a", 1000.0), ("sample_b", 0.0)], "sample_uuid string, library_size double"
    )
    kept = [r["sample_uuid"] for r in filter_nonzero_libraries(df).collect()]
    assert kept == ["sample_a"]


@pytest.mark.spark
def test_cpm_scales_counts_to_a_one_million_read_library(spark):
    """10 reads out of a 1,000-read library is 10,000 counts per million."""
    df = spark.createDataFrame(
        [("ENSG1", "sample_a", 10.0), ("ENSG2", "sample_a", 990.0)], COUNTS_SCHEMA
    )
    rows = {r["gene_id"]: r for r in normalize_counts(df).collect()}

    assert rows["ENSG1"]["library_size"] == 1000.0
    assert rows["ENSG1"]["cpm"] == pytest.approx(10_000.0)
    assert rows["ENSG2"]["cpm"] == pytest.approx(990_000.0)


@pytest.mark.spark
def test_cpm_values_sum_to_one_million_within_a_sample(spark):
    """The defining property of CPM: every sample totals 1e6."""
    df = spark.createDataFrame(
        [
            ("ENSG1", "sample_a", 1.0),
            ("ENSG2", "sample_a", 2.0),
            ("ENSG3", "sample_a", 7.0),
        ],
        COUNTS_SCHEMA,
    )
    total = sum(r["cpm"] for r in normalize_counts(df).collect())
    assert total == pytest.approx(1_000_000.0)


@pytest.mark.spark
def test_log1p_cpm_is_natural_log_of_one_plus_cpm(spark):
    """Note this is ln(1 + CPM), not the log2(CPM + 1) used by edgeR/limma."""
    df = spark.createDataFrame(
        [("ENSG1", "sample_a", 10.0), ("ENSG2", "sample_a", 990.0)], COUNTS_SCHEMA
    )
    rows = {r["gene_id"]: r for r in normalize_counts(df).collect()}
    assert rows["ENSG1"]["log1p_cpm"] == pytest.approx(math.log1p(10_000.0))


@pytest.mark.spark
def test_zero_counts_normalize_to_zero_expression(spark):
    df = spark.createDataFrame(
        [("ENSG1", "sample_a", 0.0), ("ENSG2", "sample_a", 1000.0)], COUNTS_SCHEMA
    )
    rows = {r["gene_id"]: r for r in normalize_counts(df).collect()}
    assert rows["ENSG1"]["cpm"] == 0.0
    assert rows["ENSG1"]["log1p_cpm"] == 0.0


@pytest.mark.spark
def test_samples_with_zero_library_size_are_excluded(spark):
    """Guards against dividing by zero when computing CPM."""
    df = spark.createDataFrame(
        [("ENSG1", "sample_a", 100.0), ("ENSG1", "sample_empty", 0.0)], COUNTS_SCHEMA
    )
    samples = {r["sample_uuid"] for r in normalize_counts(df).collect()}
    assert samples == {"sample_a"}
