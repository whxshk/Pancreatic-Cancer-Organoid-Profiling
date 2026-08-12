"""Tests for the Gold per-sample and per-gene feature tables."""

from __future__ import annotations

import math

import pytest

from pancreatic_organoid.features import build_gene_features, build_sample_features

NORM_SCHEMA = (
    "sample_uuid string, gene_id string, gene_name string, count double, "
    "library_size double, cpm double, log1p_cpm double"
)


def _row(sample, gene, name, count, library_size, cpm, log1p_cpm):
    return (sample, gene, name, count, library_size, cpm, log1p_cpm)


@pytest.fixture
def normalized(spark):
    """Two samples, four genes each, with a known number of zero counts.

    sample_a: 1 zero of 4 genes -> 25% zeros
    sample_b: 3 zeros of 4 genes -> 75% zeros
    """
    rows = [
        _row("sample_a", "ENSG1", "GENE1", 100.0, 400.0, 250_000.0, math.log1p(250_000.0)),
        _row("sample_a", "ENSG2", "GENE2", 100.0, 400.0, 250_000.0, math.log1p(250_000.0)),
        _row("sample_a", "ENSG3", "GENE3", 200.0, 400.0, 500_000.0, math.log1p(500_000.0)),
        _row("sample_a", "ENSG4", "GENE4", 0.0, 400.0, 0.0, 0.0),
        _row("sample_b", "ENSG1", "GENE1", 500.0, 500.0, 1_000_000.0, math.log1p(1_000_000.0)),
        _row("sample_b", "ENSG2", "GENE2", 0.0, 500.0, 0.0, 0.0),
        _row("sample_b", "ENSG3", "GENE3", 0.0, 500.0, 0.0, 0.0),
        _row("sample_b", "ENSG4", "GENE4", 0.0, 500.0, 0.0, 0.0),
    ]
    return spark.createDataFrame(rows, NORM_SCHEMA)


@pytest.mark.spark
def test_sample_features_count_zero_and_detected_genes(normalized):
    features = {r["sample_uuid"]: r for r in build_sample_features(normalized).collect()}

    assert features["sample_a"]["num_zero_genes"] == 1
    assert features["sample_a"]["num_detected_genes"] == 3
    assert features["sample_b"]["num_zero_genes"] == 3
    assert features["sample_b"]["num_detected_genes"] == 1


@pytest.mark.spark
def test_sample_features_zero_percentage(normalized):
    features = {r["sample_uuid"]: r for r in build_sample_features(normalized).collect()}

    assert features["sample_a"]["pct_zero_genes"] == pytest.approx(0.25)
    assert features["sample_b"]["pct_zero_genes"] == pytest.approx(0.75)


@pytest.mark.spark
def test_sample_features_carry_library_size_through(normalized):
    features = {r["sample_uuid"]: r for r in build_sample_features(normalized).collect()}

    assert features["sample_a"]["library_size"] == 400.0
    assert features["sample_b"]["library_size"] == 500.0


@pytest.mark.spark
def test_sample_features_one_row_per_sample(normalized):
    assert build_sample_features(normalized).count() == 2


@pytest.mark.spark
def test_gene_features_detection_counts_across_samples(normalized):
    features = {r["gene_name"]: r for r in build_gene_features(normalized).collect()}

    # GENE1 is detected in both samples, GENE4 in neither.
    assert features["GENE1"]["num_samples_detected"] == 2
    assert features["GENE1"]["num_samples_total"] == 2
    assert features["GENE1"]["pct_samples_detected"] == pytest.approx(1.0)
    assert features["GENE4"]["num_samples_detected"] == 0
    assert features["GENE4"]["pct_samples_detected"] == pytest.approx(0.0)


@pytest.mark.spark
def test_gene_features_mean_expression(normalized):
    features = {r["gene_name"]: r for r in build_gene_features(normalized).collect()}

    expected = (math.log1p(250_000.0) + math.log1p(1_000_000.0)) / 2
    assert features["GENE1"]["mean_log1p_cpm"] == pytest.approx(expected)


@pytest.mark.spark
def test_gene_features_variability_is_zero_for_constant_expression(normalized):
    """GENE4 is zero in every sample, so its standard deviation is 0."""
    features = {r["gene_name"]: r for r in build_gene_features(normalized).collect()}
    assert features["GENE4"]["std_log1p_cpm"] == pytest.approx(0.0)


@pytest.mark.spark
def test_gene_features_std_is_null_for_single_sample_genes(spark):
    """Sample standard deviation is undefined for one observation."""
    rows = [_row("sample_a", "ENSG9", "GENE9", 5.0, 5.0, 1_000_000.0, math.log1p(1_000_000.0))]
    features = build_gene_features(spark.createDataFrame(rows, NORM_SCHEMA)).collect()
    assert features[0]["std_log1p_cpm"] is None
