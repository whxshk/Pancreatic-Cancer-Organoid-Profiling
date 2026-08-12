"""Tests for the Bronze naming fixes and the Silver wide-to-long unpivot."""

from __future__ import annotations

import pytest

from pancreatic_organoid.ingestion import (
    build_bronze_counts,
    rename_gene_column,
    sanitize_column_name,
    sanitize_column_names,
)
from pancreatic_organoid.transformation import (
    build_silver_counts,
    build_stack_expression,
    filter_ensembl_genes,
    fill_missing_counts,
    sample_columns,
    unpivot_counts,
)

# --- Pure Python: no Spark required -----------------------------------------


def test_sanitize_replaces_hyphens_in_sample_uuid():
    assert (
        sanitize_column_name("00d56586-7c04-4ce3-a3ca-fe144cb65f01")
        == "00d56586_7c04_4ce3_a3ca_fe144cb65f01"
    )


def test_sanitize_replaces_dots_and_colons():
    assert sanitize_column_name("a.b:c") == "a_b_c"


def test_sanitize_leaves_valid_names_untouched():
    assert sanitize_column_name("gene_id") == "gene_id"


def test_sanitize_column_names_maps_every_entry():
    assert sanitize_column_names(["a-b", "c.d"]) == ["a_b", "c_d"]


def test_stack_expression_pairs_literal_and_column_per_sample():
    assert build_stack_expression(["s1", "s2"]) == (
        "stack(2, 's1', s1, 's2', s2) as (sample_uuid, count)"
    )


def test_stack_expression_counts_all_columns():
    expr = build_stack_expression([f"s{i}" for i in range(110)])
    assert expr.startswith("stack(110, ")


def test_stack_expression_rejects_empty_column_list():
    with pytest.raises(ValueError):
        build_stack_expression([])


# --- Spark-backed transformations -------------------------------------------

pytestmark_spark = pytest.mark.spark


@pytest.mark.spark
def test_rename_gene_column_targets_the_unnamed_export_column(spark):
    df = spark.createDataFrame([("ENSG1", 1.0)], ["Unnamed: 0", "sample_a"])
    assert rename_gene_column(df).columns == ["gene_id", "sample_a"]


@pytest.mark.spark
def test_rename_gene_column_raises_when_absent(spark):
    df = spark.createDataFrame([("ENSG1", 1.0)], ["gene", "sample_a"])
    with pytest.raises(ValueError, match="Unnamed"):
        rename_gene_column(df)


@pytest.mark.spark
def test_build_bronze_renames_gene_column_before_sanitizing(spark):
    df = spark.createDataFrame([("ENSG1", 1.0)], ["Unnamed: 0", "aa-bb"])
    # If sanitizing ran first, the gene column would become "Unnamed__0".
    assert build_bronze_counts(df).columns == ["gene_id", "aa_bb"]


@pytest.mark.spark
def test_sample_columns_excludes_the_gene_column(spark):
    df = spark.createDataFrame([("ENSG1", 1.0, 2.0)], ["gene_id", "s1", "s2"])
    assert sample_columns(df) == ["s1", "s2"]


@pytest.mark.spark
def test_unpivot_produces_one_row_per_gene_sample_pair(spark):
    df = spark.createDataFrame(
        [("ENSG1", 10.0, 20.0), ("ENSG2", 30.0, 40.0)], ["gene_id", "s1", "s2"]
    )
    rows = unpivot_counts(df, ["s1", "s2"]).collect()

    assert len(rows) == 4
    assert {(r["gene_id"], r["sample_uuid"], r["count"]) for r in rows} == {
        ("ENSG1", "s1", 10.0),
        ("ENSG1", "s2", 20.0),
        ("ENSG2", "s1", 30.0),
        ("ENSG2", "s2", 40.0),
    }


@pytest.mark.spark
def test_filter_ensembl_genes_drops_aligner_qc_rows(spark):
    df = spark.createDataFrame(
        [("ENSG00000223972.5", 1.0), ("N_multimapping", 2772225.0), ("N_noFeature", 5.0)],
        ["gene_id", "count"],
    )
    kept = [r["gene_id"] for r in filter_ensembl_genes(df).collect()]
    assert kept == ["ENSG00000223972.5"]


@pytest.mark.spark
def test_fill_missing_counts_converts_null_to_zero(spark):
    df = spark.createDataFrame([("ENSG1", None), ("ENSG2", 7.0)], "gene_id string, count double")
    counts = {r["gene_id"]: r["count"] for r in fill_missing_counts(df).collect()}
    assert counts == {"ENSG1": 0.0, "ENSG2": 7.0}


@pytest.mark.spark
def test_build_silver_counts_casts_mixed_types_and_fills_nulls(spark):
    """The end-to-end Silver step on a matrix with the source file's quirks.

    ``s2`` is a string column and contains a null, which is exactly the
    combination that made the notebook's first unpivot attempt fail.
    """
    df = spark.createDataFrame(
        [("ENSG1", 10.0, "20"), ("ENSG2", 30.0, None), ("N_multimapping", 99.0, "99")],
        "gene_id string, s1 double, s2 string",
    )
    rows = build_silver_counts(df).collect()

    assert len(rows) == 4  # 2 genes x 2 samples; the QC row is dropped
    assert {(r["gene_id"], r["sample_uuid"], r["count"]) for r in rows} == {
        ("ENSG1", "s1", 10.0),
        ("ENSG1", "s2", 20.0),
        ("ENSG2", "s1", 30.0),
        ("ENSG2", "s2", 0.0),
    }
