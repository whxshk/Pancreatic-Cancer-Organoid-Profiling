"""Tests for GTF parsing and the version-stripped gene annotation join."""

from __future__ import annotations

import pytest

from pancreatic_organoid.annotation import (
    add_gene_base,
    extract_gene_annotations,
    join_gene_annotations,
)

# A real GTF attributes string from Homo_sapiens.GRCh38.109.
ATTRIBUTES = (
    'gene_id "ENSG00000223972"; gene_version "5"; gene_name "DDX11L1"; '
    'gene_source "havana"; gene_biotype "transcribed_unprocessed_pseudogene";'
)


@pytest.mark.spark
def test_extract_gene_annotations_parses_id_and_name(spark):
    df = spark.createDataFrame(
        [("1", "havana", "gene", "11869", "14409", ".", "+", ".", ATTRIBUTES)],
        list(
            (
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attributes",
            )
        ),
    )
    row = extract_gene_annotations(df).collect()[0]

    assert row["gene_id"] == "ENSG00000223972"
    assert row["gene_name"] == "DDX11L1"
    assert row["chrom"] == "1"
    assert row["strand"] == "+"


@pytest.mark.spark
def test_extract_gene_annotations_keeps_only_gene_features(spark):
    """One row per gene, not one per transcript or exon."""
    columns = (
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    )
    df = spark.createDataFrame(
        [
            ("1", "havana", "gene", "11869", "14409", ".", "+", ".", ATTRIBUTES),
            ("1", "havana", "transcript", "11869", "14409", ".", "+", ".", ATTRIBUTES),
            ("1", "havana", "exon", "11869", "12227", ".", "+", ".", ATTRIBUTES),
        ],
        list(columns),
    )
    assert extract_gene_annotations(df).count() == 1


@pytest.mark.spark
def test_add_gene_base_strips_the_version_suffix(spark):
    df = spark.createDataFrame(
        [("ENSG00000223972.5",), ("ENSG00000227232",)], ["gene_id"]
    )
    bases = [r["gene_base"] for r in add_gene_base(df).collect()]
    assert bases == ["ENSG00000223972", "ENSG00000227232"]


@pytest.mark.spark
def test_add_gene_base_yields_empty_string_for_non_ensembl_ids(spark):
    df = spark.createDataFrame([("N_multimapping",)], ["gene_id"])
    assert add_gene_base(df).collect()[0]["gene_base"] == ""


@pytest.mark.spark
def test_join_matches_versioned_counts_to_unversioned_annotations(spark):
    """Regression test for the original join bug.

    Counts carry ``ENSG00000223972.5`` while the GTF carries
    ``ENSG00000223972``. Joining on the raw identifier matches nothing; joining
    on the base accession is what makes gene symbols appear.
    """
    counts = spark.createDataFrame(
        [("ENSG00000223972.5", "sample_a", 10.0)], ["gene_id", "sample_uuid", "count"]
    )
    annotations = spark.createDataFrame(
        [("ENSG00000223972", "DDX11L1")], ["gene_id", "gene_name"]
    )
    row = join_gene_annotations(counts, annotations).collect()[0]

    assert row["gene_name"] == "DDX11L1"
    assert row["gene_id"] == "ENSG00000223972.5"  # original identifier preserved
    assert "gene_base" not in row.asDict()  # helper column dropped


@pytest.mark.spark
def test_join_keeps_unannotated_genes_with_null_name(spark):
    """The left join must not silently drop counts for unmatched genes."""
    counts = spark.createDataFrame(
        [("ENSG00000223972.5", "sample_a", 10.0), ("ENSG99999999999.1", "sample_a", 3.0)],
        ["gene_id", "sample_uuid", "count"],
    )
    annotations = spark.createDataFrame(
        [("ENSG00000223972", "DDX11L1")], ["gene_id", "gene_name"]
    )
    result = {r["gene_id"]: r["gene_name"] for r in join_gene_annotations(counts, annotations).collect()}

    assert result == {"ENSG00000223972.5": "DDX11L1", "ENSG99999999999.1": None}
