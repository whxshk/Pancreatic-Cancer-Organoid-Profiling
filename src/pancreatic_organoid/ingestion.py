"""Bronze layer: read the raw counts matrix and store it with minimal changes.

The source file is a single multiline JSON document holding a wide RNA-seq
count matrix: one column of gene identifiers plus one column per sequenced
sample (110 in the analysed dataset). Two properties of that file drive the
whole layer:

1. The gene identifier column is exported as ``Unnamed: 0``.
2. Sample columns are named with raw UUIDs containing hyphens, which are not
   valid characters in Delta/Parquet column names.

Bronze fixes only those two naming problems. Values, row counts, and inferred
types are left untouched so the layer stays auditable against the source file.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

GENE_ID_COLUMN = "gene_id"
_RAW_GENE_COLUMN_PREFIX = "Unnamed"
_ILLEGAL_COLUMN_CHARS = ("-", ".", ":")


def read_raw_counts(spark: SparkSession, raw_path: str) -> DataFrame:
    """Reads the wide counts matrix from a multiline JSON file."""
    return spark.read.option("multiline", "true").json(raw_path)


def rename_gene_column(df: DataFrame, gene_id_column: str = GENE_ID_COLUMN) -> DataFrame:
    """Renames the exported ``Unnamed: 0`` column to ``gene_id``.

    Raises:
        ValueError: If no column starting with ``Unnamed`` is present, which
            means the source export format changed and the rest of the
            pipeline's assumptions no longer hold.
    """
    candidates = [c for c in df.columns if c.startswith(_RAW_GENE_COLUMN_PREFIX)]
    if not candidates:
        raise ValueError(
            f"No column starting with {_RAW_GENE_COLUMN_PREFIX!r} found in the raw "
            f"matrix; cannot identify the gene identifier column. Columns seen: "
            f"{df.columns[:5]}..."
        )
    return df.withColumnRenamed(candidates[0], gene_id_column)


def sanitize_column_name(name: str) -> str:
    """Replaces characters Delta rejects in column names with underscores.

    Sample UUIDs arrive as ``00d56586-7c04-4ce3-a3ca-fe144cb65f01`` and become
    ``00d56586_7c04_4ce3_a3ca_fe144cb65f01``. The hyphen is the character that
    actually breaks the Delta write; ``.`` and ``:`` are handled defensively,
    as in the original notebook.
    """
    for char in _ILLEGAL_COLUMN_CHARS:
        name = name.replace(char, "_")
    return name


def sanitize_column_names(columns: list[str]) -> list[str]:
    """Applies :func:`sanitize_column_name` across a list of column names."""
    return [sanitize_column_name(c) for c in columns]


def build_bronze_counts(df_raw: DataFrame, gene_id_column: str = GENE_ID_COLUMN) -> DataFrame:
    """Applies the full Bronze transformation to the raw matrix.

    Renames the gene column first, then sanitizes every column name. Order
    matters: doing it the other way round would turn ``Unnamed: 0`` into
    ``Unnamed__0`` and the gene column would no longer be identifiable.
    """
    df = rename_gene_column(df_raw, gene_id_column)
    return df.toDF(*sanitize_column_names(df.columns))


def ingest_bronze(
    spark: SparkSession, raw_path: str, gene_id_column: str = GENE_ID_COLUMN
) -> DataFrame:
    """Reads the raw matrix and returns the Bronze-shaped DataFrame."""
    return build_bronze_counts(read_raw_counts(spark, raw_path), gene_id_column)
