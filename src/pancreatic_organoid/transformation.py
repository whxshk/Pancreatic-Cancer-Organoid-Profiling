"""Silver layer: reshape the wide Bronze matrix into a long, typed table.

The wide layout (one column per sample) is unusable for analysis and does not
scale: adding samples would add columns. Silver converts it to the long form

    gene_id | sample_uuid | count

using Spark's ``stack`` so the unpivot stays distributed rather than pulling
the matrix onto the driver.

Two data quirks are handled here:

* **Mixed column types.** Because the source JSON contains nulls, Spark infers
  some sample columns as ``string`` and others as ``long``. ``stack`` requires
  every stacked column to share one type, so all sample columns are cast to
  ``double`` first.
* **Non-gene rows.** The matrix includes aligner QC rows (``N_multimapping``,
  ``N_noFeature``, ...) alongside real genes. Only rows whose identifier starts
  with ``ENSG`` are kept.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

GENE_ID_COLUMN = "gene_id"
SAMPLE_COLUMN = "sample_uuid"
COUNT_COLUMN = "count"
ENSEMBL_GENE_PATTERN = "^ENSG"


def sample_columns(df: DataFrame, gene_id_column: str = GENE_ID_COLUMN) -> list[str]:
    """Returns every column of the wide matrix except the gene identifier."""
    return [c for c in df.columns if c != gene_id_column]


def cast_sample_columns(
    df: DataFrame, columns: list[str], gene_id_column: str = GENE_ID_COLUMN
) -> DataFrame:
    """Casts the given sample columns to ``double``, keeping the gene column.

    Without this, ``stack`` fails with ``DATATYPE_MISMATCH.STACK_COLUMN_DIFF_TYPES``
    because JSON type inference produces a mix of string and long columns.
    """
    return df.select([gene_id_column] + [F.col(c).cast("double").alias(c) for c in columns])


def build_stack_expression(
    columns: list[str],
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> str:
    """Builds the ``stack(...)`` SQL expression that unpivots the sample columns.

    For ``["s1", "s2"]`` this produces::

        stack(2, 's1', s1, 's2', s2) as (sample_uuid, count)

    Each column contributes two arguments: its name as a string literal (which
    becomes the ``sample_uuid`` value) and the column reference (which becomes
    the ``count`` value).

    Raises:
        ValueError: If ``columns`` is empty; ``stack(0, ...)`` is not valid SQL.
    """
    if not columns:
        raise ValueError("Cannot build a stack expression with no sample columns.")
    pairs = ", ".join(f"'{c}', {c}" for c in columns)
    return f"stack({len(columns)}, {pairs}) as ({sample_column}, {count_column})"


def unpivot_counts(
    df: DataFrame,
    columns: list[str],
    gene_id_column: str = GENE_ID_COLUMN,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Turns one row per gene into one row per (gene, sample) pair."""
    stack_expr = build_stack_expression(columns, sample_column, count_column)
    return df.select(gene_id_column, F.expr(stack_expr))


def filter_ensembl_genes(df: DataFrame, gene_id_column: str = GENE_ID_COLUMN) -> DataFrame:
    """Keeps only real gene rows, dropping aligner QC summary rows."""
    return df.filter(F.col(gene_id_column).rlike(ENSEMBL_GENE_PATTERN))


def fill_missing_counts(df: DataFrame, count_column: str = COUNT_COLUMN) -> DataFrame:
    """Replaces null counts with 0.0.

    A null in an RNA-seq count matrix means "no reads assigned", which is a
    zero, not an unknown value. Filling here keeps every gene/sample pair in
    the table so per-sample gene counts stay comparable.
    """
    return df.fillna({count_column: 0.0})


def build_silver_counts(
    df_bronze: DataFrame,
    gene_id_column: str = GENE_ID_COLUMN,
    sample_column: str = SAMPLE_COLUMN,
    count_column: str = COUNT_COLUMN,
) -> DataFrame:
    """Runs the full Bronze to Silver transformation.

    Cast to a uniform type, unpivot to long form, drop QC rows, fill nulls.
    """
    columns = sample_columns(df_bronze, gene_id_column)
    df_casted = cast_sample_columns(df_bronze, columns, gene_id_column)
    df_long = unpivot_counts(df_casted, columns, gene_id_column, sample_column, count_column)
    return fill_missing_counts(filter_ensembl_genes(df_long, gene_id_column), count_column)
