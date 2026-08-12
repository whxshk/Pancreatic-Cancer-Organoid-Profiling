"""Gene annotation: parse the Ensembl GTF and attach gene symbols to counts.

The counts matrix identifies genes only by Ensembl accession
(``ENSG00000223972.5``), which is not readable and cannot be grouped by
biological name. This module parses the Ensembl GRCh38 release 109 GTF into a
gene-level table and joins it onto the counts.

The join needs one non-obvious step. Counts carry *versioned* accessions
(``ENSG00000223972.5``) while the GTF ``gene_id`` attribute is unversioned
(``ENSG00000223972``), so joining on the raw identifier matches nothing. Both
sides are reduced to the version-free base accession first.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

GTF_COLUMNS = (
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

GENE_ID_COLUMN = "gene_id"
GENE_BASE_COLUMN = "gene_base"
GENE_BASE_PATTERN = r"^(ENSG[0-9]+)"
_GENE_ID_ATTRIBUTE_PATTERN = r'gene_id "([^"]+)"'
_GENE_NAME_ATTRIBUTE_PATTERN = r'gene_name "([^"]+)"'


def read_gtf(spark: SparkSession, gtf_path: str) -> DataFrame:
    """Reads a gzipped Ensembl GTF as a headerless, tab-separated file.

    GTF is a 9-column TSV whose header lines begin with ``#``. Spark
    decompresses ``.gz`` transparently, but the file is not splittable, so this
    read is single-threaded regardless of cluster size.
    """
    return (
        spark.read.option("header", "false")
        .option("sep", "\t")
        .option("comment", "#")
        .csv(gtf_path)
        .toDF(*GTF_COLUMNS)
    )


def extract_gene_annotations(df_gtf: DataFrame) -> DataFrame:
    """Selects gene features and pulls identifiers out of the attributes string.

    A GTF row's ninth column packs key/value pairs into one string::

        gene_id "ENSG00000223972"; gene_version "5"; gene_name "DDX11L1"; ...

    Only rows with ``feature == "gene"`` are kept, so each gene contributes one
    row rather than one row per transcript or exon.

    Returns:
        A DataFrame of ``gene_id``, ``gene_name``, ``chrom``, ``start``,
        ``end``, ``strand``.
    """
    return (
        df_gtf.filter(F.col("feature") == "gene")
        .withColumn(GENE_ID_COLUMN, F.regexp_extract("attributes", _GENE_ID_ATTRIBUTE_PATTERN, 1))
        .withColumn("gene_name", F.regexp_extract("attributes", _GENE_NAME_ATTRIBUTE_PATTERN, 1))
        .select(GENE_ID_COLUMN, "gene_name", "chrom", "start", "end", "strand")
    )


def add_gene_base(
    df: DataFrame,
    gene_id_column: str = GENE_ID_COLUMN,
    gene_base_column: str = GENE_BASE_COLUMN,
) -> DataFrame:
    """Adds the version-free Ensembl accession as a join key.

    ``ENSG00000223972.5`` becomes ``ENSG00000223972``. Identifiers that do not
    match the pattern yield an empty string, which is Spark's
    ``regexp_extract`` behaviour on no match.
    """
    return df.withColumn(
        gene_base_column, F.regexp_extract(gene_id_column, GENE_BASE_PATTERN, 1)
    )


def join_gene_annotations(
    df_counts: DataFrame,
    df_annotations: DataFrame,
    gene_id_column: str = GENE_ID_COLUMN,
    gene_base_column: str = GENE_BASE_COLUMN,
) -> DataFrame:
    """Attaches ``gene_name`` to the counts table via the base accession.

    The join is a left outer join: counts for genes absent from the annotation
    release are kept with a null ``gene_name`` rather than silently dropped.

    Note:
        Only ``gene_name`` is attached. Chromosome, start, end, and strand
        remain in the annotation table and are not carried into the counts.
    """
    counts = add_gene_base(df_counts, gene_id_column, gene_base_column)
    annotations = add_gene_base(df_annotations, gene_id_column, gene_base_column)
    return counts.join(
        annotations.select(gene_base_column, "gene_name"),
        on=gene_base_column,
        how="left",
    ).drop(gene_base_column)


def build_gene_annotations(spark: SparkSession, gtf_path: str) -> DataFrame:
    """Reads the GTF and returns the gene-level annotation table."""
    return extract_gene_annotations(read_gtf(spark, gtf_path))
