"""End-to-end orchestration of the Bronze / Silver / Gold pipeline.

Each stage reads its input from storage and writes its output back, so stages
can be run individually and re-run independently. That mirrors how the pipeline
was actually developed and operated: the Bronze ingest is slow and rarely
changes, while the Gold and modeling stages were iterated on repeatedly.

Running any stage requires access to the ADLS Gen2 account holding the source
data. See "Infrastructure requirements" in the README.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pyspark.sql import SparkSession

from . import annotation, features, ingestion, modeling, normalization, transformation
from .config import AdlsConfig, configure_spark_access
from .storage import read_delta, write_delta

logger = logging.getLogger(__name__)


def run_bronze(spark: SparkSession, config: AdlsConfig) -> None:
    """Reads the raw JSON matrix and writes the Bronze Delta table."""
    logger.info("Bronze: reading raw counts from %s", config.raw_path)
    df_bronze = ingestion.ingest_bronze(spark, config.raw_path)
    # overwrite_schema is required: JSON type inference is not stable between
    # runs, so the column types can differ from the existing table's.
    write_delta(df_bronze, config.bronze_counts_path, overwrite_schema=True)
    logger.info("Bronze: wrote %s", config.bronze_counts_path)


def run_silver(spark: SparkSession, config: AdlsConfig) -> None:
    """Unpivots Bronze into the long Silver counts table."""
    logger.info("Silver: reading Bronze from %s", config.bronze_counts_path)
    df_bronze = read_delta(spark, config.bronze_counts_path)
    df_silver = transformation.build_silver_counts(df_bronze)
    write_delta(df_silver, config.silver_counts_path, overwrite_schema=True)
    logger.info("Silver: wrote %s", config.silver_counts_path)


def run_annotation(spark: SparkSession, config: AdlsConfig) -> None:
    """Parses the Ensembl GTF into the gene annotation Delta table."""
    logger.info("Annotation: reading GTF from %s", config.gtf_path)
    df_genes = annotation.build_gene_annotations(spark, config.gtf_path)
    write_delta(df_genes, config.gene_annotations_path)
    logger.info("Annotation: wrote %s", config.gene_annotations_path)


def run_gold(spark: SparkSession, config: AdlsConfig) -> None:
    """Joins annotations, normalizes, and writes the four Gold tables."""
    df_silver = read_delta(spark, config.silver_counts_path)
    df_annotations = read_delta(spark, config.gene_annotations_path)

    df_annotated = annotation.join_gene_annotations(df_silver, df_annotations)
    write_delta(df_annotated, config.gold_counts_with_genes_path)
    logger.info("Gold: wrote %s", config.gold_counts_with_genes_path)

    df_norm = normalization.normalize_counts(df_annotated)
    write_delta(df_norm, config.gold_normalized_counts_path)
    write_delta(features.build_sample_features(df_norm), config.gold_sample_features_path)
    write_delta(features.build_gene_features(df_norm), config.gold_gene_features_path)
    logger.info("Gold: wrote normalized counts, sample features, gene features")


def run_modeling_stage(spark: SparkSession, config: AdlsConfig) -> None:
    """Runs PCA, K-Means, and the Random Forest, writing sample_predictions."""
    df_norm = read_delta(spark, config.gold_normalized_counts_path)
    result_pdf, metrics = modeling.run_modeling(df_norm)

    logger.info(
        "Model metrics: R2=%.4f MAE=%.2f MSE=%.2f RMSE=%.2f MAPE=%.2f%%",
        metrics["r2"],
        metrics["mae"],
        metrics["mse"],
        metrics["rmse"],
        metrics["mape"],
    )
    write_delta(spark.createDataFrame(result_pdf), config.gold_sample_predictions_path)
    logger.info("Modeling: wrote %s", config.gold_sample_predictions_path)


STAGES = {
    "bronze": run_bronze,
    "silver": run_silver,
    "annotation": run_annotation,
    "gold": run_gold,
    "modeling": run_modeling_stage,
}

#: Order used by ``--stage all``. Annotation precedes gold because the gold
#: join depends on the annotation table.
STAGE_ORDER = ("bronze", "silver", "annotation", "gold", "modeling")


def get_spark_session(app_name: str = "pancreatic-organoid") -> SparkSession:
    """Returns the active session, or builds one outside Databricks."""
    return SparkSession.builder.appName(app_name).getOrCreate()


def run_pipeline(spark: SparkSession, config: AdlsConfig, stage: str = "all") -> None:
    """Runs a single stage, or every stage in dependency order.

    Raises:
        ValueError: If ``stage`` is not a known stage name or ``"all"``.
    """
    if stage == "all":
        stages = STAGE_ORDER
    elif stage in STAGES:
        stages = (stage,)
    else:
        raise ValueError(
            f"Unknown stage {stage!r}. Expected 'all' or one of: {', '.join(STAGE_ORDER)}"
        )
    for name in stages:
        logger.info("--- stage: %s ---", name)
        STAGES[name](spark, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pancreatic cancer organoid RNA-seq pipeline. Requires "
            "access to the configured ADLS Gen2 account."
        )
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", *STAGE_ORDER),
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level (default: INFO)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    config = AdlsConfig.from_env()
    spark = get_spark_session()
    configure_spark_access(spark, config)
    run_pipeline(spark, config, args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
