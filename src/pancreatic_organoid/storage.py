"""Thin Delta Lake read/write helpers.

The notebooks repeated ``.write.mode("overwrite").format("delta").save(path)``
in five separate cells, twice with ``overwriteSchema`` and three times without.
These two functions replace that repetition.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

DELTA_FORMAT = "delta"


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    """Loads a Delta table from ``path``."""
    return spark.read.format(DELTA_FORMAT).load(path)


def write_delta(df: DataFrame, path: str, overwrite_schema: bool = False) -> None:
    """Overwrites the Delta table at ``path`` with ``df``.

    Args:
        df: DataFrame to persist.
        path: Destination, typically an ``abfss://`` URI.
        overwrite_schema: Set when the new schema is incompatible with an
            existing table. The Bronze write needs this because the raw JSON
            infers a mix of string and long columns between runs; the notebook
            only discovered this after a failed write.
    """
    writer = df.write.mode("overwrite").format(DELTA_FORMAT)
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    writer.save(path)
