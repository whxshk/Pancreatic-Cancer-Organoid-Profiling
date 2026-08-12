"""Shared pytest fixtures.

Tests are split in two groups:

* Pure-Python tests (name sanitization, stack-expression building, regression
  metrics) need no JVM and always run.
* Spark tests need PySpark and a working Java installation. They are marked
  ``spark`` and skip with a clear message when either is missing, rather than
  failing.

Run only the Java-free tests with::

    pytest -m "not spark"
"""

from __future__ import annotations

import pytest

pytest.register_assert_rewrite("tests")


def _spark_unavailable_reason() -> str | None:
    """Returns why Spark cannot run here, or None if it can."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return "pyspark is not installed"

    import shutil
    import os

    java_home = os.environ.get("JAVA_HOME")
    if not shutil.which("java") and not java_home:
        return "no Java runtime found (JAVA_HOME unset and java not on PATH)"
    return None


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession for transformation tests."""
    reason = _spark_unavailable_reason()
    if reason:
        pytest.skip(f"Spark tests skipped: {reason}")

    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("pancreatic-organoid-tests")
        # Keep the tiny test frames in one partition; the default of 200
        # shuffle partitions makes every aggregation needlessly slow.
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
