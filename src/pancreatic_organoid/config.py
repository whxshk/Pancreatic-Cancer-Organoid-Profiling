"""Storage locations and Spark access configuration for the pipeline.

The original notebooks hard-coded an Azure Data Lake Storage (ADLS) Gen2
account name and account key directly in the first cell. That credential has
been removed from this repository; it is now read from the environment (or
passed in explicitly) and never stored in source control.

Nothing in this module contacts Azure by itself. `AdlsConfig` only builds the
`abfss://` paths that the rest of the pipeline reads and writes;
`configure_spark_access` is the single place where a credential is handed to
Spark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ACCOUNT_ENV_VAR = "ADLS_STORAGE_ACCOUNT"
CONTAINER_ENV_VAR = "ADLS_CONTAINER"
ACCOUNT_KEY_ENV_VAR = "ADLS_ACCOUNT_KEY"

DEFAULT_RAW_FILE = "raw/combined_counts_matrix.json"
DEFAULT_GTF_FILE = "reference/hg38/Homo_sapiens.GRCh38.109.chr.gtf.gz"


class MissingConfigurationError(RuntimeError):
    """Raised when required storage configuration is absent."""


@dataclass(frozen=True)
class AdlsConfig:
    """Resolves every path the pipeline reads from or writes to.

    Args:
        storage_account: ADLS Gen2 account name (without the domain suffix).
        container: Container holding the raw data and the medallion layers.
        raw_file: Container-relative path of the wide RNA-seq counts matrix.
        gtf_file: Container-relative path of the gzipped Ensembl GTF.
    """

    storage_account: str
    container: str
    raw_file: str = DEFAULT_RAW_FILE
    gtf_file: str = DEFAULT_GTF_FILE

    @classmethod
    def from_env(cls) -> "AdlsConfig":
        """Builds a config from ``ADLS_STORAGE_ACCOUNT`` and ``ADLS_CONTAINER``.

        Raises:
            MissingConfigurationError: If either variable is unset or empty.
        """
        account = os.environ.get(ACCOUNT_ENV_VAR, "").strip()
        container = os.environ.get(CONTAINER_ENV_VAR, "").strip()
        missing = [
            name
            for name, value in ((ACCOUNT_ENV_VAR, account), (CONTAINER_ENV_VAR, container))
            if not value
        ]
        if missing:
            raise MissingConfigurationError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". See the 'How to run' section of the README."
            )
        return cls(
            storage_account=account,
            container=container,
            raw_file=os.environ.get("ADLS_RAW_FILE", DEFAULT_RAW_FILE),
            gtf_file=os.environ.get("ADLS_GTF_FILE", DEFAULT_GTF_FILE),
        )

    @property
    def account_fqdn(self) -> str:
        return f"{self.storage_account}.dfs.core.windows.net"

    @property
    def base_path(self) -> str:
        return f"abfss://{self.container}@{self.account_fqdn}"

    @property
    def raw_path(self) -> str:
        return f"{self.base_path}/{self.raw_file}"

    @property
    def gtf_path(self) -> str:
        return f"{self.base_path}/{self.gtf_file}"

    @property
    def bronze_root(self) -> str:
        return f"{self.base_path}/bronze"

    @property
    def silver_root(self) -> str:
        return f"{self.base_path}/silver"

    @property
    def gold_root(self) -> str:
        return f"{self.base_path}/gold"

    # --- Individual table locations, named exactly as in the notebooks. ---

    @property
    def bronze_counts_path(self) -> str:
        return f"{self.bronze_root}/combined_counts_matrix_raw"

    @property
    def silver_counts_path(self) -> str:
        return f"{self.silver_root}/counts_long"

    @property
    def gene_annotations_path(self) -> str:
        return f"{self.base_path}/reference/hg38/gene_annotations_delta"

    @property
    def gold_counts_with_genes_path(self) -> str:
        return f"{self.gold_root}/counts_with_genes"

    @property
    def gold_normalized_counts_path(self) -> str:
        return f"{self.gold_root}/normalized_counts"

    @property
    def gold_sample_features_path(self) -> str:
        return f"{self.gold_root}/sample_features"

    @property
    def gold_gene_features_path(self) -> str:
        return f"{self.gold_root}/gene_features"

    @property
    def gold_sample_predictions_path(self) -> str:
        return f"{self.gold_root}/sample_predictions"


def configure_spark_access(spark, config: AdlsConfig, account_key: str | None = None) -> None:
    """Gives Spark an account key for the configured storage account.

    Args:
        spark: Active ``SparkSession``.
        config: Storage configuration identifying the account.
        account_key: Account key. When omitted, read from ``ADLS_ACCOUNT_KEY``.

    Raises:
        MissingConfigurationError: If no key is supplied and the environment
            variable is unset.

    Note:
        On Databricks, prefer a secret scope over an environment variable::

            configure_spark_access(
                spark, config, dbutils.secrets.get("my-scope", "adls-key")
            )
    """
    key = account_key if account_key is not None else os.environ.get(ACCOUNT_KEY_ENV_VAR, "")
    if not key.strip():
        raise MissingConfigurationError(
            f"No storage credential supplied. Set {ACCOUNT_KEY_ENV_VAR} or pass "
            "account_key explicitly. Credentials must never be committed to this "
            "repository."
        )
    spark.conf.set(f"fs.azure.account.key.{config.account_fqdn}", key)
