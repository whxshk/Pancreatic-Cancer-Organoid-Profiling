"""Bronze/Silver/Gold RNA-seq pipeline for pancreatic cancer organoid profiling.

Modules:
    config: Storage paths and Spark credential wiring.
    storage: Delta read/write helpers.
    ingestion: Bronze — raw JSON counts matrix to Delta.
    transformation: Silver — wide to long unpivot, typing, QC-row removal.
    annotation: Ensembl GTF parsing and gene-symbol join.
    normalization: CPM and log1p(CPM).
    features: Gold — per-sample and per-gene summary features.
    modeling: PCA, K-Means, and Random Forest regression.
    pipeline: Stage orchestration and CLI entry point.
"""

__version__ = "1.0.0"

__all__ = [
    "annotation",
    "config",
    "features",
    "ingestion",
    "modeling",
    "normalization",
    "pipeline",
    "storage",
    "transformation",
]
