> **Why this project is my MLH Fellowship code sample:** my more recent and technically significant work is under confidentiality and cannot be shared publicly, so I am submitting this earlier project instead — the original work is authentic, substantial, and entirely my own, and the whole repository can be inspected in full.

## Why this repository is being used as my MLH code sample

This repository contains a real RNA-seq data engineering and machine learning project I built and ran on Databricks against Azure Data Lake Storage. It was developed as academic project work and **predates this application** — it was not created for MLH.

The repository has been reorganized since that original work: the pipeline logic that lived in Jupyter/Databricks notebooks has been extracted into a tested Python package (a refactoring done with AI-assisted tooling — see [What I personally contributed](#what-i-personally-contributed)), and the credentials that were originally hard-coded in the notebooks have been removed. **The purpose of that cleanup is transparency and maintainability, not to misrepresent how the project was originally developed.** The original notebooks are preserved unchanged in [`notebooks/`](notebooks/), including the failed attempts and the fixes that followed them, so the actual development history remains inspectable. Every deviation from the notebook behaviour is listed under [Deviations from the original notebooks](#deviations-from-the-original-notebooks).

My more recent projects cannot be shown here:

- **SharkBand** — a startup I have been building since September 2023 as founder. External validation includes 1st place at the Qatar Development Bank Business Gateway Competition, 3rd place at the QSTP Ride & Pitch Competition, 3rd place at the UHUB × DMZ Summer Sprint, incubation through the DMZ Basecamp Program in Toronto, 7+ letters of interest from businesses, interest from 1,000+ students, adoption across 5+ UDST College of Business courses with 50+ students using the product, the first startup collaboration with the UDST College of Business through its 2026 Practicum & Capstone, participation in Web Summit 2026, and the Best Oral Presentation Award at the 2026 UDST Student Symposium.
- **Equipped** — an AI fashion technology project incubated by Scale7 and M7.

Both are closed-source commercial products. Their source code and implementation details cannot be published, which is why this project serves as the public code sample.

---

# Pancreatic Cancer Organoid Profiling

A PySpark pipeline that turns raw RNA-seq count matrices from patient-derived pancreatic cancer organoids into annotated, normalized, analysis-ready datasets, using a Bronze/Silver/Gold (medallion) architecture on Delta Lake, followed by dimensionality reduction, clustering, and regression on sample-level features.

## The problem

Pancreatic cancer has a five-year survival rate below 12%, and chemotherapy response varies widely between patients with no reliable predictive markers. Patient-derived organoids — lab-grown mini-tumours that retain the genetics and drug response of the original tumour — are one of the most promising systems for studying that variation.

Profiling those organoids by RNA sequencing produces data that is not directly analysable. The output of this project is the transformation layer that sits between the raw sequencer output and any downstream biological analysis: a set of clean, annotated, normalized tables that a researcher or a model can actually use.

## Why this data needs a pipeline

The raw input is a single wide JSON matrix with one row per gene and one column per sample. Four properties of that file drive nearly every design decision here:

**It is the wrong shape.** One column per sample means adding samples changes the schema. Analysis, joins, and aggregation all need the long form `(gene_id, sample_uuid, count)`, which requires unpivoting roughly 62,000 gene rows across 110 sample columns into millions of rows.

**It is not consistently typed.** The matrix contains nulls, so JSON type inference produces a mix of `string` and `long` columns for what should be uniform numeric data. Spark's `stack` requires one shared type across all stacked columns, so everything must be cast before the unpivot. Column names are raw UUIDs containing hyphens, which Delta rejects.

**It contains non-gene rows.** Alignment QC summary rows (`N_multimapping`, `N_noFeature`, ...) sit in the same matrix as real genes and would silently corrupt any per-sample total.

**Its identifiers are unreadable and version-stamped.** Genes are identified as `ENSG00000223972.5`. Attaching human-readable symbols means parsing an Ensembl GTF and joining on the version-stripped accession — joining on the raw identifier matches nothing.

Sequencing depth also varies between samples, so raw counts are not comparable across organoids without normalization.

## Architecture

```
raw/combined_counts_matrix.json          reference/hg38/*.gtf.gz
            │                                       │
            ▼                                       ▼
┌─────────────────────────┐             ┌───────────────────────┐
│ BRONZE                  │             │ gene_annotations_delta│
│ combined_counts_        │             │ gene_id, gene_name,   │
│ matrix_raw              │             │ chrom, start, end,    │
│ wide, names fixed only  │             │ strand                │
└───────────┬─────────────┘             └───────────┬───────────┘
            ▼                                       │
┌─────────────────────────┐                         │
│ SILVER                  │                         │
│ counts_long             │                         │
│ gene_id, sample_uuid,   │                         │
│ count — typed, no QC    │                         │
│ rows, nulls filled      │                         │
└───────────┬─────────────┘                         │
            └──────────────────┬──────────────────  ┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │ GOLD                                         │
        │  counts_with_genes    annotated long counts  │
        │  normalized_counts    + library_size, cpm,   │
        │                         log1p_cpm            │
        │  sample_features      per-sample QC summary  │
        │  gene_features        per-gene summary       │
        │  sample_predictions   PCA, clusters, model   │
        └──────────────────────────────────────────────┘
                               ▼
                    Power BI  /  matplotlib
```

Each layer is persisted to Delta and each stage reads its input back from storage, so stages are independently re-runnable. That mattered in practice: the Bronze ingest is slow and almost never changes, while the Gold and modeling stages were iterated on repeatedly.

### Bronze — raw, preserved

Reads the multiline JSON matrix and writes it to Delta with **only** naming changes: `Unnamed: 0` becomes `gene_id`, and hyphens, dots, and colons in sample UUID column names become underscores. No values, rows, or types are altered, so the layer remains auditable against the source file.

The write uses `overwriteSchema` because JSON type inference is not stable between runs; the notebook discovered this the hard way when a re-run failed against the existing table.

*Module:* [`ingestion.py`](src/pancreatic_organoid/ingestion.py) · *Output:* `bronze/combined_counts_matrix_raw`

### Silver — cleaned and reshaped

Casts every sample column to `double`, unpivots wide to long with `stack`, drops non-`ENSG` QC rows, and fills null counts with `0.0` (a null in a count matrix means "no reads assigned", which is a zero, not an unknown).

The unpivot is Spark-native. The matrix is never collected to the driver.

*Module:* [`transformation.py`](src/pancreatic_organoid/transformation.py) · *Output:* `silver/counts_long`

### Gold — annotated, normalized, curated

Parses the Ensembl GTF (`feature == "gene"` rows only, so each gene contributes one row rather than one per transcript), extracts `gene_id` and `gene_name` from the packed attributes string by regex, and left-joins gene symbols onto the counts via the version-stripped accession. The join is a left outer join so counts for genes absent from the annotation release are retained with a null `gene_name` rather than silently dropped.

*Modules:* [`annotation.py`](src/pancreatic_organoid/annotation.py), [`normalization.py`](src/pancreatic_organoid/normalization.py), [`features.py`](src/pancreatic_organoid/features.py)
*Outputs:* `gold/counts_with_genes`, `gold/normalized_counts`, `gold/sample_features`, `gold/gene_features`, `gold/sample_predictions`

## Normalization

Raw counts are not comparable across samples: 500 reads in a 5-million-read library and 500 reads in a 1-million-read library represent very different expression levels. Counts Per Million divides out sequencing depth:

```
library_size = Σ count            (per sample)
cpm          = count × 1e6 / library_size
log1p_cpm    = ln(1 + cpm)
```

Expression is also heavily right-skewed — a few genes dominate the total while most sit near zero — so a log transform is applied before any distance-based method. `log1p` is used so zero counts map to zero rather than negative infinity.

Samples with a zero library size are removed before CPM is computed, via an inner join against the filtered totals, because dividing by zero would produce null or infinite expression for every gene in the sample.

> **Convention note:** this is `ln(1 + CPM)`, not the `log2(CPM + 1)` used by edgeR and limma. The two differ by a constant factor of `ln 2 ≈ 0.693`. Since the transform is monotonic, PCA and clustering structure is unaffected, but the values are not directly comparable to a standard bioinformatics logCPM.

## Feature engineering

Two summary tables are derived from the normalized counts.

**`gene_features`** — one row per gene: `mean_log1p_cpm`, `std_log1p_cpm` (expression variability), `num_samples_detected`, `num_samples_total`, `pct_samples_detected`. Genes detected in very few samples are normally excluded from downstream analysis.

**`sample_features`** — one row per organoid: `library_size`, `avg_log1p_cpm`, `num_zero_genes`, `num_detected_genes`, `pct_zero_genes`. RNA-seq matrices are heavily zero-inflated, so the proportion of zero-count genes is a useful quality signal — an unusually high value typically indicates shallow sequencing or degraded input material.

Counting rows conditionally inside an aggregation requires `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` rather than `SUM(condition)`; Spark rejects boolean input to `SUM`.

> **Note:** the modeling step in [`modeling.py`](src/pancreatic_organoid/modeling.py) builds its **own** sample-level matrix, which averages raw `count` rather than `log1p_cpm`. The two tables are similar but not interchangeable, and the function names reflect that.

## Machine learning

All models run on the sample-level matrix — one row per organoid, ~110 rows — so the Spark-to-pandas conversion happens only after the heavy aggregation has been done in Spark.

**Unsupervised.** Features are standardized (they span millions for `library_size` down to a bounded fraction for `pct_zero_genes`, and both methods are distance-based), then projected with PCA to 2 components and clustered with K-Means (k=3). K-Means is fitted on the standardized features, not the PCA components.

**Supervised.** A Random Forest regressor (200 trees, 80/20 split, `random_state=42`) predicts `library_size` from `avg_count`, `num_zero_genes`, `num_detected_genes`, and `pct_zero_genes`. Random Forest was chosen because it handles non-linear relationships, needs no feature scaling, and is robust to outliers.

> ### ⚠️ The supervised target is derivable from its own features
>
> This is the most important thing to understand about the modeling, and it is stated here rather than buried.
>
> The Silver unpivot gives every sample a row for every gene, so the number of rows per sample, `N`, is identical across samples. That makes `avg_count = library_size / N`, and therefore:
>
> ```
> library_size ≡ avg_count × N
> ```
>
> The model is predicting a rescaled copy of one of its own inputs. **The reported R² of 0.9719 measures that algebraic identity, not genuine predictive capability.** The PCA output confirms it independently: explained variance across two components is `[0.842, 0.158]`, summing to exactly 1.0, which can only happen if the four features span a two-dimensional space — precisely what the identity implies.
>
> The cause is a real constraint on the data, not an oversight in method: **the open-access tier of this dataset carries no drug-response or clinical-outcome labels**, so the only available regression targets were quality-control metrics derived from the counts themselves. The code is preserved exactly as originally written and the limitation is documented rather than papered over. See [Limitations](#limitations) for what a genuine version of this task would require.

*Module:* [`modeling.py`](src/pancreatic_organoid/modeling.py) · *Output:* `gold/sample_predictions`

## Technologies

| Area | Technology |
|---|---|
| Distributed processing | PySpark (DataFrame API) |
| Compute platform | Databricks |
| Storage | Azure Data Lake Storage Gen2 (`abfss://`) |
| Table format | Delta Lake |
| Machine learning | scikit-learn, pandas, NumPy |
| Visualization | Power BI Desktop, matplotlib |
| Testing | pytest |
| Language | Python |

**Why Spark:** the unpivot expands ~62,000 gene rows × 110 samples into millions of rows, and the normalization requires a per-sample aggregate joined back onto every row. Both are shuffle-heavy operations on data too large to hold comfortably in memory on a single machine.

**Why Delta over plain Parquet:** the pipeline is re-run repeatedly during development. Delta's atomic overwrites mean a failed run cannot leave a half-written table behind, and `overwriteSchema` handles the unstable JSON type inference explicitly rather than by deleting and recreating directories.

## Project structure

```
├── src/pancreatic_organoid/
│   ├── config.py           storage paths + credential wiring (no secrets in source)
│   ├── storage.py          Delta read/write helpers
│   ├── ingestion.py        Bronze: raw JSON → Delta
│   ├── transformation.py   Silver: cast, unpivot, filter QC rows, fill nulls
│   ├── annotation.py       GTF parsing + version-stripped gene symbol join
│   ├── normalization.py    library size, CPM, log1p(CPM)
│   ├── features.py         per-sample and per-gene Gold features
│   ├── modeling.py         PCA, K-Means, Random Forest, metrics
│   └── pipeline.py         stage orchestration + CLI
├── tests/                  43 tests (see Testing)
├── notebooks/              original Databricks notebooks, preserved
├── docs/                   original project documentation (Phase I, Phase II)
├── reports/                Power BI report
├── pyproject.toml
└── requirements.txt
```

## What runs where

| Component | Status |
|---|---|
| All transformation, annotation, normalization, feature, and modeling logic | **Implemented in this repository** and unit-tested on synthetic fixtures |
| Reading the raw matrix / GTF, writing any Delta table, running the pipeline end to end | **Requires Databricks + Azure.** Needs the ADLS account, credentials, and the source data |
| Schema exploration, raw-file previews, min/max range checks, storage connectivity checks | **Exploratory**, kept in `notebooks/` only |
| PCA scatter, predicted-vs-actual, and residual plots | **Exploratory**, in `notebooks/Modeling.ipynb` |
| Power BI report | **Static artifact** in `reports/`, not produced by this code |

## How to run

### Test suite (no cloud access needed)

```bash
pip install -r requirements.txt
```

Run everything, including tests that spin up a local `SparkSession`:

```bash
pytest
```

Spark tests need a Java runtime (8, 11, or 17). Without one, run only the Java-free tests:

```bash
pytest -m "not spark"
```

### Full pipeline (needs Databricks + Azure)

Configuration comes entirely from the environment; no credential is stored in this repository.

```bash
export ADLS_STORAGE_ACCOUNT=your-account
export ADLS_CONTAINER=your-container
export ADLS_ACCOUNT_KEY=your-key
```

```bash
python -m pancreatic_organoid.pipeline --stage all
```

Individual stages, in dependency order: `bronze`, `silver`, `annotation`, `gold`, `modeling`.

```bash
python -m pancreatic_organoid.pipeline --stage silver
```

On Databricks, prefer a secret scope over an environment variable:

```python
from pancreatic_organoid.config import AdlsConfig, configure_spark_access

config = AdlsConfig(storage_account="...", container="...")
configure_spark_access(spark, config, dbutils.secrets.get("my-scope", "adls-key"))
```

## Infrastructure requirements

Running the pipeline end to end requires all of:

- A Spark cluster with Delta Lake — the original runs used Databricks
- An ADLS Gen2 account with credentials, holding the raw matrix and the Ensembl GTF
- `raw/combined_counts_matrix.json` — the wide RNA-seq counts matrix
- `reference/hg38/Homo_sapiens.GRCh38.109.chr.gtf.gz` — Ensembl GRCh38 release 109
- Power BI Desktop, to open the report in `reports/`

**Data sources.** The RNA-seq data comes from dbGaP study [phs001611.v1.p1](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001611.v1.p1), whose open-access tier is mirrored at `s3://gdc-organoid-pancreatic-phs001611-2-open/` ([AWS Open Data Registry](https://registry.opendata.aws/organoid-pancreatic)). The annotation is from [Ensembl release 109](https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/). Neither the source data nor the derived Delta tables are redistributed in this repository.

## Testing

43 tests, all passing (verified against PySpark 3.5.1 on Java 8).

| File | Covers |
|---|---|
| `test_transformation.py` | Column-name sanitization, gene-column renaming, `stack` expression construction, unpivot correctness, QC-row filtering, null filling |
| `test_annotation.py` | GTF attribute parsing, gene-feature filtering, accession version stripping, the versioned/unversioned join, unmatched-gene retention |
| `test_normalization.py` | Library size totals, CPM scaling, the CPM-sums-to-1e6 property, `log1p` behaviour, zero-library exclusion |
| `test_features.py` | Zero/detected gene counts, percentage arithmetic, per-gene detection and variability, null standard deviation for single-sample genes |
| `test_modeling.py` | Regression metrics against hand-computed values, feature matrix construction, and an executable demonstration of the target/feature collinearity described above |

Tests split into two groups: 12 are pure Python and run anywhere; 31 use a local `SparkSession` and are marked `spark` so they skip with a clear message when Java or PySpark is unavailable, rather than failing.

**Not covered by tests:** anything requiring ADLS credentials, Delta writes, or the real dataset. Those paths cannot be exercised without the infrastructure, and no mock stands in for them. There are no tests asserting scikit-learn's own behaviour.

## Example outputs and results

All figures below are actual recorded outputs from the original Databricks runs, preserved in `notebooks/`.

**Scale.** 111 columns in the raw matrix (1 gene column + 110 samples). Raw counts range from `0.0` to `896,464.0`, consistent with untransformed RNA-seq.

**PCA.** Explained variance ratio `[0.84217929, 0.15782071]` across two components.

**K-Means (k=3) cluster profiles.**

| Cluster | Mean library size | Std | Mean % zero genes | Std |
|---|---|---|---|---|
| 0 | 3,503,231 | 850,021 | 0.6099 | 0.0073 |
| 1 | 2,539,674 | 367,562 | 0.6623 | 0.0110 |
| 2 | 3,267,319 | 501,662 | 0.6358 | 0.0082 |

Clusters separate primarily by sequencing depth, with the shallower cluster showing the higher proportion of undetected genes — the expected relationship, and a reasonable sanity check on the feature engineering.

**Random Forest regression.**

```
R² Score:   0.9719
MAE:        87,294.95
MSE:        12,837,985,974.22
RMSE:       113,304.84
MAPE:       3.31%
```

These metrics are computed on the 20% held-out split. **They should be read in light of the target/feature relationship described above** — they reflect an algebraic identity in the feature set rather than predictive power.

**Power BI report** (`reports/organoid_profiling_dashboard.pbix`): a single page with 7 visuals over the annotated counts — two clustered bar charts (total count by gene name; average count by gene ID), a pivot table of counts by gene, a treemap of counts by gene and sample, and two card visuals. The PCA, prediction-quality, and residual plots are matplotlib figures in `notebooks/Modeling.ipynb`, not Power BI visuals.

## Limitations

**No biological outcome labels.** The open-access tier carries no drug-response or clinical-outcome data, so the supervised task is a QC-metric regression rather than the chemotherapy-response prediction the underlying research question calls for. Doing that properly would require dbGaP-controlled access to the clinical metadata.

**The regression target is not independent of its features.** Documented in full under [Machine learning](#machine-learning). The R² should not be read as evidence of a predictive feature set.

**Small sample count for supervised learning.** 110 samples with an 80/20 split leaves 22 test samples. Metrics from a test set that size carry wide confidence intervals, and no cross-validation or repeated splitting was performed.

**Saved predictions mix in-sample and held-out values.** The `pred_library_size` column in `gold/sample_predictions` is generated for every sample, including the ~80% the model was trained on. The reported metrics use only the test split, but that column should not be treated as held-out performance.

**Genomic coordinates are not carried into the counts.** The annotation join attaches `gene_name` only. Chromosome, start, end, and strand are parsed and stored in `gene_annotations_delta` but are not present in `counts_with_genes`.

**Possible row multiplication in the annotation join.** If the Ensembl release maps more than one gene feature to the same base accession, the join would duplicate count rows. This was not verified against the real GTF, and the infrastructure is no longer accessible to check. It is called out here rather than assumed away.

**The matrix assembly step is not in this repository.** The pipeline starts from an already-combined `combined_counts_matrix.json`. The step that merged per-sample GDC count files into that wide matrix was performed separately and is not part of this codebase.

**`log1p_cpm` is a natural log**, not the base-2 logCPM standard in transcriptomics. See the note under [Normalization](#normalization).

**No cluster-count selection.** k=3 was chosen directly; no elbow or silhouette analysis was performed to justify it.

## Deviations from the original notebooks

Everything that differs from the notebook behaviour, in full:

1. **Credentials removed.** The notebooks hard-coded an Azure storage account key. It is now read from the environment or passed explicitly, and never stored in source. In the preserved notebooks, the key, the storage account name, and the Databricks workspace host are replaced with `REDACTED_*` placeholders in both code and saved outputs. That credential is not in use and has not been reused anywhere.
2. **Null filling moved before the Silver write.** The notebook wrote `silver/counts_long` and *then* applied `fillna(0)` in memory, so the persisted Silver table still contained nulls; the fill was re-applied downstream when the table was reloaded. Final results were unaffected, but the stored table did not match its documented contract. The fill now happens before the write.
3. **Failed attempts not carried into the package.** Where a component had several attempts, only the working version was extracted. The failed attempts remain visible in `notebooks/`.
4. **Notebook header cells added.** One markdown cell at the top of each preserved notebook explains its provenance and the redaction. No other cell was modified.

Everything else — every transformation, every aggregation, every hyperparameter, every random seed — is preserved as originally written.

## What I personally contributed

All of the original project is my own work: the pipeline design, the Databricks/PySpark implementation and its debugging (the failed attempts preserved in [`notebooks/`](notebooks/) are mine too), the feature engineering, the modeling, the Power BI report, and the Phase I/II documentation in [`docs/`](docs/).

The later reorganization — extracting that existing notebook logic into the tested package under `src/` and writing its test suite — was done with AI-assisted tooling, which I directed and reviewed. It preserves the original work rather than replacing it: every algorithm, transformation, and hyperparameter in the package traces back to a specific notebook cell, and the [Deviations](#deviations-from-the-original-notebooks) section lists everything that differs.

## Attribution

Pipeline, analysis, and modeling by Wahed Shaik. Source data from dbGaP study phs001611.v1.p1 via the AWS Open Data Registry; gene annotation from Ensembl release 109.
