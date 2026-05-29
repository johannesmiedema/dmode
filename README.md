# DModE – Differential Modification and Expression Analysis

> A Python package and command-line interface for integrated RNA modification and expression analyses from long-read sequencing data.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data & Metadata Requirements](#data--metadata-requirements)
- [CLI Tutorial](#cli-tutorial)
- [Command Reference](#command-reference)
- [Python API](#python-api)
- [HTML Reports](#html-reports)
- [License](#license)

---

## Introduction

Long-read sequencing technologies (e.g., Oxford Nanopore) can simultaneously capture RNA sequence and base modification information, enabling researchers to study how epitranscriptomic marks such as m⁶A, m⁵C, Ψ (pseudouridine), and 2′-O-methylations change across biological conditions. Analysing this data in a rigorous and reproducible way, however, typically requires navigating multiple tools, file formats, and statistical frameworks.

**DModE** addresses this by providing a single, unified framework for:

- **Exploratory analysis** – quality-control visualisations (PCA, UMAP, violin/bar plots, metagene profiles) that give an immediate overview of modification landscapes across samples.
- **Differential modification calling** – site-level statistical testing comparing modification frequencies between two or more conditions, with multiple-testing correction, volcano plots, and annotated result tables.
- **Differential expression analysis** – DESeq2-style gene- and transcript-level expression testing, generating MA plots, heatmaps, and sample correlation matrices.
- **Integrated analysis** – joint workflows that link expression changes to modification dynamics, producing correlation scatter plots and ranked gene/transcript summaries.

DModE is designed for both interactive use in Jupyter notebooks (via its Python API) and automated pipelines (via its CLI). Every run can optionally generate a single, self-contained HTML report that bundles all figures and tables into a navigable interface.

---

## Key Features

- Gene- and transcript-level QC: PCA/UMAP, violin & bar plots, RMBase database intersections, metagene coverage profiles.
- Differential modification calling with SLIM or Benjamini–Hochberg multiple-testing correction, per-chromosome or genome-wide volcano plots, and significance tables annotated with gene/transcript IDs.
- Differential gene/transcript expression using `pydeseq2`, with MA plots, pairwise and global heatmaps, and sample–sample correlation matrices.
- Integrated expression + modification workflows (gene or transcript level) with scatter/correlation plots linking both data modalities.
- Flexible input: provide a tab-separated metadata sheet **or** explicit file lists on the command line — both modes are supported for every command.
- Filter to specific modification types (e.g. `--modification m6A Cm`) or analyse all types detected in the data automatically.
- Automatic `dmode_report.html` generation grouping all outputs by analysis, modification type, and condition comparison.
- Python API (`DmodE` class) for notebooks and custom scripted analyses.

---

## Installation

**Prerequisites:** Python 3.10 or later (developed on 3.12), a C compiler toolchain, and system packages required by `pydeseq2` / `statsmodels`.

```bash
# 1. Clone the repository
git clone https://github.com/stegiopast/dmode.git
cd dmode

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install DModE and all dependencies
pip install -e .
```

After installation, the `dmode` command becomes available in your environment. To upgrade, pull the latest changes and re-run `pip install -e .`.

---

## Data & Metadata Requirements

### Input files
| Workflow | Required files |
|---|---|
| Modification (gene or transcript level) | Modkit BED files, one per sample |
| Gene expression | featureCounts TSV/CSV matrices |
| Transcript expression | Salmon `quant.sf` files |

### Metadata sheet
A tab-separated file with at minimum the `File` and `Condition` columns. A `Sample` column is required for expression workflows.

```tsv
File                              Sample      Condition
/data/UHRR1_modkit.bed            UHRR1       UHRR
/data/UHRR2_modkit.bed            UHRR2       UHRR
/data/HEK293_1_modkit.bed          HEK293_1     HEK293
/data/HEK293_2_modkit.bed          HEK293_2     HEK293
```

All commands also accept explicit file lists (see `--data-files`) instead of a metadata sheet.

### Annotation
A GTF file matching the genome or transcriptome used for alignment is required by every workflow (`--gtf-file`). GENCODE or Ensembl GTFs work directly.

### Reference levels
One or more control/baseline condition labels must be designated as reference levels (`--reference_levels`). All other conditions are compared against them.

---

## CLI Tutorial

This tutorial walks through the most common DModE workflows using the example HEK293 vs. UHRR dataset. Each command can be run with a metadata file (`--metadata`) **or** with explicit file lists (`--data-files`, `--conditions`, `--samplenames`); both styles are shown where instructive.

### 1. Exploratory statistics at the gene level

Before running any differential analysis, it is good practice to inspect the data. The `gene_basic_statistics` command generates PCA and UMAP plots for sample clustering, bar/violin plots of modification site counts and frequencies, and — optionally — overlaps with the RMBase database of known modification sites.

```bash
dmode gene_basic_statistics \
  --metadata metadata_gene.tsv \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/gene_basic_stats \
  --coverage_filter 20 \
  --frequency_filter 10 \
  --modification m6A
```

Using explicit file lists instead of a metadata sheet:

```bash
dmode gene_basic_statistics \
  --data-files UHRR1_modkit.bed UHRR2_modkit.bed \
               HEK293_1_modkit.bed HEK293_2_modkit.bed \
  --conditions UHRR UHRR HEK293 HEK293 \
  --samplenames UHRR1 UHRR2 HEK293_1 HEK293_2 \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/gene_basic_stats \
  --modification m6A
```

Outputs saved under `results/gene_basic_stats/gene_basic_statistics/`:
- `*_PCA_plot.png` and `*_UMAP_plot.png` — sample clustering.
- `*_modsite_barplot.png` — number of detected modification sites per sample.
- `*_modification_distribution.png` and `*_violin_plot.png` — frequency distributions.
- `*_sample_to_sample_correlation.png` — pairwise sample correlation.

To also compare your sites against a known database derived bedfile, add:
```bash
  --rmbase_database_file m6A:/path/to/rmbase_m6A.bed
```

---

### 2. Metagene profile

The `metagene_plot` command maps modification frequencies along a normalised gene body (5′UTR → CDS → 3′UTR), showing where modifications preferentially accumulate.

```bash
dmode metagene_plot \
  --metadata metadata_gene.tsv \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/metagene \
  --modification m6A \
  --metagene_min_coverage 10
```

Add `--gene_id ENSG00000XXXXXX` to produce a profile for a single gene of interest.

---

### 3. Differential modification at the gene level

The `diff_gene_modification` command tests each modification site for a significant difference in modification frequency between conditions. It uses a logistic-regression-based test with either SLIM or Benjamini–Hochberg correction.

```bash
dmode diff_gene_modification \
  --metadata metadata_gene.tsv \
  --reference_levels UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/diff_gene_mod \
  --alpha 0.05 \
  --coverage_filter 20 \
  --frequency_filter 10 \
  --p_value_correction SLIM \
  --modification m6A
```

To restrict the analysis to a single chromosome:
```bash
  --chromosome chr1
```

Outputs under `results/diff_gene_mod/diff_gene_modification/<HEK293_VS_UHRR>/`:
- Volcano plots (all chromosomes combined, and per chromosome).
- `*_diff_gene_mod.tsv` — all tested sites with test statistics, fold change, and adjusted p-value.
- `*_diff_gene_mod_genes.tsv` — significant sites annotated with gene names from the GTF.

---

### 4. Differential modification at the transcript level

Identical to the gene-level workflow but operates on transcriptome-aligned Modkit BED files.

```bash
dmode diff_transcript_modification \
  --metadata metadata_transcript.tsv \
  --reference_levels UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/diff_transcript_mod \
  --alpha 0.05 \
  --coverage_filter 5 \
  --frequency_filter 5 \
  --p_value_correction SLIM
```

---

### 5. Differential gene expression

The `diff_gene_expression` command uses `pydeseq2` to identify differentially expressed genes from raw count matrices (e.g. featureCounts output).

```bash
dmode diff_gene_expression \
  --metadata metadata_counts.tsv \
  --reference_level UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/diff_gene_exp \
  --alpha 0.05
```

Outputs for each condition comparison:
- Volcano and MA plots.
- Pairwise and global expression heatmaps.
- Sample–sample correlation matrix.
- `*_genome_diff_expression_results.tsv` — full DESeq2-style result table.

---

### 6. Differential transcript expression

Mirrors the gene expression workflow but accepts Salmon `quant.sf` files.

```bash
dmode diff_transcript_expression \
  --metadata metadata_salmon.tsv \
  --reference_level UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/diff_transcript_exp \
  --alpha 0.05
```

---

### 7. Integrated gene expression + modification analysis

This is the most comprehensive workflow. It runs both differential expression and differential modification analyses in a single command, then produces correlation scatter plots that link expression fold changes to modification frequency changes for each gene.

```bash
dmode diff_gene_expression_and_modification \
  --metadata_count_file metadata_counts.tsv \
  --metadata_modkit_bed_file metadata_gene.tsv \
  --reference_levels UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/integrated_gene \
  --alpha_modification 0.05 \
  --alpha_expression 0.05 \
  --coverage_filter 20 \
  --frequency_filter 10 \
  --p_value_correction SLIM \
  --modification m6A
```

Or using explicit file lists:
```bash
dmode diff_gene_expression_and_modification \
  --data-count-files UHRR1_counts.csv UHRR2_counts.csv \
                     HEK293_1_counts.csv HEK293_2_counts.csv \
  --data-mod-bed-files UHRR1_modkit.bed UHRR2_modkit.bed \
                       HEK293_1_modkit.bed HEK293_2_modkit.bed \
  --conditions UHRR UHRR HEK293 HEK293 \
  --samplenames UHRR1 UHRR2 HEK293_1 HEK293_2 \
  --reference_levels UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/integrated_gene \
  --alpha_modification 0.05 \
  --alpha_expression 0.05
```

Additional outputs beyond the individual analyses:
- Expression vs. modification difference scatter plot ("eruption plot").
- Expression–modification correlation scatter plot.
- Bar plot of the top 50 genes carrying the most significant differential modification sites.

The transcript-level equivalent is:
```bash
dmode diff_transcript_expression_and_modification \
  --metadata_count_file metadata_salmon.tsv \
  --metadata_modkit_bed_file metadata_transcript.tsv \
  --reference_levels UHRR \
  --gtf-file gencode.v49.annotation.gtf \
  --output_folder results/integrated_transcript \
  --alpha_modification 0.05 \
  --alpha_expression 0.05
```

---

### Common options across all commands

| Flag | Default | Description |
|---|---|---|
| `--modification` | *(all detected)* | Restrict to one or more modification types: `m6A`, `Am`, `Ino`, `pseU`, `Um`, `Gm`, `Cm`, `m5C`. |
| `--coverage_filter` / `-f` | `20` | Minimum read coverage for a site to be included. |
| `--frequency_filter` / `-q` | `10` | Minimum modification frequency (%) for inclusion. |
| `--alpha` / `-a` | `0.01` | Significance threshold for statistical tests. |
| `--p_value_correction` / `-v` | `SLIM` | Multiple-testing correction: `SLIM` or `BH`. |
| `--chromosome` / `-x` | *(all)* | Restrict to a single chromosome. |
| `--output_folder` / `-o` | *(required)* | All results, plots, and logs are written here. |

Run `dmode <command> --help` for the full argument list of any command.

---

## Command Reference

| Command | Purpose | Key outputs |
|---|---|---|
| `gene_basic_statistics` | Exploratory QC on gene-level modification tables. | PCA/UMAP, violin/bar plots, RMBase intersections. |
| `transcript_basic_statistics` | Same as above on transcript-level inputs. | PCA/UMAP, distribution plots, RMBase overlap. |
| `metagene_plot` | Metagene coverage profiles from gene-level Modkit BEDs. | Coverage curves per modification type. |
| `diff_gene_modification` | Gene-level differential modification across conditions. | Volcano plots, significant loci & gene tables. |
| `diff_transcript_modification` | Transcript-level differential modification. | Volcano plots, significant transcript tables. |
| `diff_gene_expression` | Differential gene expression via `pydeseq2`. | MA/volcano plots, heatmaps, correlation plots. |
| `diff_transcript_expression` | Transcript-level DE analysis. | MA/volcano plots, heatmaps, correlations. |
| `diff_gene_expression_and_modification` | Joint gene-level expression + modification workflow. | All DE & modification plots, correlation summaries. |
| `diff_transcript_expression_and_modification` | Joint transcript-level workflow. | Same as above for transcripts. |

---

## Python API

The `DmodE` class exposes the same functionality as the CLI, making it suitable for interactive notebooks or custom pipelines.

### Basic structure

Every workflow follows the same three-step pattern:
1. **Preprocess** — load and filter raw data.
2. **Analyse** — run statistical tests.
3. **Visualise / export** — generate plots and annotated tables.

### Gene-level differential modification

```python
from dmode import (
    DmodE,
    diff_gene_mod_cond_extraction,
    diff_gene_mod_cond_generate_volcano_plots,
    diff_gene_mod_cond_extract_significant_positions,
    _prepare_gene_annotation_lookup,
)

# --- 1. Preprocess ---
dme = DmodE()
dme.gene_mod_preprocess_data(
    data_files=[
        "UHRR1_modkit.bed", "UHRR2_modkit.bed",
        "HEK293_1_modkit.bed",   "HEK293_2_modkit.bed",
    ],
    conditions=["UHRR", "UHRR", "HEK293", "HEK293"],
    samplenames=["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"],
    reference_levels=["UHRR"],
    coverage_filter=20,
    mod_freq_filter=10,
)

# Or load from a metadata sheet:
# dme.gene_mod_preprocess_data(metadata="metadata_gene.tsv",
#                              reference_levels=["UHRR"],
#                              coverage_filter=20, mod_freq_filter=10)

# --- 2. Analyse ---
diff_gene_mod_cond_extraction(
    dmode_obj=dme,
    output_folder="results/",
    correction_method="SLIM",   # or "BH"
)

# --- 3. Visualise & export ---
gtf_file = "gencode.v49.annotation.gtf"
alpha = 0.05
comparison = "HEK293_VS_UHRR"

annotations = _prepare_gene_annotation_lookup(dme, gtf_file, alpha, "results/")

# Volcano plot across all chromosomes
fig = diff_gene_mod_cond_generate_volcano_plots(
    dme.diff_gene_mod_chr_names,
    dme.diff_gene_mod_comparisons_dict[comparison],
    alpha=alpha,
    modtype="m6A",
    annotation_df=annotations.get(comparison),
)
fig.savefig("results/volcano_all_chrom.png", dpi=300)

# Significant sites table
sig_df = diff_gene_mod_cond_extract_significant_positions(
    dmode_obj=dme,
    ref_level="UHRR",
    cond_level="HEK293",
    alpha=alpha,
    chr="",            # empty string = all chromosomes
    output_folder="results/",
)
```

### Transcript-level differential modification

```python
from dmode import (
    DmodE,
    diff_transcript_mod_cond_extraction,
    diff_transcript_mod_cond_generate_volcano_plot_single_chr,
    diff_transcript_mod_cond_extract_significant_positions,
    _prepare_transcript_annotation_lookup,
)

dme = DmodE()
dme.transcript_mod_preprocess_data(
    metadata="metadata_transcript.tsv",
    reference_levels=["UHRR"],
    coverage_filter=5,
    mod_freq_filter=5,
)

diff_transcript_mod_cond_extraction(
    dmode_obj=dme, output_folder="results/", correction_method="SLIM"
)

annotations = _prepare_transcript_annotation_lookup(
    dme, "gencode.v49.annotation.gtf", 0.05, "results/"
)

comparison = "HEK293_VS_UHRR"
fig = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
    "",   # empty = all transcripts
    dme.diff_transcript_mod_comparisons_dict[comparison],
    alpha=0.05,
    modtype="m6A",
    annotation_df=annotations.get(comparison),
)
```

### Differential gene expression

```python
import pandas as pd
from dmode import DmodE, diff_exp_analysis, diff_exp_volcano_plot, diff_exp_ma_plot

dme = DmodE()
counts_df, condition_dict = dme.diff_gene_exp_preprocess_data(
    metadata="metadata_counts.tsv",
    reference_levels=["UHRR"],
    output_folder="results/",
)

result_df = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=condition_dict,
    alpha=0.05,
    output_folder="results/",
    gtf_file="gencode.v49.annotation.gtf",
    operational_level="genome",
)

result_df = pd.DataFrame(result_df)
diff_exp_volcano_plot(result_df=result_df, output_folder="results/",
                      pvalue_threshold=0.05, lfc_threshold=1,
                      ref_level="UHRR", first_level="HEK293",
                      operational_level="genome")
diff_exp_ma_plot(result_df=result_df, output_folder="results/",
                 pvalue_threshold=0.05, lfc_threshold=1,
                 ref_level="UHRR", first_level="HEK293",
                 operational_level="genome")
```

For transcript-level expression, replace `diff_gene_exp_preprocess_data` with `diff_transcript_exp_preprocess_data` and set `operational_level="transcriptome"`.


### Integrated expression + modification analysis

This is the API equivalent of the CLI's `diff_gene_expression_and_modification` command. A single `DmodE` object holds both data types, the two differential workflows are run in turn, and three joint plots — only available in this combined mode — link expression fold-change to modification frequency change for every gene.

```python
import pandas as pd
from dmode import (
    DmodE,
    diff_gene_mod_cond_extraction,
    diff_exp_analysis,
    _prepare_gene_annotation_lookup,
    gene_expression_vs_modification_differcence_scatterplot,
    expression_modification_correlation_scatterplot,
    plot_top50_genes_with_most_number_of_significant_modifications,
)

GTF, OUTPUT, ALPHA = "gencode.v49.annotation.gtf", "results/", 0.05
CONDITIONS  = ["UHRR", "UHRR", "HEK293", "HEK293"]
SAMPLENAMES = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

# 1. Preprocess BOTH expression and modification data on the same DmodE object
dme = DmodE()
counts_df, conditions = dme.diff_gene_exp_preprocess_data(
    data_files=[
        "UHRR1_counts.csv", "UHRR2_counts.csv",
        "HEK293_1_counts.csv",   "HEK293_2_counts.csv",
    ],
    conditions=CONDITIONS,
    samplenames=SAMPLENAMES,
    reference_levels=["UHRR"],
    output_folder=OUTPUT,
)
dme.gene_mod_preprocess_data(
    data_files=[
        "UHRR1_modkit.bed", "UHRR2_modkit.bed",
        "HEK293_1_modkit.bed",   "HEK293_2_modkit.bed",
    ],
    conditions=CONDITIONS,
    samplenames=SAMPLENAMES,
    reference_levels=["UHRR"],
    coverage_filter=20,
    mod_freq_filter=10,
)
# (Metadata-sheet form: pass metadata="metadata_counts.tsv" and metadata="metadata_gene.tsv"
#  to the two preprocess calls instead of explicit file lists.)

# 2. Run differential modification and differential expression
mod_comparisons, _ = diff_gene_mod_cond_extraction(
    dmode_obj=dme,
    correction_method="SLIM",
    output_folder=OUTPUT,
)
expression_result = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=conditions,
    alpha=ALPHA,
    output_folder=OUTPUT,
    gtf_file=GTF,
    operational_level="genome",
)

# 3. Annotate significant positions and assemble the integrated input dict
annotations = _prepare_gene_annotation_lookup(dme, GTF, ALPHA, OUTPUT)
comparison  = "HEK293_VS_UHRR"

comparison_data = {
    "modification": mod_comparisons[comparison],
    "expression":   pd.DataFrame(expression_result),
    "annotation":   annotations.get(comparison),
}

# 4. Joint visualisations unique to this workflow
gene_expression_vs_modification_differcence_scatterplot(
    comparison_data, OUTPUT, operational_level="genome"
)   # "eruption" plot — Δexpression vs Δmodification per gene

expression_modification_correlation_scatterplot(
    comparison_data, OUTPUT, operational_level="genome"
)   # per-gene correlation between expression and modification levels

plot_top50_genes_with_most_number_of_significant_modifications(
    comparison_data, OUTPUT, operational_level="genome"
)   # bar chart of the most heavily modified genes
```

You can of course also call the standard volcano/MA/heatmap plots from the gene modification and gene expression workflows on the same `DmodE` object — the integrated workflow is a strict superset.

**Transcript-level equivalent.** Replace `diff_gene_exp_preprocess_data` → `diff_transcript_exp_preprocess_data`, `gene_mod_preprocess_data` → `transcript_mod_preprocess_data`, `diff_gene_mod_cond_extraction` → `diff_transcript_mod_cond_extraction`, `_prepare_gene_annotation_lookup` → `_prepare_transcript_annotation_lookup`, and set `operational_level="transcriptome"` everywhere. The three joint-plot functions are the same.



### Exploratory statistics

```python
from dmode import (
    DmodE,
    generate_PCA_Plot, generate_UMAP_plot,
    generate_modsite_barplot,
    generate_modification_distribution_plot,
    generate_modification_violin_plot,
    generate_sample_to_sample_correlation_plot,
    moddict_mapping,
)

dme = DmodE()
dme.gene_mod_preprocess_data(
    metadata="metadata_gene.tsv",
    coverage_filter=20,
    mod_freq_filter=10,
    comparison=False,   # skip differential testing, only compute basic stats
)

dataframes = dme.basic_gene_mod_dataframe_condition_dict

generate_PCA_Plot(dataframes, modification_code="m6A",
                  output_folder="results/", annotate_samples=False,
                  operational_level="gene")
generate_UMAP_plot(dataframes, modification_code="m6A",
                   output_folder="results/", annotate_samples=False,
                   operational_level="gene")
generate_modsite_barplot(dataframes, modification_code="m6A",
                         output_folder="results/", operational_level="gene")
generate_modification_distribution_plot(dataframes, modification_code="m6A",
                                        output_folder="results/", operational_level="gene")
generate_modification_violin_plot(dataframes, modification_code="m6A",
                                  output_folder="results/", operational_level="gene")
generate_sample_to_sample_correlation_plot(dataframes, modification_code="m6A",
                                           output_folder="results/", operational_level="gene")
```

### Metagene analysis

```python
from dmode import DmodE, prepare_gene_body_coverage, metagene_body_coverage

gtf_df, gene_body_df = prepare_gene_body_coverage(gtf_file="gencode.v49.annotation.gtf")

(interpolated_df, feature_df,
 fig_all, fig_bar, fig_intronic, fig_exonic) = metagene_body_coverage(
    gtf_df=gtf_df,
    gene_body_df=gene_body_df,
    data_files=["UHRR1_modkit.bed", "UHRR2_modkit.bed",
                "HEK293_1_modkit.bed",   "HEK293_2_modkit.bed"],
    conditions=["UHRR", "UHRR", "HEK293", "HEK293"],
    samplenames=["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"],
    normalization="max",
    coverage_filter=10,
    mod_type="m6A",          # m6A base code
    gene_id="",            # empty = all genes; provide an Ensembl ID for single-gene profiles
    output_folder="results/",
)
```

### AvailableModifications

| Modification | `ChEBI / Modkit Code` |
|---|---|
| m6A | `"a"` |
| Am | `"69426"` |
| Inosine | `"17596"` |
| Pseudouridine (ψ) | `"17802"` |
| Um | `"19227"` |
| Gm | `"19229"` |
| Cm | `"19228"` |
| m5C | `"m"` |

---

## HTML Reports

After every CLI command completes, DModE attempts to generate a `dmode_report.html` in the output folder. This self-contained file:

- Scans every analysis subfolder for PNG/SVG figures.
- Groups results by analysis type, modification, and condition comparison.
- Provides a responsive Bootstrap-based interface with click-to-zoom plots.

You can also regenerate a report manually:

```python
from dmode.report.generate_report import generate_html_report
report_path = generate_html_report("results/diff_gene_mod")
print(f"Report written to: {report_path}")
```

Open the resulting HTML file directly in any browser. See `REPORT_GENERATION.md` for a detailed walkthrough.

---

## License

Distributed under the [MIT License](LICENSE). Contributions are welcome under the same terms.
