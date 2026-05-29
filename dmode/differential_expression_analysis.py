import logging
import warnings
import pandas as pd
import numpy as np
from adjustText import adjust_text
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA
from matplotlib.offsetbox import AnchoredText

import matplotlib.pyplot as plt
import seaborn as sns
import os
from dmode.utility import parse_gtf, gtf_to_df

logger = logging.getLogger(__name__) 


def merge_single_sample_feature_count_tables(input_files: list, conditions: list, samplenames:list, output_folder: str="")->pd.DataFrame | dict:
    """
    Merge multiple featureCounts output files into a single count matrix.
    
    This function combines individual featureCounts output files from multiple samples
    into a unified count matrix suitable for differential expression analysis.
    
    Parameters
    ----------
    input_files : list
        List of file paths to featureCounts output files (one per sample).
    conditions : list
        List of condition labels corresponding to each sample (e.g., ['control', 'treatment']).
    samplenames : list
        List of sample names corresponding to each input file.
    output_folder : str
        Directory path where the merged count table will be saved.
    
    Returns
    -------
    pd.DataFrame
        Merged count matrix with genes as rows and samples as columns.
    dict
        Dictionary mapping sample names to their corresponding conditions.
    
    Notes
    -----
    The output file 'merged_feature_counts.tsv' is saved to the specified output folder.
    """
    df_temp = pd.read_csv(input_files[0], sep="\t", header=1)
    df_construct = pd.DataFrame([[0 for i in range(len(samplenames))] for k in range(df_temp.shape[0])],columns=samplenames)
    df = pd.concat([df_temp.iloc[:,0],df_construct], axis = 1)
    df.columns = ["Name"] + samplenames
    condition_dict = dict(zip(samplenames, conditions))
    for input_file, samplename in zip(input_files, samplenames):
        df_temp = pd.read_csv(input_file, sep="\t", header=1)
        df_temp.columns = ['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length','Counts']
        df[samplename] = df_temp['Counts']
    if output_folder != "":
        output_file = os.path.join(output_folder,"merged_feature_counts.tsv")
        df.to_csv(output_file, sep="\t", index=False)
    return df,condition_dict

def merge_single_sample_salmon_count_tables(input_files: list, conditions: list, samplenames:list, output_folder: str="")->pd.DataFrame | dict:
    """
    Merge multiple Salmon quantification files into a single count matrix.
    
    This function combines individual Salmon output files from multiple samples
    into a unified count matrix for differential expression analysis.
    
    Parameters
    ----------
    input_files : list
        List of file paths to Salmon quant.sf files (one per sample).
    conditions : list
        List of condition labels corresponding to each sample (e.g., ['control', 'treatment']).
    samplenames : list
        List of sample names corresponding to each input file.
    output_folder : str
        Directory path where the merged count table will be saved.
    
    Returns
    -------
    pd.DataFrame
        Merged count matrix with transcripts/genes as rows and samples as columns.
    dict
        Dictionary mapping sample names to their corresponding conditions.
    
    Notes
    -----
    The function extracts gene/transcript IDs before the '|' separator.
    The output file 'merged_salmon_counts.tsv' is saved to the specified output folder.
    """
    df_temp = pd.read_csv(input_files[0], sep = "\t", header = 0)
    df_temp["Name"] = [i.split(sep = "|")[0] for i in df_temp["Name"]]
    df_construct = pd.DataFrame([[0 for i in range(len(samplenames))] for k in range(df_temp.shape[0])],columns=samplenames)
    df = pd.concat([df_temp.iloc[:,0],df_construct], axis = 1)
    df.columns = ["Name"] + samplenames
    condition_dict = dict(zip(samplenames, conditions))
    for input_file, samplename in zip(input_files, samplenames):
        df_temp = pd.read_csv(input_file, sep = "\t", header = 0)
        df[samplename] = df_temp["NumReads"]
    if output_folder != "":
        output_file = os.path.join(output_folder,"merged_salmon_counts.tsv")
        df.to_csv(output_file, sep="\t", index=False)
    return df,condition_dict


def prepare_deseq_dataset(counts_df: pd.DataFrame, condition_dict: dict) -> DeseqDataSet | pd.DataFrame:
    """
    Prepare and fit a DESeq2 dataset for differential expression analysis.
    
    This function initializes a DESeq2 dataset, estimates size factors, dispersions,
    and fits the model to the count data.
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Count matrix with genes/transcripts as rows and samples as columns.
        Must include a 'Name' column with feature identifiers.
    condition_dict : dict
        Dictionary mapping sample names to their experimental conditions.
    
    Returns
    -------
    DeseqDataSet
        Fitted DESeq2 dataset object ready for differential expression testing.
    pd.DataFrame
        Normalized count matrix with size-factor normalization applied.
    
    Notes
    -----
    The function performs the following DESeq2 steps:
    - Size factor estimation
    - Genewise dispersion estimation
    - Dispersion trend fitting
    - Dispersion prior fitting
    - MAP dispersion estimation
    - Log fold change estimation
    - Model refitting
    """
    counts_df_iteration = pd.DataFrame(counts_df.set_index("Name", inplace=False))
    #Transform every number to integer
    counts_df_iteration = counts_df_iteration.astype(int)
    metadata_df = pd.DataFrame.from_dict(condition_dict, orient="index", columns=["condition"])
    deseq_dataset = DeseqDataSet(
        counts=counts_df_iteration.T,
        metadata=metadata_df,
        design="condition"
    )
    deseq_dataset.fit_size_factors()
    deseq_dataset.fit_genewise_dispersions()
    deseq_dataset.fit_dispersion_trend()
    deseq_dataset.fit_dispersion_prior()
    deseq_dataset.fit_MAP_dispersions()
    deseq_dataset.fit_LFC()
    deseq_dataset.refit()
    norm_counts = deseq_dataset.layers["normed_counts"].T
    gene_ids = list(deseq_dataset.var.index)
    samples = list(deseq_dataset.obs.index)
    norm_counts = pd.DataFrame(norm_counts, index = gene_ids, columns = samples)
    return deseq_dataset, norm_counts
    
def diff_exp_analysis(counts_df: pd.DataFrame, condition_dict: dict, output_folder: str, ref_level: str, first_level: str, alpha: str=0.05, operational_level:str = "genome", gtf_file:str = "") -> pd.DataFrame:
    """
    Perform differential expression analysis using DESeq2.
    
    This function runs a complete DESeq2 differential expression analysis
    between two conditions, including statistical testing and result annotation.
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Count matrix with genes/transcripts as rows and samples as columns.
    condition_dict : dict
        Dictionary mapping sample names to their experimental conditions.
    output_folder : str
        Directory path where results will be saved.
    ref_level : str
        Reference condition level (baseline for comparison).
    first_level : str
        Test condition level (compared against reference).
    alpha : float, optional
        Significance level for adjusted p-value cutoff (default: 0.05).
    operational_level : str, optional
        Analysis level: 'genome' for genes or 'transcriptome' for transcripts (default: 'genome').
    gtf_file : str, optional
        Path to GTF annotation file for adding gene/transcript metadata (default: '').
    
    Returns
    -------
    pd.DataFrame
        Results dataframe containing:
        - log2FoldChange: Log2 fold change between conditions
        - pvalue: Raw p-values from Wald test
        - padj: Adjusted p-values (Benjamini-Hochberg)
        - baseMean: Mean normalized counts across samples
        - Additional gene/transcript annotation from GTF
    
    Notes
    -----
    Results are saved to '{output_folder}/{first_level}_VS_{ref_level}/' directory.
    Two output files are generated:
    - Differential expression results table
    - Normalized count matrix
    """
    deseq_dataset, norm_counts = prepare_deseq_dataset(counts_df, condition_dict)
    deseq_stats = DeseqStats(
    deseq_dataset,
    contrast=["condition", first_level, ref_level],
    alpha=alpha,
    cooks_filter=True,
    independent_filter=True,
    )
    deseq_stats.run_wald_test()
    #deseq_stats.p_values
    # if deseq_stats.independent_filter:
    #     deseq_stats._independent_filtering()
    # else:
    deseq_stats._p_value_adjustment()
        #deseq_stats.padj
    deseq_stats.summary()
    result_df = deseq_stats.results_df
    if output_folder != "":
        if not os.path.exists(os.path.join(output_folder,f"{first_level}_VS_{ref_level}")):
            os.makedirs(os.path.join(output_folder,f"{first_level}_VS_{ref_level}"), exist_ok=True)
    gtf_df = gtf_to_df(gtf_file)
    if operational_level == "genome":
        gtf_df = gtf_df[gtf_df["feature"] == "gene"]
        result_df = result_df.merge(gtf_df, left_index=True, right_on="gene_id", how="inner")
    elif operational_level == "transcriptome":
        gtf_df = gtf_df[gtf_df["feature"] == "transcript"]
        result_df = result_df.merge(gtf_df, left_index=True, right_on="transcript_id", how="inner")
    if output_folder != "":
        output_file = os.path.join(output_folder,f"{first_level}_VS_{ref_level}",f"{first_level}_VS_{ref_level}_{operational_level}_diff_expression_results.tsv")
        result_df.to_csv(output_file, sep="\t")
        output_file2 = os.path.join(output_folder,f"{first_level}_VS_{ref_level}",f"{first_level}_VS_{ref_level}_{operational_level}_diff_expression_normalized_counts.tsv")
        norm_counts.to_csv(output_file2,sep="\t")
    return result_df

def diff_exp_volcano_plot(result_df: pd.DataFrame, output_folder: str = "", pvalue_threshold: float = 0.05, lfc_threshold: float = 1, ref_level: str = "control", first_level: str = "treatment", operational_level: str = "genome"):
    """
    Generate a volcano plot for differential expression results.
    
    Creates a volcano plot visualizing log2 fold changes versus adjusted p-values,
    with color-coded significant genes and text annotations.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Differential expression results from diff_exp_analysis().
    output_folder : str
        Directory path where the plot will be saved.
    pvalue_threshold : float, optional
        Adjusted p-value threshold for significance (default: 0.05).
    lfc_threshold : float, optional
        Absolute log2 fold change threshold for significance (default: 1).
    ref_level : str, optional
        Reference condition label (default: 'control').
    first_level : str, optional
        Test condition label (default: 'treatment').
    operational_level : str, optional
        Analysis level: 'genome' or 'transcriptome' (default: 'genome').
    
    Returns
    -------
    matplotlib.figure.Figure
        Volcano plot figure object.
    
    Notes
    -----
    Significant genes are colored red (upregulated) or blue (downregulated).
    Gene/transcript names are annotated on significant points.
    Plots are saved as both SVG and PNG formats.
    """
    import seaborn as sns
    
    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    
    # Define colors matching scientific palette
    color_decrease = '#4575B4'  # Blue for decrease
    color_increase = '#D73027'  # Red for increase
    color_ns = '#BDBDBD'  # Gray for not significant
    
    # Calculate -log10(padj)
    result_df['neg_log_padj'] = -np.log10(result_df['padj'])
    
    # Create masks
    upregulated = (result_df["padj"] < pvalue_threshold) & (result_df["log2FoldChange"] > lfc_threshold)
    downregulated = (result_df["padj"] < pvalue_threshold) & (result_df["log2FoldChange"] < -lfc_threshold)
    not_significant = ~(upregulated | downregulated)
    
    # Plot not significant points (gray)
    ax.scatter(
        result_df.loc[not_significant, 'log2FoldChange'],
        result_df.loc[not_significant, 'neg_log_padj'],
        color=color_ns, alpha=0.4, s=30, edgecolor='none', label='Not significant'
    )
    
    # Plot downregulated points (blue)
    ax.scatter(
        result_df.loc[downregulated, 'log2FoldChange'],
        result_df.loc[downregulated, 'neg_log_padj'],
        color=color_decrease, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant decrease'
    )
    
    # Plot upregulated points (red)
    ax.scatter(
        result_df.loc[upregulated, 'log2FoldChange'],
        result_df.loc[upregulated, 'neg_log_padj'],
        color=color_increase, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant increase'
    )
    
    if operational_level == "genome":
        name = "gene_name"
    elif operational_level == "transcriptome":
        name = "transcript_name"
    
    # Annotate top 10 most significant genes
    significant_df = result_df[upregulated | downregulated].copy()
    if len(significant_df) > 0:
        top_sig = significant_df.nlargest(min(10, len(significant_df)), 'neg_log_padj')
        texts = []
        for _, row in top_sig.iterrows():
            texts.append(ax.text(row['log2FoldChange'], row['neg_log_padj'], row[name], fontsize=6, color='black'))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    # Decorations - scientific style
    ax.set_xlabel('Log$_2$ fold change', fontsize=16)
    ax.set_ylabel('-log$_{10}$(p-adjusted)', fontsize=16)
    
    # Threshold lines
    ax.axhline(y=-np.log10(pvalue_threshold), color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=lfc_threshold, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=-lfc_threshold, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    
    ax.set_facecolor('white')
    
    # Clean axes - only show left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Add legend
    ax.legend(frameon=False, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    if output_folder != "":
        if not os.path.exists(os.path.join(output_folder,f"{first_level}_VS_{ref_level}")):
            os.makedirs(os.path.join(output_folder,f"{first_level}_VS_{ref_level}"), exist_ok=True)
        output_name = os.path.join(output_folder,f"{first_level}_VS_{ref_level}",f"{first_level}_VS_{ref_level}_{operational_level}")
        fig.savefig(f"{output_name}_volcano.svg", format="svg")
        fig.savefig(f"{output_name}_volcano.png", format="png")
    return fig

def diff_exp_ma_plot(result_df: pd.DataFrame, output_folder: str = "", pvalue_threshold: float = 0.05, lfc_threshold: float = 1, ref_level: str = "control", first_level: str = "treatment", operational_level: str = "genome"):
    """
    Generate an MA plot for differential expression results.
    
    Creates an MA plot showing the relationship between mean expression level
    and log2 fold change, with significant genes highlighted.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Differential expression results from diff_exp_analysis().
    output_folder : str
        Directory path where the plot will be saved.
    pvalue_threshold : float, optional
        Adjusted p-value threshold for significance (default: 0.05).
    lfc_threshold : float, optional
        Absolute log2 fold change threshold (shown as horizontal lines) (default: 1).
    ref_level : str, optional
        Reference condition label (default: 'control').
    first_level : str, optional
        Test condition label (default: 'treatment').
    operational_level : str, optional
        Analysis level: 'genome' or 'transcriptome' (default: 'genome').
    
    Returns
    -------
    matplotlib.figure.Figure
        MA plot figure object.
    
    Notes
    -----
    Significant genes (padj < threshold) are colored red, others are black.
    Plots are saved as both SVG and PNG formats.
    """
    mean_expression = (result_df['baseMean'])
    log2_fold_change = result_df['log2FoldChange']
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(np.log2(mean_expression + 1), log2_fold_change, c = ["black" if p > pvalue_threshold else "red" for p in result_df['padj']], alpha=0.5)
    ax.axhline(lfc_threshold, color='blue', linestyle='--')
    ax.axhline(-lfc_threshold, color='blue', linestyle='--')
    ax.set_xlabel('Log2 Mean Expression')
    ax.set_ylabel('Log2 Fold Change')
    ax.set_title(f"{first_level} vs {ref_level}")
    if output_folder != "":
            if not os.path.exists(os.path.join(output_folder,f"{first_level}_VS_{ref_level}")):
                os.makedirs(os.path.join(output_folder,f"{first_level}_VS_{ref_level}"), exist_ok=True)
            output_name = os.path.join(output_folder,f"{first_level}_VS_{ref_level}",f"{first_level}_VS_{ref_level}_{operational_level}")
            fig.savefig(f"{output_name}_MAplot.svg", format="svg")
            fig.savefig(f"{output_name}_MAplot.png", format="png")
    return fig

def diff_exp_heatmap_pairwise(result_df: pd.DataFrame, counts_df: pd.DataFrame, condition_dict: dict, output_folder: str = "", pvalue_threshold: float = 0.05, lfc_threshold: float = 1, first_level: str = "treatment", ref_level: str = "control", operational_level: str = "genome"):
    """
    Generate a clustered heatmap for significant genes between two conditions.
    
    Creates a hierarchical clustering heatmap showing expression patterns of
    significant genes across samples from two conditions.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Differential expression results from diff_exp_analysis().
    counts_df : pd.DataFrame
        Original count matrix with all samples.
    condition_dict : dict
        Dictionary mapping sample names to their experimental conditions.
    output_folder : str
        Directory path where the heatmap will be saved.
    pvalue_threshold : float, optional
        Adjusted p-value threshold for selecting significant genes (default: 0.05).
    lfc_threshold : float, optional
        Log2 fold change threshold (not currently used in filtering) (default: 1).
    first_level : str, optional
        Test condition label (default: 'treatment').
    ref_level : str, optional
        Reference condition label (default: 'control').
    operational_level : str, optional
        Analysis level: 'genome' or 'transcriptome' (default: 'genome').
    
    Returns
    -------
    seaborn.ClusterGrid
        Clustered heatmap object.
    
    Notes
    -----
    If no significant genes are found, the top 30 genes by p-value are used.
    Displays the top 50 genes with highest variance across samples.
    Expression values are log2-transformed and standardized.
    Samples are color-coded by condition.
    Plots are saved as both SVG and PNG formats.
    """
    if operational_level == "genome":
        significant_genes = result_df[result_df['padj'] < pvalue_threshold]["gene_id"]
        if list(significant_genes) == []:
            logger.info("No significant genes found for the given thresholds. Falling back to top 30 by p-value.")
            significant_genes = result_df.sort_values(by="padj",ascending=True).head(30)["gene_id"]
    elif operational_level == "transcriptome":
        significant_genes = result_df[result_df['padj'] < pvalue_threshold]["transcript_id"]
        if list(significant_genes) == []:
            logger.info("No significant transcripts found for the given thresholds. Falling back to top 30 by p-value.")
            significant_genes = result_df.sort_values(by="padj",ascending=True).head(30)["transcript_id"]
    # Abort early if we still have no candidates to plot
    if len(significant_genes) == 0:
        logger.warning("Skipping heatmap: no genes/transcripts available after filtering.")
        return None
    reference_samples = [sample for sample, condition in condition_dict.items() if condition == ref_level]
    first_level_samples = [sample for sample, condition in condition_dict.items() if condition == first_level]
    relevant_samples = reference_samples + first_level_samples
    counts_df_temp = counts_df.set_index("Name", inplace=False, drop=True)
    counts_df_filtered = pd.DataFrame(counts_df_temp[relevant_samples])
    counts_df_filtered = counts_df_filtered[counts_df_filtered.index.isin(significant_genes)]
    if counts_df_filtered.empty:
        logger.warning("Skipping heatmap: filtered count matrix is empty.")
        return None
    if operational_level == "genome":
        counts_df_filtered = pd.merge(counts_df_filtered, result_df[["gene_id","gene_name"]], left_index=True, right_on="gene_id", how="left")
        counts_df_filtered.set_index("gene_name", inplace=True, drop=True)
        counts_df_filtered.drop(columns="gene_id", inplace=True)
    elif operational_level == "transcriptome":
        counts_df_filtered = pd.merge(counts_df_filtered, result_df[["transcript_id","transcript_name"]], left_index=True, right_on="transcript_id", how="left")
        counts_df_filtered.set_index("transcript_name", inplace=True, drop=True)
        counts_df_filtered.drop(columns="transcript_id", inplace=True)
    # Normalize counts for better visualization
    normalized_counts = np.log2(counts_df_filtered + 1)
    # Guard against all-zero variance rows and columns (would create NaNs during clustering)
    top_ids = normalized_counts.var(axis=1).nlargest(50).index
    normalized_counts = normalized_counts.loc[top_ids]
    normalized_counts.columns = counts_df_filtered.columns
    # Drop columns with zero variance to avoid division-by-zero in seaborn standardization
    zero_var_cols = normalized_counts.var(axis=0)
    zero_var_cols = list(zero_var_cols[zero_var_cols == 0].index)
    if zero_var_cols:
        logger.info("Dropping heatmap columns with zero variance: %s", zero_var_cols)
        normalized_counts = normalized_counts.drop(columns=zero_var_cols)
    # Drop rows that are now constant or empty
    zero_var_rows = normalized_counts.var(axis=1)
    zero_var_rows = list(zero_var_rows[zero_var_rows == 0].index)
    if zero_var_rows:
        logger.info("Dropping heatmap rows with zero variance: %s", zero_var_rows)
        normalized_counts = normalized_counts.drop(index=zero_var_rows)
    # Final sanity checks before plotting
    if normalized_counts.empty or normalized_counts.shape[0] < 2 or normalized_counts.shape[1] < 2:
        logger.warning("Skipping heatmap: insufficient non-constant data for clustering.")
        return None
    if not np.isfinite(normalized_counts.values).all():
        logger.warning("Skipping heatmap: non-finite values detected after preprocessing.")
        return None

    # Create a color palette for conditions
    cmap = plt.get_cmap("tab20c")
    condition_colors = {condition: idx for idx, condition in enumerate(sorted(set(condition_dict.values())))}
    col_colors = [cmap(condition_colors[condition_dict[sample]]) for sample in normalized_counts.columns]
    # Plot heatmap
    try:
        fig = sns.clustermap(normalized_counts, xticklabels=True, yticklabels=True, col_colors=col_colors, standard_scale=0)
    except ValueError as exc:
        logger.warning("Skipping heatmap: clustering failed — %s", exc)
        return None
    if output_folder != "":
        if not os.path.exists(os.path.join(output_folder,f"{first_level}_VS_{ref_level}")):
            os.makedirs(os.path.join(output_folder,f"{first_level}_VS_{ref_level}"), exist_ok=True)
        output_name = os.path.join(output_folder,f"{first_level}_VS_{ref_level}",f"{first_level}_VS_{ref_level}_{operational_level}")
        fig.savefig(f"{output_name}_heatmap.svg", format="svg")
        fig.savefig(f"{output_name}_heatmap.png", format="png")
    return fig

def diff_exp_heatmap_all_conditions(result_df: pd.DataFrame, counts_df: pd.DataFrame, condition_dict: dict, output_folder: str = "", pvalue_threshold: float = 0.05, lfc_threshold: float = 1,operational_level: str = "genome"):
    """
    Generate a clustered heatmap showing significant genes across all conditions.
    
    Creates a hierarchical clustering heatmap displaying expression patterns of
    significant genes across all samples from all experimental conditions.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        Differential expression results from diff_exp_analysis().
    counts_df : pd.DataFrame
        Original count matrix with all samples.
    condition_dict : dict
        Dictionary mapping sample names to their experimental conditions.
    output_folder : str
        Directory path where the heatmap will be saved.
    pvalue_threshold : float, optional
        Adjusted p-value threshold for selecting significant genes (default: 0.05).
    lfc_threshold : float, optional
        Log2 fold change threshold (not currently used in filtering) (default: 1).
    operational_level : str, optional
        Analysis level: 'genome' or 'transcriptome' (default: 'genome').
    
    Returns
    -------
    seaborn.ClusterGrid
        Clustered heatmap object.
    
    Notes
    -----
    If no significant genes are found, the top 30 genes by p-value are used.
    Displays the top 50 genes with highest variance across all samples.
    Expression values are log2-transformed and standardized.
    All samples from all conditions are included and color-coded.
    Plots are saved to 'all_conditions' subfolder as both SVG and PNG formats.
    """
    if operational_level == "genome":
        significant_genes = result_df[result_df['padj'] < pvalue_threshold]["gene_id"]
        if list(significant_genes) == []:
            logger.info("No significant genes found for the given thresholds. Falling back to top 30 by p-value.")
            significant_genes = result_df.sort_values(by="padj",ascending=True).head(30)["gene_id"]
    elif operational_level == "transcriptome":
        significant_genes = result_df[result_df['padj'] < pvalue_threshold]["transcript_id"]
        if list(significant_genes) == []:
            logger.info("No significant transcripts found for the given thresholds. Falling back to top 30 by p-value.")
            significant_genes = result_df.sort_values(by="padj",ascending=True).head(30)["transcript_id"]
    if len(significant_genes) == 0:
        logger.warning("Skipping heatmap: no genes/transcripts available after filtering.")
        return None
    counts_df_temp = counts_df.set_index("Name", inplace=False, drop=True)
    counts_df_filtered = pd.DataFrame(counts_df_temp[list(condition_dict.keys())])
    counts_df_filtered = counts_df_filtered[counts_df_filtered.index.isin(significant_genes)]
    if counts_df_filtered.empty:
        logger.warning("Skipping heatmap: filtered count matrix is empty.")
        return None
    if operational_level == "genome":
        counts_df_filtered = pd.merge(counts_df_filtered, result_df[["gene_id","gene_name"]], left_index=True, right_on="gene_id", how="left")
        counts_df_filtered.set_index("gene_name", inplace=True,drop=True)
        counts_df_filtered.drop(columns="gene_id", inplace=True)
    elif operational_level == "transcriptome":
        counts_df_filtered = pd.merge(counts_df_filtered, result_df[["transcript_id","transcript_name"]], left_index=True, right_on="transcript_id", how="left")
        counts_df_filtered.set_index("transcript_name", inplace=True, drop=True)
        counts_df_filtered.drop(columns="transcript_id", inplace=True)
    # Normalize counts for better visualization
    normalized_counts = np.log2(counts_df_filtered + 1)
    top_ids = normalized_counts.var(axis=1).nlargest(50).index
    normalized_counts = normalized_counts.loc[top_ids]
    normalized_counts.columns = counts_df_filtered.columns
    zero_var_cols = normalized_counts.var(axis=0)
    zero_var_cols = list(zero_var_cols[zero_var_cols == 0].index)
    if zero_var_cols:
        logger.info("Dropping heatmap columns with zero variance: %s", zero_var_cols)
        normalized_counts = normalized_counts.drop(columns=zero_var_cols)
    zero_var_rows = normalized_counts.var(axis=1)
    zero_var_rows = list(zero_var_rows[zero_var_rows == 0].index)
    if zero_var_rows:
        logger.info("Dropping heatmap rows with zero variance: %s", zero_var_rows)
        normalized_counts = normalized_counts.drop(index=zero_var_rows)
    if normalized_counts.empty or normalized_counts.shape[0] < 2 or normalized_counts.shape[1] < 2:
        logger.warning("Skipping heatmap: insufficient non-constant data for clustering.")
        return None
    if not np.isfinite(normalized_counts.values).all():
        logger.warning("Skipping heatmap: non-finite values detected after preprocessing.")
        return None
    # Create a color palette for conditions
    cmap = plt.get_cmap("tab20c")
    condition_colors = {condition: idx for idx, condition in enumerate(sorted(set(condition_dict.values())))}
    col_colors = [cmap(condition_colors[condition_dict[sample]]) for sample in normalized_counts.columns]
    # Plot heatmap
    try:
        fig = sns.clustermap(normalized_counts,xticklabels=True, yticklabels=True, col_colors=col_colors, standard_scale=0)
    except ValueError as exc:
        logger.warning("Skipping heatmap: clustering failed — %s", exc)
        return None
    if output_folder != "":
        if not os.path.exists(os.path.join(output_folder,f"all_conditions")):
            os.makedirs(os.path.join(output_folder,f"all_conditions"), exist_ok=True)
        output_name = os.path.join(output_folder,f"all_conditions",f"all_conditions_{operational_level}")
        fig.savefig(f"{output_name}_heatmap_all_conditions.svg", format="svg")
        fig.savefig(f"{output_name}_heatmap_all_conditions.png", format="png")
    return fig

# Sample to sample plot funtion
def diff_exp_sample_to_sample_plot(counts_df: pd.DataFrame, output_folder: str = "", operational_level: str = "genome"):
    """
    Generate a sample-to-sample correlation heatmap.
    
    Creates a correlation heatmap showing Pearson correlation coefficients
    between all samples based on log2-transformed expression values.
    
    Parameters
    ----------
    counts_df : pd.DataFrame
        Count matrix with genes/transcripts as rows and samples as columns.
    output_folder : str
        Directory path where the plot will be saved.
    operational_level : str, optional
        Analysis level: 'genome' or 'transcriptome' (default: 'genome').
    
    Returns
    -------
    matplotlib.figure.Figure
        Sample correlation heatmap figure object.
    
    Notes
    -----
    Expression values are log2-transformed: log2(counts + 1).
    Correlation is calculated using Pearson method.
    Useful for quality control and identifying sample outliers or batch effects.
    Plots are saved to 'all_conditions' subfolder as both SVG and PNG formats.
    """
    counts_df_temp = counts_df.set_index("Name", inplace=False, drop=True).dropna()
    log_counts = np.log2(counts_df_temp + 1)
    correlation_matrix = log_counts.corr()
    g = sns.clustermap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        figsize=(10, 10)
    )
    g.fig.suptitle('Sample-to-Sample Correlation')
    if output_folder != "":
        if not os.path.exists(os.path.join(output_folder,f"all_conditions")):
            os.makedirs(os.path.join(output_folder,f"all_conditions"), exist_ok=True)
        output_name = os.path.join(output_folder,f"all_conditions",f"all_conditions_{operational_level}")
        g.savefig(f"{output_name}_sample_correlation.svg", format="svg")
        g.savefig(f"{output_name}_sample_correlation.png", format="png")
    return g


def diff_exp_pca_plot(
    counts_df: pd.DataFrame,
    condition_dict: dict,
    output_folder: str = "",
    operational_level: str = "genome",
    n_top_features: int = 500,
    annotate_samples: bool = True,
) -> plt.Figure:
    """
    Generate a PCA plot from expression count data.

    Applies pydeseq2's variance stabilizing transformation (VST) to the raw
    counts and selects the top ``n_top_features`` most variable features —
    the standard DESeq2 PCA approach.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Raw count matrix with a 'Name' column (genes/transcripts as rows,
        samples as columns).
    condition_dict : dict
        Mapping of sample names to condition labels.
    output_folder : str
        Directory to save the plot. Saved under ``all_conditions/``.
    operational_level : str
        ``'genome'`` for gene-level or ``'transcriptome'`` for transcript-level.
    n_top_features : int
        Number of most-variable features to use for PCA (default: 500).
    annotate_samples : bool
        Whether to label each point with its sample name (default: True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from dmode.base_stats import get_condition_palette, move_legend_if_exists

    # Fit DESeq2 model and apply variance stabilizing transformation
    dds, _ = prepare_deseq_dataset(counts_df, condition_dict)
    dds.vst(use_design=False)

    # dds.layers["vst_counts"] has shape (n_samples, n_genes) — AnnData convention
    gene_ids = list(dds.var.index)
    sample_ids = list(dds.obs.index)
    vst_counts = pd.DataFrame(
        dds.layers["vst_counts"].T,  # → genes × samples
        index=gene_ids,
        columns=sample_ids,
    )

    # Select top n_top_features by variance across samples
    row_vars = vst_counts.var(axis=1)
    top_features = row_vars.nlargest(min(n_top_features, len(row_vars))).index
    vst_counts_top = vst_counts.loc[top_features]  # features × samples
    n_used = vst_counts_top.shape[0]

    # Transpose: samples × features for PCA
    X = vst_counts_top.T
    sample_names = list(X.index)
    categories = [condition_dict[s] for s in sample_names]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Samples"] = sample_names

    palette = get_condition_palette(categories)

    sns.set_theme(
        style="white",
        font_scale=1.3,
        rc={
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        },
    )
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=categories,
        palette=palette,
        s=100,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
        marker="o",
    )

    move_legend_if_exists(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}% var)", fontsize=16)
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}% var)", fontsize=16)
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_facecolor("white")

    text_box = AnchoredText(f"n = {n_used} features", frameon=False, loc="upper left", pad=0.5)
    ax.add_artist(text_box)

    if annotate_samples:
        texts = []
        for i, sample in enumerate(sample_names):
            texts.append(ax.text(
                pca_df["PC1"][i],
                pca_df["PC2"][i],
                sample,
                fontsize=12,
                weight="bold",
                color="black",
                alpha=0.8,
            ))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.tight_layout()

    if output_folder:
        out_dir = os.path.join(output_folder, "all_conditions")
        os.makedirs(out_dir, exist_ok=True)
        out_base = os.path.join(out_dir, f"{operational_level}_expression_PCA")
        fig.savefig(f"{out_base}.png", format="png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_base}.svg", format="svg", bbox_inches="tight")

    return fig