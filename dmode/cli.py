import logging
import os
import numpy as np
import sys
import pandas as pd
from dmode import (
    log,
    exp,
    isinf,
    sm,
    f,
    chi2,
    rankdata,
    tqdm,
    convert_bed_to_df,
    DmodE,
    argparse,
    logReg,
    SLIMfunc,
    benjamini_hochberg,
    fisher_exact_test,
    diff_gene_mod_cond_extract_significant_positions,
    diff_gene_mod_cond_extract_significant_genes,
    diff_gene_mod_cond_generate_volcano_plots,
    diff_gene_mod_cond_generate_volcano_plot_single_chr,
    diff_gene_mod_cond_extraction,
    diff_transcript_mod_cond_extract_significant_positions,
    diff_transcript_mod_cond_extract_significant_transcripts,
    diff_transcript_mod_cond_generate_volcano_plot_single_chr,
    diff_transcript_mod_cond_extraction,
    gtf_to_df,
    merge_single_sample_feature_count_tables,
    merge_single_sample_salmon_count_tables,
    prepare_deseq_dataset,
    diff_exp_analysis,
    diff_exp_volcano_plot,
    diff_exp_heatmap_all_conditions,
    diff_exp_ma_plot,
    diff_exp_heatmap_pairwise,
    diff_exp_sample_to_sample_plot,
    diff_exp_pca_plot,
    generate_PCA_Plot,
    generate_UMAP_plot,
    generate_modsite_barplot,
    generate_modification_distribution_plot,
    generate_modification_violin_plot,
    generate_sample_to_sample_correlation_plot,
    barplot_intersections_database,
    gene_expression_vs_modification_differcence_scatterplot,
    expression_modification_correlation_scatterplot,
    plot_top50_genes_with_most_number_of_significant_modifications,
    prepare_gene_body_coverage,
    metagene_body_coverage,
    _append_positions_column,
    _prepare_gene_annotation_lookup,
    _prepare_transcript_annotation_lookup,
    #generate_numberofreads_barplot,
    moddict,
    moddict_mapping,
)
from dmode.utility import moddict_mapping_reverse

### Initialize logger
logger = logging.getLogger(__name__)
stream = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(levelname)s - %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
stream.setFormatter(formatter)


# Global dictionary mapping modification type names to their base codes
BASE_MODIFICATION_TYPE_DICT = moddict()

def _generate_report_if_possible(output_folder: str):
    """
    Attempt to generate an HTML report from the output folder.
    This is called at the end of each analysis command. Non-fatal: failures are logged but don't stop execution.
    """
    try:
        from dmode.report.generate_report import generate_html_report
        analysis_folder = os.path.abspath(output_folder)
        parent_folder = os.path.dirname(analysis_folder)
        report_root = parent_folder if parent_folder and parent_folder != analysis_folder else analysis_folder
        report_path = generate_html_report(report_root)
        logger.info(f"\n✓ HTML report generated: {report_path}")
        logger.info(f"  Open in browser: file://{os.path.abspath(report_path)}\n")
    except Exception as e:
        # Non-fatal: just print a warning
        logger.warning(f"\nNote: Could not generate HTML report: {e}\n")

def get_modification_codes(modification_types, dmode_object):
    """
    Get modification codes based on user input or auto-detect from data.
    
    Parameters
    ----------
    modification_types : list or None
        List of modification type names (keys from BASE_MODIFICATION_TYPE_DICT)
        or None to auto-detect from data
    dmode_object : DmodE
        DmodE object containing preprocessed data
        
    Returns
    -------
    list
        List of modification codes (base codes) to analyze
    """
    if modification_types:
        # Convert modification type names to codes
        modification_codes = []
        for mod_type in modification_types:
            if mod_type in BASE_MODIFICATION_TYPE_DICT:
                modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
            else:
                logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
        return modification_codes
    else:
        # Auto-detect modification types from data
        present_modifications = []
        
        # Check gene-level data
        if hasattr(dmode_object, 'diff_gene_mod_dataframe_condition_dict'):
            for df_list in dmode_object.diff_gene_mod_dataframe_condition_dict.values():
                for df in list(df_list.values()):
                    present_modifications.append(list(set(df['modified base code'].to_list())))

        if hasattr(dmode_object, 'basic_gene_mod_dataframe_condition_dict'):
            for df_list in dmode_object.basic_gene_mod_dataframe_condition_dict.values():
                for df in list(df_list.values()):
                    present_modifications.append(list(set(df['modified base code'].to_list())))
        
        # Check transcript-level data
        if hasattr(dmode_object, 'diff_transcript_mod_dataframe_condition_dict'):
            for df_list in dmode_object.diff_transcript_mod_dataframe_condition_dict.values():
                for df in list(df_list.values()):
                    present_modifications.append(list(set(df['modified base code'].to_list())))

        if hasattr(dmode_object, 'basic_transcript_mod_dataframe_condition_dict'):
            for df_list in dmode_object.basic_transcript_mod_dataframe_condition_dict.values():
                for df in list(df_list.values()):
                    present_modifications.append(list(set(df['modified base code'].to_list())))
        
        present_modifications = [x for xs in present_modifications for x in xs]
        return list(set(present_modifications))

def filter_dmode_data_by_modification(dmode_object, modification_codes):
    """
    Filter DmodE object data to include only specified modification types.
    
    Parameters
    ----------
    dmode_object : DmodE
        DmodE object containing preprocessed data
    modification_codes : list
        List of modification codes to keep
        
    Returns
    -------
    None
        Modifies the dmode_object in place
    """
    if not modification_codes:
        return  # No filtering needed
        
    # Filter gene-level modification data if it exists
    if hasattr(dmode_object, 'diff_gene_mod_dataframe_condition_dict'):
        for condition, df_dict in dmode_object.diff_gene_mod_dataframe_condition_dict.items():
            for sample, df in df_dict.items():
                # Filter dataframe to only include specified modification codes
                filtered_df = df[df['modified base code'].isin(modification_codes)]
                dmode_object.diff_gene_mod_dataframe_condition_dict[condition][sample] = filtered_df
    
    # Filter transcript-level modification data if it exists
    if hasattr(dmode_object, 'diff_transcript_mod_dataframe_condition_dict'):
        for condition, df_dict in dmode_object.diff_transcript_mod_dataframe_condition_dict.items():
            for sample, df in df_dict.items():
                # Filter dataframe to only include specified modification codes
                filtered_df = df[df['modified base code'].isin(modification_codes)]
                dmode_object.diff_transcript_mod_dataframe_condition_dict[condition][sample] = filtered_df
    
    logger.info(f"Filtered data to include only modification types: {modification_codes}")

### Functions for gene based differential modification analysis between conditions ###

def diff_gene_mod_fundamental_analysis(args):
    """
    Perform differential gene modification analysis based on input metadata or explicit data files.

    This function initializes a `DmodE` analysis object, preprocesses the input data according to
    the specified filters, and then performs condition-based differential modification extraction.
    The user can either provide a metadata file describing samples and conditions, or directly
    supply lists of data files and corresponding condition labels.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:
        
        metadata : str
            Path to a tab-separated metadata file containing sample annotations.
            If provided, this takes precedence over direct file inputs.
        data_files : list of str
            List of file paths to input data tables. Used if no metadata file is given.
        conditions : list of str
            Condition labels corresponding to the `data_files`. Used if no metadata file is given.
        samplenames : list of str
            Optional list of sample names corresponding to the input files.
        reference_levels : list of str
            Reference levels for conditions to be used in differential analysis.
        minimum_coverage : int or float
            Minimum coverage threshold for including modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for inclusion.
        output_folder : str
            Directory path where output results will be saved.
        pvalue_correction : str
            Method used for p-value correction (e.g., 'bonferroni', 'fdr').

    Returns
    -------
    DmodE
        The `DmodE` object containing preprocessed data and results of the differential
        modification analysis.

    Raises
    ------
    ValueError
        If neither a metadata file nor data files with conditions are provided.

    Notes
    -----
    - The function assumes that `DmodE` and `diff_gene_mod_cond_extraction` are available in the namespace.
    - When both `metadata` and `data_files` are provided, the metadata input is prioritized.
    """
    if args.metadata != "":
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(metadata=args.metadata, reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.diff_gene_mod_data_files,DmodE_object.diff_gene_mod_conditions, DmodE_object.diff_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")
        


        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)
        
        diff_gene_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    elif args.data_files != [] and args.conditions != [] and args.samplenames != []:
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(data_files=args.data_files,conditions=args.conditions,samplenames=args.samplenames,reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.diff_gene_mod_data_files,DmodE_object.diff_gene_mod_conditions, DmodE_object.diff_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")
        
        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        diff_gene_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    else:
        logger.error("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
        raise ValueError("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
    return DmodE_object


def differential_gene_modification_condition_comparison(args):
    """
    Perform a full differential gene modification analysis and generate visual and tabular outputs.

    This function runs the complete pipeline for identifying and visualizing differentially modified
    positions between experimental conditions. It first executes the fundamental analysis using 
    `diff_gene_mod_fundamental_analysis`, then produces volcano plots, extracts statistically 
    significant modification sites, and maps them to annotated genes based on a provided GTF file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:
        
        metadata : str
            Path to a tab-separated metadata file describing sample information and conditions.
        data_files : list of str
            List of file paths to input data tables. Used if no metadata file is given.
        conditions : list of str
            Experimental condition labels corresponding to the `data_files`.
        samplenames : list of str
            Optional list of sample names corresponding to the input files.
        reference_levels : list of str
            Reference condition levels used as baselines in the differential analysis.
        minimum_coverage : int or float
            Minimum coverage threshold for including modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for inclusion.
        output_folder : str
            Path to the output directory where all results and figures will be saved.
        pvalue_correction : str
            Statistical correction method for multiple testing (e.g., 'bonferroni', 'fdr').
        alpha : float
            Significance threshold for determining differential modification (e.g., 0.05).
        chrom : str
            Chromosome name for which to restrict analysis and plotting.
            If empty (""), analysis is performed across all chromosomes.
        gtf_file : str
            Path to the GTF annotation file used to map significant sites to genes.

    Returns
    -------
    None
        The function saves all output plots and tables to disk within the specified
        `output_folder`. The main analysis object (`DmodE_object`) is generated internally
        but not returned.

    Outputs
    -------
    - Volcano plots for each condition comparison:
        * Combined across all chromosomes
        * Individual plots per chromosome (if specified)
    - Tables of significantly differentially modified positions
    - Gene-level annotation of significant modification sites

    Raises
    ------
    ValueError
        If required input arguments for `diff_gene_mod_fundamental_analysis` are missing.

    Notes
    -----
    - Volcano plots are saved at 500 DPI for publication-quality resolution.
    - The function assumes that the following functions are available in the namespace:
      `diff_gene_mod_fundamental_analysis`, 
      `diff_gene_mod_cond_generate_volcano_plot_single_chr`, 
      `diff_gene_mod_cond_generate_volcano_plots`, 
      `diff_gene_mod_cond_extract_significant_positions`, 
      `diff_gene_mod_cond_extract_significant_genes`, and `gtf_to_df`.
    - Directory structure is automatically created if missing.

    See Also
    --------
    diff_gene_mod_fundamental_analysis : Core preprocessing and statistical testing.
    diff_gene_mod_cond_generate_volcano_plots : Multi-chromosome volcano plot generation.
    diff_gene_mod_cond_extract_significant_positions : Extraction of significant loci.
    diff_gene_mod_cond_extract_significant_genes : Gene annotation for significant sites.
    """
    # Create function-specific output folder
    logger.info("Preprocessing for differential gene modification analysis")
    args.output_folder = os.path.join(args.output_folder, "diff_gene_modification")
    os.makedirs(args.output_folder, exist_ok=True)
    DmodE_object = diff_gene_mod_fundamental_analysis(args)
    
    # Check how many modification types are present
    modification_codes = get_modification_codes(args.modification if hasattr(args, 'modification') else None, DmodE_object)
    generate_all_mods_folder = len(modification_codes) > 1
    comparison_annotations = _prepare_gene_annotation_lookup(
        DmodE_object,
        args.gtf_file,
        args.alpha,
        args.output_folder,
    )

    if generate_all_mods_folder:
        logger.info("Generate volcano plots for all modifications combined")
        for comparison in DmodE_object.diff_gene_mod_comparisons_dict.keys():
            if args.chrom == "":
                logger.info("Volcano plots")
                fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                    "",
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    annotation_df=comparison_annotations.get(comparison),
                )
                fig2 = diff_gene_mod_cond_generate_volcano_plots(
                    DmodE_object.diff_gene_mod_chr_names,
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    annotation_df=comparison_annotations.get(comparison),
                )

                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/all_modifications", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_chromosomes_in_one_volcano_plot.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_chromosomes_in_one_volcano_plot.svg", format="svg"
                    )
                    fig2.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_chromosomes_grid_volcano_plot.png"
                    )
                    fig2.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_chromosomes_grid_volcano_plot.svg", format="svg"
                    )
            else:
                fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                    args.chrom,
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    annotation_df=comparison_annotations.get(comparison),
                )
                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/all_modifications", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{args.chrom}_{comparison}_volcano_plot.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{args.chrom}_{comparison}_volcano_plot.svg", format="svg"
                    )

    logger.info("Generate volcano plots for individual modifications")
    for comparison in DmodE_object.diff_gene_mod_comparisons_dict.keys():
        for mod_code in modification_codes:
            mod_code = moddict_mapping_reverse(mod_code)
            if args.chrom == "":
                fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                    "",
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    modtype=mod_code,
                    annotation_df=comparison_annotations.get(comparison),
                )
                fig2 = diff_gene_mod_cond_generate_volcano_plots(
                    DmodE_object.diff_gene_mod_chr_names,
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    modtype=mod_code,
                    annotation_df=comparison_annotations.get(comparison),
                )

                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/{mod_code}", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_chromosomes_in_one_volcano_plot_{mod_code}.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_chromosomes_in_one_volcano_plot_{mod_code}.svg", format="svg"
                    )
                    fig2.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_chromosomes_grid_volcano_plot_{mod_code}.png"
                    )
                    fig2.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_chromosomes_grid_volcano_plot_{mod_code}.svg", format="svg"
                    )
            else:
                fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                    args.chrom,
                    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    modtype=mod_code,
                    annotation_df=comparison_annotations.get(comparison),
                )
                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/{mod_code}", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{args.chrom}_{comparison}_volcano_plot_{mod_code}.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{args.chrom}_{comparison}_volcano_plot_{mod_code}.svg", format="svg"
                    )
    logger.info("Collect significant differentially modified positions")
    for comparison in DmodE_object.diff_gene_mod_comparisons_dict.keys():
        cond_level = comparison.split("_VS_")[0]
        reference_level = comparison.split("_VS_")[1]
        if args.chrom == "":
            diff_gene_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha,
                chr="",
                output_folder=args.output_folder,
            )
        else:
            diff_gene_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha,
                chr=args.chrom,
                output_folder=args.output_folder,
            )
    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    logger.info("Done")




### Functions for transcript based differential modification analysis between conditions ###
def diff_transcript_mod_fundamental_analysis(args):
    """
    Perform fundamental differential transcript modification analysis.

    This function initializes a `DmodE` object and conducts the core preprocessing and 
    statistical analysis required to identify differentially modified transcript positions 
    between experimental conditions. Input can be provided either through a metadata file 
    describing samples and conditions or directly via lists of data files and condition labels.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:

        metadata : str
            Path to a tab-separated metadata file containing sample annotations.
            If provided, this input takes precedence over explicit file lists.
        data_files : list of str
            List of file paths to per-sample transcript-level modification data.
            Used if no metadata file is specified.
        conditions : list of str
            Condition labels corresponding to the `data_files`. Required if using direct input.
        samplenames : list of str
            Optional list of sample names corresponding to the input data files.
        reference_levels : list of str
            Reference condition levels used as baselines for differential analysis.
        minimum_coverage : int or float
            Minimum read coverage threshold required for a site to be included in the analysis.
        minimum_mod_frequency : float
            Minimum modification frequency required for inclusion of a site.
        output_folder : str
            Path to the directory where output results and intermediate data will be saved.
        pvalue_correction : str
            Statistical correction method applied to p-values (e.g., 'bonferroni', 'fdr').

    Returns
    -------
    DmodE
        A fully initialized `DmodE` object containing preprocessed transcript-level data
        and results from the differential modification analysis.

    Raises
    ------
    ValueError
        If neither a metadata file nor lists of data files and conditions are provided.

    Notes
    -----
    - When both `metadata` and explicit file inputs are given, the metadata input is prioritized.
    - The function depends on the presence of the following functions in the namespace:
      `DmodE`, `diff_transcript_mod_cond_extraction`.
    - This function only performs the core extraction and statistical testing;
      visualization and downstream annotation steps should be handled separately.
    """
    
    if args.metadata != "":
        DmodE_object = DmodE()
        DmodE_object.transcript_mod_preprocess_data(metadata=args.metadata, reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)


        logger.info("Metadata\n")
        for data_file,condition,sample in zip(DmodE_object.diff_transcript_mod_data_files,DmodE_object.diff_transcript_mod_conditions,DmodE_object.diff_transcript_mod_samplenames):
            logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    
    elif args.data_files != [] and args.conditions != []:
        DmodE_object = DmodE()
        DmodE_object.transcript_mod_preprocess_data(data_files=args.data_files,conditions=args.conditions,samplenames=args.samplenames, reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        logger.info("Metadata\n")
        for data_file,condition,sample in zip(DmodE_object.diff_transcript_mod_data_files,DmodE_object.diff_transcript_mod_conditions,DmodE_object.diff_transcript_mod_samplenames):
            logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)
        
    else:
        raise ValueError("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
    return DmodE_object


def differential_transcript_modification_condition_comparison(args):
    """
    Perform differential transcript modification analysis and generate visual and tabular outputs.

    This function executes the complete transcript-level differential modification workflow.
    It performs preprocessing and statistical analysis using `diff_transcript_mod_fundamental_analysis`,
    generates volcano plots for each condition comparison, extracts significantly modified transcript positions,
    and associates them with annotated transcripts based on a provided GTF file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:

        metadata : str
            Path to a tab-separated metadata file describing sample information and conditions.
        data_files : list of str
            List of file paths to input transcript-level modification data tables.
            Used if no metadata file is provided.
        conditions : list of str
            Experimental condition labels corresponding to the `data_files`.
        samplenames : list of str
            Optional list of sample names corresponding to the input files.
        reference_levels : list of str
            Reference condition levels used as baselines for differential analysis.
        minimum_coverage : int or float
            Minimum read coverage threshold required for inclusion of transcript sites.
        minimum_mod_frequency : float
            Minimum modification frequency threshold required for inclusion.
        output_folder : str
            Path to the output directory where all results and figures will be saved.
        pvalue_correction : str
            Statistical correction method for multiple testing (e.g., 'bonferroni', 'fdr').
        alpha : float
            Significance threshold for differential modification (e.g., 0.05).
        chrom : str
            Chromosome name for which to restrict analysis and plotting.
            If empty (""), the analysis includes all transcripts.
        gtf_file : str
            Path to the GTF annotation file used to map significant positions to transcripts.

    Returns
    -------
    None
        The function saves all output plots and result tables to the specified `output_folder`.
        The main analysis object (`DmodE_object`) is generated internally but not returned.

    Outputs
    -------
    - Volcano plots for each condition comparison:
        * Combined plot across all transcripts (if `chrom` is empty)
    - Tables of significantly differentially modified transcript positions
    - Transcript-level annotation of significant modification sites

    Raises
    ------
    ValueError
        If required input arguments for `diff_transcript_mod_fundamental_analysis` are missing.

    Notes
    -----
    - Volcano plots are saved at 500 DPI to ensure publication-quality resolution.
    - The following functions are assumed to exist in the namespace:
      `diff_transcript_mod_fundamental_analysis`,
      `diff_transcript_mod_cond_generate_volcano_plot_single_chr`,
      `diff_transcript_mod_cond_extract_significant_positions`,
      `diff_transcript_mod_cond_extract_significant_transcripts`, and `gtf_to_df`.
    - The function automatically creates output directories if they do not exist.

    See Also
    --------
    diff_transcript_mod_fundamental_analysis : Core preprocessing and statistical testing.
    diff_transcript_mod_cond_generate_volcano_plot_single_chr : Single volcano plot generation.
    diff_transcript_mod_cond_extract_significant_positions : Extraction of significant loci.
    diff_transcript_mod_cond_extract_significant_transcripts : Transcript annotation of significant sites.
    """
    # Create function-specific output folder
    logger.info("Preprocessing for differential transcript modification analysis")
    args.output_folder = os.path.join(args.output_folder, "diff_transcript_modification")
    os.makedirs(args.output_folder, exist_ok=True)
    
    DmodE_object = diff_transcript_mod_fundamental_analysis(args)
    
    # Check how many modification types are present
    modification_codes = get_modification_codes(args.modification if hasattr(args, 'modification') else None, DmodE_object)
    generate_all_mods_folder = len(modification_codes) > 1
    comparison_annotations = _prepare_transcript_annotation_lookup(
        DmodE_object,
        args.gtf_file,
        args.alpha,
        args.output_folder,
    )
    
    if generate_all_mods_folder:
        logger.info("Generate volcano plots for all modifications combined")
        for comparison in DmodE_object.diff_transcript_mod_comparisons_dict:
            if args.chrom == "":
                logger.info("Volcano plots")
                fig1 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
                    "",
                    DmodE_object.diff_transcript_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    annotation_df=comparison_annotations.get(comparison),
                )
                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/all_modifications", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_transcripts_in_one_volcano_plot.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{comparison}_all_transcripts_in_one_volcano_plot.svg", format="svg"
                    )
            else:
                fig1 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
                    args.chrom,
                    DmodE_object.diff_transcript_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    annotation_df=comparison_annotations.get(comparison),
                )
                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/all_modifications", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{args.chrom}_{comparison}_volcano_plot.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/all_modifications/{args.chrom}_{comparison}_volcano_plot.svg", format="svg"
                    )

    logger.info("Generate volcano plots for individual modifications")
    for comparison in DmodE_object.diff_transcript_mod_comparisons_dict:
        for mod_code in modification_codes:
            mod_code = moddict_mapping_reverse(mod_code)
            if args.chrom == "":
                fig1 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
                    "",
                    DmodE_object.diff_transcript_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    modtype=mod_code,
                    annotation_df=comparison_annotations.get(comparison),
                )

                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/{mod_code}", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_transcripts_in_one_volcano_plot_{mod_code}.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{comparison}_all_transcripts_in_one_volcano_plot_{mod_code}.svg", format="svg"
                    )
            else:
                fig1 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
                    args.chrom,
                    DmodE_object.diff_transcript_mod_comparisons_dict[comparison],
                    alpha=args.alpha,
                    modtype=mod_code,
                    annotation_df=comparison_annotations.get(comparison),
                )
                if args.output_folder != "":
                    os.makedirs(args.output_folder, exist_ok=True)
                    os.makedirs(f"{args.output_folder}/{comparison}/{mod_code}", exist_ok=True)
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{args.chrom}_{comparison}_volcano_plot_{mod_code}.png", dpi=500
                    )
                    fig1.savefig(
                        f"{args.output_folder}/{comparison}/{mod_code}/{args.chrom}_{comparison}_volcano_plot_{mod_code}.svg", format="svg"
                    )
    logger.info("Collect significant differentially modified positions")
    for comparison in DmodE_object.diff_transcript_mod_comparisons_dict:
        
        cond_level = comparison.split("_VS_")[0]
        reference_level = comparison.split("_VS_")[1]
        if args.chrom == "":
            diff_transcript_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha,
                chr="",
                output_folder=args.output_folder,
            )
        else:
            diff_transcript_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha,
                chr=args.chrom,
                output_folder=args.output_folder,
            )
    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    logger.info("Done")






def initialize_gene_expression_analysis(args):
    """
    Initialize and preprocess data for differential gene expression analysis.

    This function sets up a `DmodE` object and performs the preprocessing required for 
    downstream differential gene expression analysis. It loads count data and condition 
    metadata either from a tab-separated metadata file or directly from user-provided data files 
    and condition labels. The output includes the preprocessed count matrix and the condition mapping.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:

        metadata : str
            Path to a tab-separated metadata file describing sample annotations and conditions.
            If provided, this input takes precedence over explicit file and condition lists.
        data_files : list of str
            List of file paths to raw count tables or gene expression matrices.
            Used if no metadata file is provided.
        conditions : list of str
            Experimental condition labels corresponding to the `data_files`.
        samplenames : list of str
            Optional list of sample names corresponding to the input data files.
        reference_levels : list of str
            Reference condition levels to be used as baselines in differential expression analysis.
        output_folder : str
            Path to the output directory where preprocessed data and intermediate results
            will be saved.

    Returns
    -------
    tuple
        A tuple containing:
        
        DmodE_object : DmodE
            The initialized `DmodE` analysis object containing loaded data and settings.
        counts_df : pandas.DataFrame
            The preprocessed gene expression count matrix used for downstream analysis.
        condition_dict : dict
            A dictionary mapping samples to experimental conditions.

    Raises
    ------
    ValueError
        If neither a metadata file nor lists of data files and conditions are provided.

    Notes
    -----
    - The function automatically prioritizes metadata-based input if both input types are provided.
    - Preprocessing includes consistency checks for sample names and reference levels.
    - The returned objects are suitable for immediate use in differential expression testing.
    - The function assumes that `DmodE` and its method `diff_gene_exp_preprocess_data`
      are available in the namespace.

    See Also
    --------
    DmodE.diff_gene_exp_preprocess_data : Preprocessing routine for differential gene expression.
    """
    DmodE_object = DmodE()
    if args.metadata != "":
        counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(metadata = args.metadata, reference_levels= args.reference_levels, output_folder = args.output_folder)
    else:
        counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(data_files = args.data_files, conditions = args.conditions, samplenames = args.samplenames, reference_levels= args.reference_levels, output_folder = args.output_folder)
    
    logger.info("Metadata")
    for data_file,condition,sample in zip(DmodE_object.diff_gene_exp_data_files,DmodE_object.diff_gene_exp_conditions,DmodE_object.diff_gene_exp_samplenames):
        logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}")
        
    return DmodE_object, counts_df, condition_dict


### Functions for differential gene expression analysis between conditions ###
def differential_gene_expression_analysis(args):
    """
    Perform differential gene expression analysis and generate visualization outputs.

    This function executes the full gene-level differential expression (DE) workflow.
    It initializes a `DmodE` object and preprocesses input data using 
    `initialize_gene_expression_analysis`. Pairwise comparisons are then performed 
    between all experimental conditions and specified reference levels. The function 
    produces differential expression statistics and a series of publication-quality 
    plots, including volcano plots, MA plots, and heatmaps.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:

        metadata : str
            Path to a tab-separated metadata file describing sample annotations and conditions.
        data_files : list of str
            List of file paths to raw count tables. Used if no metadata file is provided.
        conditions : list of str
            Experimental condition labels corresponding to the input data files.
        samplenames : list of str
            Optional list of sample names corresponding to the input files.
        reference_levels : list of str
            Reference condition levels used as baselines for pairwise differential analysis.
        output_folder : str
            Directory where all results and generated plots will be saved.
        alpha : float
            Significance threshold (e.g., 0.05) used for adjusted p-values in DE testing.
        gtf_file : str
            Path to the GTF annotation file used to map gene IDs to genomic features.

    Returns
    -------
    None
        The function saves all result tables and visualization plots to the specified
        `output_folder`. Intermediate data and the main `DmodE` object are not returned.

    Outputs
    -------
    For each condition-vs-reference comparison, the following are generated:

    - **Statistical results table** (`*_DE_results.tsv`)
        Contains log2 fold changes, p-values, adjusted p-values, and gene annotations.
    - **Volcano plot** (`*_volcano_plot.png`)
        Depicts significance versus log2 fold change for each gene.
    - **MA plot** (`*_MA_plot.png`)
        Shows average expression (A) versus log2 fold change (M) to visualize systematic trends.
    - **Heatmaps**
        * `*_pairwise_heatmap.png` — top DE genes between each condition pair.
        * `*_all_conditions_heatmap.png` — expression heatmap across all conditions.
    - **Sample-to-sample correlation plot** (`*_sample_correlation.png`)
        Displays pairwise expression correlations across samples.

    Raises
    ------
    ValueError
        If neither a metadata file nor lists of data files and conditions are provided.

    Notes
    -----
    - Pairwise DE comparisons are automatically generated between each experimental
      condition and every reference condition defined in `args.reference_levels`.
    - The function assumes the following helper functions are available:
      `initialize_gene_expression_analysis`, `diff_exp_analysis`,
      `diff_exp_volcano_plot`, `diff_exp_ma_plot`, 
      `diff_exp_heatmap_pairwise`, `diff_exp_heatmap_all_conditions`, and
      `diff_exp_sample_to_sample_plot`.
    - All figures are saved at 500 DPI for publication-quality resolution.
    - The `operational_level` is fixed to `"genome"` for gene-level analyses.

    See Also
    --------
    initialize_gene_expression_analysis : Preprocesses data and initializes analysis.
    diff_exp_analysis : Performs statistical differential expression testing.
    diff_exp_volcano_plot : Generates volcano plots for DE results.
    diff_exp_ma_plot : Creates MA plots.
    diff_exp_heatmap_pairwise : Produces heatmaps for significant genes per comparison.
    diff_exp_heatmap_all_conditions : Generates global expression heatmaps.
    diff_exp_sample_to_sample_plot : Visualizes sample correlations based on expression data.
    """
    logger.info("Preprocessing differential gene expression analysis")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "diff_gene_expression")
    os.makedirs(args.output_folder, exist_ok=True)
    DmodE_object, counts_df, condition_dict = initialize_gene_expression_analysis(args)
    conditions = list([str(key) for key in DmodE_object.diff_gene_exp_conditions])

    # Check if at least 2 samples per condition are provided, if not raise ValueError
    conditions_check = dict((x,list(condition_dict.values()).count(x)) for x in set(list(condition_dict.values())))
    logger.info(f"Samples per condition: {conditions_check}")
    for condition in conditions_check.keys():
        if conditions_check[condition] < 2:
            raise ValueError(f"At least 2 samples per condition are required for differential gene expression analysis! Condition '{condition}' has only {conditions_check[condition]} sample.")
    
    logger.info("Expression PCA")
    diff_exp_pca_plot(counts_df=counts_df, condition_dict=condition_dict, output_folder=args.output_folder, operational_level="genome")

    for reference in np.unique(args.reference_levels):
        for condition in np.unique(conditions):
            if condition != reference:
                logger.info(f"Comparison level: {condition} VS {reference}")
                result_df = diff_exp_analysis(counts_df = counts_df, ref_level = reference, first_level = condition, condition_dict=condition_dict, alpha = args.alpha, output_folder = args.output_folder, gtf_file=args.gtf_file, operational_level="genome")
                result_df_temp = pd.DataFrame(result_df)
                logger.info("Volcano plot")
                diff_exp_volcano_plot(result_df = result_df_temp, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "genome")
                logger.info("MA plot")
                diff_exp_ma_plot(result_df = result_df_temp, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "genome")
                logger.info("Heatmap plots")
                diff_exp_heatmap_pairwise(result_df = result_df_temp, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, first_level = condition, ref_level = reference, operational_level = "genome")
                diff_exp_heatmap_all_conditions(result_df = result_df_temp, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1,operational_level = "genome")
                logger.info("Sample to Sample plot")
                diff_exp_sample_to_sample_plot(counts_df = counts_df, output_folder = args.output_folder, operational_level = "genome")

    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    logger.info("Done")




                
def initialize_transcript_expression_analysis(args):
    """
    Initialize and preprocess data for differential transcript expression analysis.

    This function sets up a `DmodE` object and performs preprocessing of transcript-level
    expression data in preparation for downstream differential expression analysis.
    Input can be supplied either as a metadata file containing sample-to-condition
    mappings or as explicit lists of data files and condition labels.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace object containing:

        metadata : str
            Path to a tab-separated metadata file describing samples and experimental conditions.
            If provided, this input takes precedence over direct file lists.
        data_files : list of str
            List of file paths to transcript-level expression matrices or count tables.
            Used if no metadata file is provided.
        conditions : list of str
            Condition labels corresponding to the `data_files`. Required if using direct input.
        samplenames : list of str
            Optional list of sample names corresponding to the input data files.
        reference_levels : list of str
            Reference condition levels used as baselines for differential expression analysis.
        output_folder : str
            Directory path where preprocessed data and intermediate files will be saved.

    Returns
    -------
    tuple
        A tuple containing:

        DmodE_object : DmodE
            The initialized `DmodE` analysis object containing transcript-level expression data
            and preprocessing parameters.
        counts_df : pandas.DataFrame
            A matrix of transcript-level expression counts for all samples.
        condition_dict : dict
            A dictionary mapping sample names to experimental conditions.

    Raises
    ------
    ValueError
        If neither a metadata file nor explicit lists of data files and conditions are provided.

    Notes
    -----
    - When both `metadata` and direct data file inputs are given, the metadata input is prioritized.
    - This function only handles preprocessing and setup; statistical analysis and visualization
      should be performed with downstream pipeline functions.
    - The function assumes that `DmodE` and its method `diff_transcript_exp_preprocess_data`
      are available in the namespace.

    See Also
    --------
    DmodE.diff_transcript_exp_preprocess_data : Preprocessing routine for transcript-level differential expression.
    differential_transcript_expression_analysis : Executes the full DE workflow including testing and visualization.
    """
    DmodE_object = DmodE()
    if args.metadata != "":
        counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(metadata = args.metadata, reference_levels= args.reference_levels, output_folder = args.output_folder)
    else:
        counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(data_files = args.data_files, conditions = args.conditions, samplenames = args.samplenames, reference_levels= args.reference_levels, output_folder = args.output_folder)
    
    logger.info("Metadata")
    for data_file,condition,sample in zip(DmodE_object.diff_transcript_exp_data_files,DmodE_object.diff_transcript_exp_conditions, DmodE_object.diff_transcript_exp_samplenames):
        logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")    
    
    return DmodE_object, counts_df, condition_dict






### Functions for differential transcript expression analysis between conditions ###
def differential_transcript_expression_analysis(args):
    """
    Perform differential transcript expression analysis and generate visualization outputs.

    This function executes the full workflow for transcript-level differential expression (DE)
    analysis, including statistical testing, volcano and MA plots, pairwise and global heatmaps,
    and sample-to-sample correlation visualization. It compares transcript expression between
    reference and experimental conditions defined in the input arguments.

    The function builds upon preprocessed data generated by
    `initialize_transcript_expression_analysis`, runs statistical testing for all pairwise
    combinations of conditions, and exports publication-ready figures and results tables.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata : str
            Path to a tab-separated metadata file describing sample-to-condition mappings.
        data_files : list of str
            List of transcript-level expression data files (if no metadata file is provided).
        conditions : list of str
            Experimental conditions corresponding to the data files.
        samplenames : list of str
            Sample names corresponding to the input data files.
        reference_levels : list of str
            Reference condition(s) used as baseline(s) for differential expression testing.
        alpha : float
            Significance threshold for adjusted p-values (e.g., 0.05).
        gtf_file : str
            Path to a GTF annotation file for mapping transcript IDs to gene annotations.
        output_folder : str
            Directory in which all plots and result tables will be stored.

    Returns
    -------
    None
        The function saves analysis results and figures directly to disk; no object is returned.

    Workflow
    --------
    1. Initializes and preprocesses transcript-level count data using `DmodE`.
    2. Iterates over all pairwise combinations of reference and non-reference conditions.
    3. Performs statistical testing via `diff_exp_analysis` to identify differentially expressed transcripts.
    4. Generates and saves the following visualizations for each comparison:
       - Volcano plots of log2 fold change vs. significance.
       - MA plots of mean expression vs. log2 fold change.
       - Pairwise condition heatmaps for significant transcripts.
       - Global heatmap across all conditions.
       - Sample-to-sample correlation plots for quality control.

    Raises
    ------
    ValueError
        If input arguments are incomplete or inconsistent (e.g., missing metadata or condition labels).

    Notes
    -----
    - Differential expression is computed at the *transcriptome* level (`operational_level="transcriptome"`).
    - Output figures and tables are saved to structured subdirectories within `args.output_folder`.
    - Each pairwise comparison is labeled as `<condition>_VS_<reference>`.

    See Also
    --------
    initialize_transcript_expression_analysis : Preprocessing of transcript-level expression data.
    diff_exp_analysis : Core statistical testing routine for differential expression.
    diff_exp_volcano_plot : Visualization of significant differential expression via volcano plots.
    diff_exp_ma_plot : Visualization of mean expression vs. fold change.
    diff_exp_heatmap_pairwise : Pairwise comparison heatmap.
    diff_exp_heatmap_all_conditions : Global heatmap across all experimental conditions.
    diff_exp_sample_to_sample_plot : Sample correlation heatmap for quality control.
    """
    logger.info("Preprocessing differential trnascript expression analysis")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "diff_transcript_expression")
    os.makedirs(args.output_folder, exist_ok=True)
    
    DmodE_object, counts_df, condition_dict = initialize_transcript_expression_analysis(args)

    # Check if at least 2 samples per condition are provided, if not raise ValueError
    conditions_check = dict((x,list(condition_dict.values()).count(x)) for x in set(list(condition_dict.values())))
    logger.info(f"Samples per condition: {conditions_check}")
    for condition in conditions_check.keys():
        if conditions_check[condition] < 2:
            raise ValueError(f"At least 2 samples per condition are required for differential transcript expression analysis! Condition '{condition}' has only {conditions_check[condition]} sample.")
    
    
    
    logger.info("Expression PCA")
    diff_exp_pca_plot(counts_df=counts_df, condition_dict=condition_dict, output_folder=args.output_folder, operational_level="transcriptome")

    for reference in np.unique(args.reference_levels):
        for condition in np.unique(DmodE_object.diff_transcript_exp_conditions):
            if condition != reference:
                logger.info(f"Comparison level: {condition} VS {reference}")
                result_df = diff_exp_analysis(counts_df = counts_df, ref_level = reference, first_level = condition, condition_dict=condition_dict, alpha = args.alpha, output_folder = args.output_folder, gtf_file=args.gtf_file, operational_level="transcriptome")
                logger.info("Volcano plot")
                diff_exp_volcano_plot(result_df = result_df, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "transcriptome")
                logger.info("MA plot")
                diff_exp_ma_plot(result_df = result_df, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "transcriptome")
                logger.info("Heatmap plots")
                diff_exp_heatmap_pairwise(result_df = result_df, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1, first_level = condition, ref_level = reference, operational_level = "transcriptome")
                diff_exp_heatmap_all_conditions(result_df = result_df, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha, lfc_threshold = 1,operational_level = "transcriptome")
                logger.info("Sample to Sample plot")
                diff_exp_sample_to_sample_plot(counts_df = counts_df, output_folder = args.output_folder, operational_level = "transcriptome")

    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    logger.info("Done")    





def gene_basic_statistics(args):
    """
    Compute and visualize fundamental statistics of gene-level RNA modification data.

    This function performs exploratory and descriptive analyses on differential gene
    modification data processed through the `DmodE` pipeline. It identifies all modification
    types present across samples and generates multiple statistical and visualization
    outputs, including PCA, UMAP, barplots, and violin plots for each detected modification
    type. Optionally, it compares modification sites with external databases.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata : str
            Path to a tab-separated metadata file describing sample-to-condition mappings.
        data_files : list of str
            List of input data files containing per-gene or site-level modification frequencies.
            Used when `metadata` is not provided.
        conditions : list of str
            Condition labels corresponding to the data files.
        samplenames : list of str
            Sample names corresponding to the input data files.
        minimum_coverage : int
            Minimum read coverage threshold for including modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for site inclusion.
        output_folder : str
            Directory where all plots and result files will be saved.
        rmbase_database_file : list of str
            Optional list of database file paths for known RNA modification sites.
            Each entry should follow the format `"ModType:/path/to/database.txt"`.
        gtf_file : str
            Path to a GTF annotation file for mapping genes and transcript features.

    Returns
    -------
    None
        The function saves all generated plots and results to the specified `output_folder`.
        No Python objects are returned.

    Workflow
    --------
    1. Preprocess modification data using `diff_gene_mod_fundamental_analysis`.
    2. Identify all unique modification types present across conditions and samples.
    3. For each detected modification type:
       - Generate PCA and UMAP plots for sample clustering.
       - Generate barplots and violin plots summarizing modification site distributions.
       - Visualize genomic modification distributions across conditions.
       - If RMBBase database paths are provided, compute overlaps between detected
         modifications and known modification sites via `barplot_intersections_database`.

    Raises
    ------
    ValueError
        If neither a metadata file nor explicit data file and condition lists are provided.

    Notes
    -----
    - The function dynamically detects all modification types (e.g., m6A, m5C, ψU) present
      in the dataset from the `modified base code` column.
    - RMBBase database comparisons require specifying recognized modification type labels
      (e.g., `"m6A"`, `"m5C"`, `"Um"`) mapped to internal base codes.


    See Also
    --------
    basic_gene_mod_fundamental_analysis : Preprocess gene-level modification data.
    generate_PCA_Plot : Principal Component Analysis visualization.
    generate_UMAP_plot : UMAP dimensionality reduction visualization.
    generate_modsite_barplot : Distribution summary of modification sites.
    generate_modification_distribution_plot : Genome-wide distribution of modification frequencies.
    generate_modification_violin_plot : Violin plots showing modification frequency variation.
    barplot_intersections_database : Comparison of detected sites with known modification databases.
    """
    logger.info("Preprocessing basic statistics")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "gene_basic_statistics")
    os.makedirs(args.output_folder, exist_ok=True)
    
    ###### PREPROCESS DATA ######
    
    if args.metadata != "":
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(metadata=args.metadata,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.basic_gene_mod_data_files,DmodE_object.basic_gene_mod_conditions, DmodE_object.basic_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

    elif args.data_files != [] and args.conditions != [] and args.samplenames != []:
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(data_files=args.data_files,conditions=args.conditions,samplenames=args.samplenames,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.basic_gene_mod_data_files,DmodE_object.basic_gene_mod_conditions, DmodE_object.basic_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")
        
        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

    else:
        logger.error("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
        raise ValueError("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")

    ##### CALCULATE BASIC STATISTICS #####

    logger.info("Calculate basic statistics")

    ### determine modification types to analyze based on user input or auto-detect
    possible_modifications = get_modification_codes(args.modification, DmodE_object)
    for possible_mod_code in possible_modifications:
        mod = moddict_mapping_reverse(possible_mod_code)
        #Only create UMAP and PCA if there are at least 2 samples per condition
        try:
            logger.info(f"PCA for {mod}")
            generate_PCA_Plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder,annotate_samples=False, operational_level="gene")
        except Exception as e:
            logger.warning(f"Could not generate PCA plot for modification {mod} due to error: {e}")
        try:
            logger.info(f"UMAP for {mod}")
            generate_UMAP_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder,annotate_samples=False, operational_level="gene")
        except Exception as e:
            logger.warning(f"Could not generate UMAP plot for modification {mod} due to error: {e}")
        logger.info(f"Barplot modification sites for {mod}")
        generate_modsite_barplot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="gene")
        logger.info(f"Plot modification probablity distribution for {mod}")
        generate_modification_distribution_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="gene")
        logger.info(f"Violin plot modification probability distribution {mod}")
        generate_modification_violin_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="gene")
        logger.info(f"Sample to sample plot of modification sites for {mod}")
        generate_sample_to_sample_correlation_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="gene")
        
        if args.rmbase_database_file != "":
            logger.info(f"Generate intersection plots of modification sites with RMBase data for {mod}")
            rmbase_database = {}
            for database_file in args.rmbase_database_file:
                database_split = database_file.split(":")
                modification_of_split = BASE_MODIFICATION_TYPE_DICT[database_split[0]]
                database_path_split = database_split[1]
                rmbase_database[modification_of_split] = database_path_split
            logger.info(f"RMBase database: {rmbase_database}")
            barplot_intersections_database(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict,rmbase_database=rmbase_database,modification=possible_mod_code, output_folder=args.output_folder)    
    
    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
        
        

def transcript_basic_statistics(args):
    """
    Compute and visualize fundamental statistics of transcript-level RNA modification data.

    This function performs exploratory and descriptive analyses on differential transcript
    modification data processed through the `DmodE` pipeline. It identifies all modification
    types present across samples and generates multiple statistical and visualization
    outputs, including PCA, UMAP, barplots, and violin plots for each detected modification
    type. Optionally, it compares modification sites with external databases.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata : str
            Path to a tab-separated metadata file describing sample-to-condition mappings.
        data_files : list of str
            List of input data files containing per-transcript or site-level modification frequencies.
            Used when `metadata` is not provided.
        conditions : list of str
            Condition labels corresponding to the data files.
        samplenames : list of str
            Sample names corresponding to the input data files.
        minimum_coverage : int
            Minimum read coverage threshold for including modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for site inclusion.
        output_folder : str
            Directory where all plots and result files will be saved.
        rmbase_database_file : list of str
            Optional list of database file paths for known RNA modification sites.
            Each entry should follow the format `"ModType:/path/to/database.txt"`.
        gtf_file : str
            Path to a GTF annotation file for mapping transcripts and features.

    Returns
    -------
    None
        The function saves all generated plots and results to the specified `output_folder`.
        No Python objects are returned.

    Workflow
    --------
    1. Preprocess modification data using `diff_transcript_mod_fundamental_analysis`.
    2. Identify all unique modification types present across conditions and samples.
    3. For each detected modification type:
       - Generate PCA and UMAP plots for sample clustering.
       - Generate barplots and violin plots summarizing modification site distributions.
       - Visualize genomic modification distributions across conditions.
       - If RMBBase database paths are provided, compute overlaps between detected
         modifications and known modification sites via `barplot_intersections_database`.
    Raises
    ------
    ValueError
        If neither a metadata file nor explicit data file and condition lists are provided.

    Notes
    -----
    - The function dynamically detects all modification types (e.g., m6A, m5C, ψU) present
      in the dataset from the `modified base code` column.
    - RMBBase database comparisons require specifying recognized modification type labels
      (e.g., `"m6A"`, `"m5C"`, `"Um"`) mapped to internal base codes.

    See Also
    --------
    basic_transcript_mod_fundamental_analysis : Preprocess transcript-level modification data.
    generate_PCA_Plot : Principal Component Analysis visualization.
    generate_UMAP_plot : UMAP dimensionality reduction visualization.
    generate_modsite_barplot : Distribution summary of modification sites.
    generate_modification_distribution_plot : Genome-wide distribution of modification frequencies.
    generate_modification_violin_plot : Violin plots showing modification frequency variation.
    barplot_intersections_database : Comparison of detected sites with known modification databases.
    """

    logger.info("Preprocessing basic statistics")

    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "transcript_basic_statistics")
    os.makedirs(args.output_folder, exist_ok=True)
    
    ###### PREPROCESS DATA ######

    if args.metadata != "":
        DmodE_object = DmodE()
        DmodE_object.transcript_mod_preprocess_data(metadata=args.metadata,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)


        logger.info("Metadata\n")
        for data_file,condition,sample in zip(DmodE_object.basic_transcript_mod_data_files,DmodE_object.basic_transcript_mod_conditions,DmodE_object.basic_transcript_mod_samplenames):
            logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

    
    elif args.data_files != [] and args.conditions != []:
        DmodE_object = DmodE()
        DmodE_object.transcript_mod_preprocess_data(data_files=args.data_files,conditions=args.conditions,samplenames=args.samplenames,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)

        logger.info("Metadata\n")
        for data_file,condition,sample in zip(DmodE_object.basic_transcript_mod_data_files,DmodE_object.basic_transcript_mod_conditions,DmodE_object.basic_transcript_mod_samplenames):
            logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)
        
    else:
        raise ValueError("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")

    logger.info("Calculate basic statistics")

    # Check if at least 4 samples in total are provided, if not, skip UMAP and PCA Plot
    at_least_4_total = True
    total_samples = sum(len(samples) for samples in DmodE_object.basic_transcript_mod_dataframe_condition_dict.values())
    if total_samples < 4:
        at_least_4_total = False
        logger.info("No sufficient samplesize for UMAP and PCA Plot - skip")
        
    
    ### determine modification types to analyze based on user input or auto-detect
    possible_modifications = get_modification_codes(args.modification, DmodE_object)
    for possible_mod_code in possible_modifications:
        mod = moddict_mapping_reverse(possible_mod_code)
        if at_least_4_total:
            logger.info(f"PCA for {mod}")
            generate_PCA_Plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder,annotate_samples=False, operational_level="transcript")
            logger.info(f"UMAP for {mod}")
            generate_UMAP_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder,annotate_samples=False, operational_level="transcript")
        logger.info(f"Barplot modification sites for {mod}")
        generate_modsite_barplot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="transcript")
        logger.info(f"Plot modification probablity distribution for {mod}")
        generate_modification_distribution_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="transcript")
        logger.info(f"Violin plot modification probability distribution for {mod}")
        generate_modification_violin_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="transcript")
        logger.info(f"Sample to sample plot of modification sites for {mod}")
        generate_sample_to_sample_correlation_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict,modification=possible_mod_code, output_folder=args.output_folder, operational_level="transcript")
    
    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    

def metagene_plot(args):
    logger.info("Preprocessing Metagene analysis")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "metagene_plot")
    os.makedirs(args.output_folder, exist_ok=True)
    
    ###### PREPROCESS DATA ######
    
    if args.metadata != "":
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(metadata=args.metadata,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.basic_gene_mod_data_files,DmodE_object.basic_gene_mod_conditions, DmodE_object.basic_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

    elif args.data_files != [] and args.conditions != [] and args.samplenames != []:
        DmodE_object = DmodE()
        DmodE_object.gene_mod_preprocess_data(data_files=args.data_files,conditions=args.conditions,samplenames=args.samplenames,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency, comparison=False)

        logger.info("Metadata\n")
        for data_file, condition, sample in zip(DmodE_object.basic_gene_mod_data_files,DmodE_object.basic_gene_mod_conditions, DmodE_object.basic_gene_mod_samplenames):
            logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")
        
        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

    else:
        logger.error("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
        raise ValueError("You must provide a tab separated metadata sheet or a list with the data filepaths and conditions to run the analysis!")
        
    possible_modifications = get_modification_codes(args.modification, DmodE_object)
    logger.info("Preparation of gene body analysis")
    gtf_df, gene_body_df = prepare_gene_body_coverage(gtf_file=args.gtf_file)
    
    logger.info("Metagene coverage plot and stacked barplots modification quantification (5'UTR, CDS, 3'UTR)")
    for possible_mod_code in np.unique(possible_modifications):
        if args.metadata != "":
            data_files = list(pd.read_csv(args.metadata, sep="\t", header=0)["File"])
            conditions = list(
                pd.read_csv(args.metadata, sep="\t", header=0)["Condition"]
            )
            conditions = [str(i) for i in conditions]
            samplenames = list(
                pd.read_csv(args.metadata, sep="\t", header=0)["Sample"]
            )
            samplenames = [str(i) for i in samplenames]
            metagene_body_coverage(gtf_df = gtf_df, gene_body_df = gene_body_df, data_files=data_files, conditions=conditions, samplenames=samplenames, normalization="max", coverage_filter=args.metagene_min_coverage, mod_type=possible_mod_code, gene_id=args.gene_id, output_folder=args.output_folder)
            
        else:
            metagene_body_coverage(gtf_df = gtf_df, gene_body_df = gene_body_df, data_files=args.data_files, conditions=args.conditions, samplenames=args.samplenames, normalization="max", coverage_filter=args.metagene_min_coverage, mod_type=possible_mod_code, gene_id=args.gene_id, output_folder=args.output_folder)

    # Generate HTML report
    _generate_report_if_possible(args.output_folder)



def initialize_gene_expression_and_modification_analysis(args):
    """
    Initialize combined gene expression and modification analysis.

    This function prepares both gene expression and RNA modification data for
    integrated differential analysis using the `DmodE` framework. It preprocesses
    read count and modification datasets, extracts differential modification sites,
    and returns data structures required for downstream correlation or joint
    expression–modification studies.

    Input data can be provided either through metadata files (for both expression
    and modification datasets) or as explicit lists of file paths and condition
    labels.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata_modkit_file : str
            Path to a metadata file describing samples and conditions for
            RNA modification (modkit) data.
        metadata_count_file : str
            Path to a metadata file describing samples and conditions for
            gene expression (count) data.
        data_count_files : list of str
            List of expression count files used if no metadata file is provided.
        data_mod_bed_files : list of str
            List of BED files containing per-site RNA modification data.
        conditions : list of str
            Experimental condition labels corresponding to the input samples.
        samplenames : list of str
            Sample names corresponding to the data files.
        reference_levels : list of str
            Reference condition(s) used as baseline(s) for both expression and
            modification analysis.
        minimum_coverage : int
            Minimum read coverage required for a site to be included in
            modification analysis.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for inclusion in analysis.
        pvalue_correction : str
            Method used for multiple testing correction of modification site
            p-values (e.g., `"fdr_bh"`, `"bonferroni"`).
        output_folder : str
            Directory where intermediate and output files will be saved.

    Returns
    -------
    tuple
        A tuple containing:

        DmodE_object : DmodE
            The initialized analysis object containing both expression and
            modification data.
        counts_df : pandas.DataFrame
            A gene-level expression matrix for all samples.
        condition_dict : dict
            A dictionary mapping sample names to experimental conditions.

    Raises
    ------
    ValueError
        If neither metadata files nor explicit file and condition lists are provided.

    Workflow
    --------
    1. Initializes a `DmodE` object for managing both data types.
    2. Preprocesses gene expression data using `diff_gene_exp_preprocess_data`.
    3. Preprocesses modification data using `diff_gene_mod_preprocess_data`.
    4. Extracts differential modification sites via `diff_gene_mod_cond_extraction`.
    5. Returns the processed data structures for downstream analyses.

    Notes
    -----
    - If both metadata-based and direct file input modes are provided, the metadata
      files take precedence.
    - The function prepares the data for joint analyses, such as correlating
      expression changes with modification dynamics at the gene level.
    - Differential modification extraction is performed automatically during
      initialization to ensure the availability of processed results for subsequent
      analyses.

    See Also
    --------
    DmodE.diff_gene_exp_preprocess_data : Preprocessing routine for gene expression data.
    DmodE.diff_gene_mod_preprocess_data : Preprocessing routine for RNA modification data.
    diff_gene_mod_cond_extraction : Extraction of condition-specific differential modification sites.
    """  
    DmodE_object = DmodE()
    
    # Validate input: require BOTH metadata files OR BOTH data file lists with conditions/samplenames
    has_metadata = args.metadata_modkit_file != "" and args.metadata_count_file != ""
    has_data_files = (args.data_count_files != [] and args.data_mod_bed_files != [] and 
                      args.conditions != [] and args.samplenames != [])
    
    if not has_metadata and not has_data_files:
        logger.error(
            "Invalid input combination. You must provide EITHER:\n"
            "  1) Both --metadata_count_file AND --metadata_modkit_bed_file, OR\n"
            "  2) Both --data-count-files AND --data-mod-bed-files (with --conditions and --samplenames)"            
        )
        raise ValueError(
            "Invalid input combination. You must provide EITHER:\n"
            "  1) Both --metadata_count_file AND --metadata_modkit_bed_file, OR\n"
            "  2) Both --data-count-files AND --data-mod-bed-files (with --conditions and --samplenames)"
        )
    
    if has_metadata and has_data_files:
        logger.error(
            "Conflicting input: Cannot use both metadata files and explicit data file lists. "
            "Choose one input mode."            
        )
        raise ValueError(
            "Conflicting input: Cannot use both metadata files and explicit data file lists. "
            "Choose one input mode."
        )
    
    if has_metadata:
        
        counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(metadata = args.metadata_count_file,reference_levels= args.reference_levels, output_folder = args.output_folder)
        DmodE_object.gene_mod_preprocess_data(metadata=args.metadata_modkit_file, reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        # Filter data by specified modification types if provided
        modification_codes = []
        if hasattr(args, 'modification') and args.modification:
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Warning: Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        # Continue with differential modification analysis and extraction
        diff_gene_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    elif has_data_files:
        counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(data_files = args.data_count_files, conditions = args.conditions, samplenames = args.samplenames, reference_levels= args.reference_levels, output_folder = args.output_folder)
        DmodE_object.gene_mod_preprocess_data(data_files=args.data_mod_bed_files,conditions=args.conditions,samplenames=args.samplenames,reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        # Filter data by specified modification types if provided
        modification_codes = []
        if hasattr(args, 'modification') and args.modification:
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)     

        #Continue with differential modification analysis and extraction
        diff_gene_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)
    logger.info("Metadata gene modification\n")
    for data_file, condition, sample in zip(DmodE_object.diff_gene_mod_data_files,DmodE_object.diff_gene_mod_conditions, DmodE_object.diff_gene_mod_samplenames):
        logger.info(f"\nFile: {data_file}\nCondition: {condition}\nSample: {sample}\n")
    logger.info("Metadata gene expression\n")
    for data_file,condition,sample in zip(DmodE_object.diff_gene_exp_data_files,DmodE_object.diff_gene_exp_conditions,DmodE_object.diff_gene_exp_samplenames):
        logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")
    return DmodE_object, counts_df, condition_dict


def gene_expression_and_modification_analysis(args):
    """
    Perform integrated differential gene expression and RNA modification analysis.

    This function executes the full workflow for joint differential analysis of 
    gene expression and RNA modification data across experimental conditions. It 
    initializes the analysis using `initialize_gene_expression_and_modification_analysis()`,
    generates differential modification volcano plots, extracts and annotates significant
    modification sites, performs differential gene expression testing, and integrates both
    result types into combined correlation and visualization outputs.

    The analysis supports both metadata-driven and direct file-based modes of input.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata_modkit_file : str
            Path to a tab-separated metadata file describing sample names, file paths,
            and conditions for RNA modification (modkit) data.
        metadata_count_file : str
            Path to a metadata file describing sample names, file paths,
            and conditions for expression (count) data.
        data_count_files : list of str
            List of count matrix file paths if no metadata file is provided.
        data_mod_bed_files : list of str
            List of BED files with per-site RNA modification frequencies.
        conditions : list of str
            Experimental condition labels corresponding to each sample.
        samplenames : list of str
            Sample names for input files.
        reference_levels : list of str
            Condition(s) used as baseline for both expression and modification analysis.
        chrom : str
            Chromosome name to restrict modification plots and site extraction to,
            or empty string `""` to process all chromosomes.
        gtf_file : str
            Path to a GTF file used for mapping modification sites to gene coordinates.
        output_folder : str
            Directory for saving all generated figures and result tables.
        alpha_modification : float
            Significance threshold (p-value) for identifying differentially modified positions.
        alpha_expression : float
            Significance threshold (adjusted p-value) for differential expression testing.
        minimum_coverage : int
            Minimum read coverage threshold for inclusion of modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency required for analysis.
        pvalue_correction : str
            Method for multiple testing correction of modification p-values
            (e.g., "fdr_bh", "bonferroni").

    Workflow
    --------
    1. **Initialization** — Calls 
       `initialize_gene_expression_and_modification_analysis()` 
       to preprocess count and modification data.
    2. **Modification Analysis**
       - Generates volcano plots of differential modification per condition comparison.
       - Extracts significant modification sites by condition and chromosome.
       - Annotates sites with gene identifiers using a GTF file.
    3. **Expression Analysis**
       - Performs differential expression analysis for each pairwise comparison.
       - Generates volcano, MA, and heatmap plots for expression data.
       - Creates sample-to-sample correlation plots.
    4. **Integration**
       - Loads generated per-comparison result files.
       - Merges modification, expression, and gene annotation data.
       - Generates combined scatterplots showing expression vs. modification differences.
       - Computes correlation and regression plots where possible.

    Returns
    -------
    None
        Results are written to the specified output folder, including:

        - Volcano and MA plots for both expression and modification analyses.
        - Gene-level heatmaps across conditions.
        - Annotated tables of significant modification sites.
        - Correlation scatterplots comparing gene expression and modification changes.

    Raises
    ------
    ValueError
        If neither metadata files nor explicit file paths and conditions are provided.
    FileNotFoundError
        If required differential analysis output files are missing in the output directory.

    Notes
    -----
    - Each condition pair is processed independently to produce both expression and 
      modification differential results.
    - Chromosome-specific analyses can be restricted using `--chrom <chromosome>`.
    - Integrated correlation analyses require both differential expression and 
      modification results to be successfully generated.

    See Also
    --------
    initialize_gene_expression_and_modification_analysis : Prepares inputs for this workflow.
    diff_gene_mod_cond_generate_volcano_plot_single_chr : Generates single-chromosome volcano plots.
    diff_gene_mod_cond_extract_significant_positions : Extracts significantly modified sites.
    diff_exp_analysis : Performs differential expression testing.
    expression_modification_correlation_scatterplot : Computes and plots gene-level correlations.
    gene_expression_vs_modification_differcence_scatterplot : Visualizes paired expression-modification changes.
    """
    logger.info("Preprocessing combined differential gene expression and modification analysis between conditions")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "diff_gene_expression_and_modification")
    os.makedirs(args.output_folder, exist_ok=True)
    
    DmodE_object, counts_df, condition_dict = initialize_gene_expression_and_modification_analysis(args)

    # Check if at least 2 samples per condition are provided, if not, skip UMAP and PCA Plot
    at_least_2_per_con = True
    for condition in DmodE_object.diff_gene_mod_dataframe_condition_dict.keys():
        if len(DmodE_object.diff_gene_mod_dataframe_condition_dict[condition].values()) == 1:
            raise ValueError(f"You must provide at least 2 samples per condition to run the differential gene expression analysis! Condition '{condition}' has only 1 sample.")
    
    comparison_annotations = _prepare_gene_annotation_lookup(
        DmodE_object,
        args.gtf_file,
        args.alpha_modification,
        args.output_folder,
    )

    logger.info("Volcano plot differential gene modification")
    for comparison in DmodE_object.diff_gene_mod_comparisons_dict.keys():
        if args.chrom == "":
            fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                "",
                DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                alpha=args.alpha_modification,
                annotation_df=comparison_annotations.get(comparison),
            )
            fig2 = diff_gene_mod_cond_generate_volcano_plots(
                DmodE_object.diff_gene_mod_chr_names,
                DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                alpha=args.alpha_modification,
                annotation_df=comparison_annotations.get(comparison),
            )

            if args.output_folder != "":
                os.makedirs(args.output_folder, exist_ok=True)
                os.makedirs(f"{args.output_folder}/{comparison}", exist_ok=True)
                fig1.savefig(
                    f"{args.output_folder}/{comparison}/{comparison}_all_chromosomes_in_one_volcano_plot.png", dpi=500
                )
                fig1.savefig(
                    f"{args.output_folder}/{comparison}/{comparison}_all_chromosomes_in_one_volcano_plot.svg", format="svg"
                )
                fig2.savefig(
                    f"{args.output_folder}/{comparison}/{comparison}_all_chromosomes_grid_volcano_plot.png"
                )
                fig2.savefig(
                    f"{args.output_folder}/{comparison}/{comparison}_all_chromosomes_grid_volcano_plot.svg", format="svg"
                )
        else:
            fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
                args.chrom,
                DmodE_object.diff_gene_mod_comparisons_dict[comparison],
                alpha=args.alpha_modification,
                annotation_df=comparison_annotations.get(comparison),
            )
            if args.output_folder != "":
                os.makedirs(args.output_folder, exist_ok=True)
                os.makedirs(f"{args.output_folder}/{comparison}", exist_ok=True)
                fig1.savefig(
                    f"{args.output_folder}/{comparison}/{args.chrom}_{comparison}_volcano_plot.png", dpi=500
                )
                fig1.savefig(
                    f"{args.output_folder}/{comparison}/{args.chrom}_{comparison}_volcano_plot.svg", format="svg"
                )
    logger.info("Collect significant differentially modified positions")
    for comparison in DmodE_object.diff_gene_mod_comparisons_dict.keys():

        cond_level = comparison.split("_VS_")[0]
        reference_level = comparison.split("_VS_")[1]
        if args.chrom == "":
            diff_gene_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha_modification,
                chr="",
                output_folder=args.output_folder,
            )
        else:
            diff_gene_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha_modification,
                chr=args.chrom,
                output_folder=args.output_folder,
            )
    conditions = list([str(key) for key in DmodE_object.diff_gene_exp_conditions])
    for reference in np.unique(args.reference_levels):
        for condition in np.unique(conditions):
            if condition != reference:
                logger.info(f"Comparison level: {condition} VS {reference}")
                result_df = diff_exp_analysis(counts_df = counts_df, ref_level = reference, first_level = condition, condition_dict=condition_dict, alpha = args.alpha_expression, output_folder = args.output_folder, gtf_file=args.gtf_file, operational_level="genome") 
                result_df_temp = pd.DataFrame(result_df)
                logger.info("Volcano plot differential gene expression")
                diff_exp_volcano_plot(result_df = result_df_temp, output_folder = args.output_folder, pvalue_threshold =args.alpha_expression, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "genome")
                logger.info("MA plot differential gene expression")
                diff_exp_ma_plot(result_df = result_df_temp, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "genome")
                logger.info("Heatmap differential gene expression")
                diff_exp_heatmap_pairwise(result_df = result_df_temp, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1, first_level = condition, ref_level = reference, operational_level = "genome")
                diff_exp_heatmap_all_conditions(result_df = result_df_temp, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1,operational_level = "genome")
                logger.info("Sample to sample plot differential gene expression")
                diff_exp_sample_to_sample_plot(counts_df = counts_df, output_folder = args.output_folder, operational_level = "genome")
    
    comparison_dirs = [d for d in os.listdir(args.output_folder) if os.path.isdir(os.path.join(args.output_folder, d)) and '_VS_' in d]
    if not comparison_dirs:
        logger.error(f"[ERROR] No subdirectories with gene expression / modification testing from dmode found in {args.output_folder}.")
        logger.error("Please run differential gene expression and modification analysis first!")
        return
    comparison_data = {}
    for comp_dir in comparison_dirs:
        comp_path = os.path.join(args.output_folder, comp_dir)
        mod_file = os.path.join(comp_path, f"{comp_dir}_diff_gene_mod.tsv")
        annotation_file = os.path.join(comp_path, f"{comp_dir}_diff_gene_mod_genes.tsv")
        exp_file = os.path.join(comp_path, f"{comp_dir}_genome_diff_expression_results.tsv")
        mod_df = None
        exp_df = None
        if not os.path.exists(mod_file):
            logger.error(f"[ERROR] No Modification Analysis file found in {comp_dir}")
        else:
            mod_df = pd.read_csv(mod_file, sep='\t')
        if not os.path.exists(exp_file):
            logger.error(f"[ERROR] No differential Gene Expression file found in {comp_dir}")
        else:
            exp_df = pd.read_csv(exp_file, sep='\t')
        if not os.path.exists(annotation_file):
            logger.error(f"[ERROR] No annotatet modification sites file found in {comp_dir}")
        else:
            annotation_df = pd.read_csv(annotation_file, sep='\t')    
        comparison_data = {'modification': mod_df, 'expression': exp_df, 'annotation': annotation_df}
        if mod_df is not None and exp_df is not None:
            logger.info(f"Eruption plot")
            gene_expression_vs_modification_differcence_scatterplot(comparison_data, args.output_folder, operational_level="genome")
            try:
                logger.info("Eruption plot with expression/modification correlations")
                expression_modification_correlation_scatterplot(comparison_data, args.output_folder, operational_level="genome")
            except ValueError:
                logger.error("Regression scatterplot could not be calculated from the data.")
                pass
            try:
                logger.info("Top 50 genes with most number of dignificant differential modification sites")
                plot_top50_genes_with_most_number_of_significant_modifications(comparison_data, args.output_folder, operational_level="genome")
            except ValueError:
                logger.error("Top 50 genes plot could not be calculated from the data.")
    # Generate HTML report
    _generate_report_if_possible(args.output_folder)
    
    
def initialize_transcript_expression_and_modification_analysis(args):
    """
    Initialize combined transcript expression and modification analysis.

    This function prepares both transcript expression and RNA modification data for
    integrated differential analysis using the `DmodE` framework. It preprocesses
    read count and modification datasets, extracts differential modification sites,
    and returns data structures required for downstream correlation or joint
    expression–modification studies.

    Input data can be provided either through metadata files (for both expression
    and modification datasets) or as explicit lists of file paths and condition
    labels.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata_modkit_file : str
            Path to a metadata file describing samples and conditions for
            RNA modification (modkit) data.
        metadata_count_file : str
            Path to a metadata file describing samples and conditions for
            gene expression (count) data.
        data_count_files : list of str
            List of expression count files used if no metadata file is provided.
        data_mod_bed_files : list of str
            List of BED files containing per-site RNA modification data.
        conditions : list of str
            Experimental condition labels corresponding to the input samples.
        samplenames : list of str
            Sample names corresponding to the data files.
        reference_levels : list of str
            Reference condition(s) used as baseline(s) for both expression and
            modification analysis.
        minimum_coverage : int
            Minimum read coverage required for a site to be included in
            modification analysis.
        minimum_mod_frequency : float
            Minimum modification frequency threshold for inclusion in analysis.
        pvalue_correction : str
            Method used for multiple testing correction of modification site
            p-values (e.g., `"fdr_bh"`, `"bonferroni"`).
        output_folder : str
            Directory where intermediate and output files will be saved.

    Returns
    -------
    tuple
        A tuple containing:

        DmodE_object : DmodE
            The initialized analysis object containing both expression and
            modification data.
        counts_df : pandas.DataFrame
            A gene-level expression matrix for all samples.
        condition_dict : dict
            A dictionary mapping sample names to experimental conditions.

    Raises
    ------
    ValueError
        If neither metadata files nor explicit file and condition lists are provided.

    Workflow
    --------
    1. Initializes a `DmodE` object for managing both data types.
    2. Preprocesses transcript expression data using `diff_transcr_exp_preprocess_data`.
    3. Preprocesses modification data using `diff_gene_mod_preprocess_data`.
    4. Extracts differential modification sites via `diff_gene_mod_cond_extraction`.
    5. Returns the processed data structures for downstream analyses.

    Notes
    -----
    - If both metadata-based and direct file input modes are provided, the metadata
      files take precedence.
    - The function prepares the data for joint analyses, such as correlating
      expression changes with modification dynamics at the gene level.
    - Differential modification extraction is performed automatically during
      initialization to ensure the availability of processed results for subsequent
      analyses.

    See Also
    --------
    DmodE.diff_gene_exp_preprocess_data : Preprocessing routine for gene expression data.
    DmodE.diff_gene_mod_preprocess_data : Preprocessing routine for RNA modification data.
    diff_gene_mod_cond_extraction : Extraction of condition-specific differential modification sites.
    """    
    DmodE_object = DmodE()
    
    # Validate input: require BOTH metadata files OR BOTH data file lists with conditions/samplenames
    has_metadata = args.metadata_modkit_file != "" and args.metadata_count_file != ""
    has_data_files = (args.data_count_files != [] and args.data_mod_bed_files != [] and 
                      args.conditions != [] and args.samplenames != [])
    
    if not has_metadata and not has_data_files:
        raise ValueError(
            "Invalid input combination. You must provide EITHER:\n"
            "  1) Both --metadata_count_file AND --metadata_modkit_bed_file, OR\n"
            "  2) Both --data-count-files AND --data-mod-bed-files (with --conditions and --samplenames)"
        )
    
    if has_metadata and has_data_files:
        raise ValueError(
            "Conflicting input: Cannot use both metadata files and explicit data file lists. "
            "Choose one input mode."
        )
    
    if has_metadata:
        counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(metadata = args.metadata_count_file,reference_levels= args.reference_levels, output_folder = args.output_folder)
        DmodE_object.transcript_mod_preprocess_data(metadata=args.metadata_modkit_file, reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        # Continue with differential modification analysis and extraction
        diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    elif has_data_files:
        counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(data_files = args.data_count_files, conditions = args.conditions, samplenames = args.samplenames, reference_levels= args.reference_levels, output_folder = args.output_folder)
        DmodE_object.transcript_mod_preprocess_data(data_files=args.data_mod_bed_files,conditions=args.conditions,samplenames=args.samplenames,reference_levels=args.reference_levels,coverage_filter=args.minimum_coverage, mod_freq_filter=args.minimum_mod_frequency)

        # Filter data by specified modification types if provided
        if hasattr(args, 'modification') and args.modification:
            modification_codes = []
            for mod_type in args.modification:
                if mod_type in BASE_MODIFICATION_TYPE_DICT:
                    modification_codes.append(BASE_MODIFICATION_TYPE_DICT[mod_type])
                else:
                    logger.warning(f"Unknown modification type '{mod_type}'. Available types: {list(BASE_MODIFICATION_TYPE_DICT.keys())}")
            if modification_codes:
                filter_dmode_data_by_modification(DmodE_object, modification_codes)

        # Continue with differential modification analysis and extraction
        diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object,output_folder=args.output_folder,correction_method=args.pvalue_correction)

    logger.info("Metadata for transcript modification\n")
    for data_file,condition,sample in zip(DmodE_object.diff_transcript_mod_data_files,DmodE_object.diff_transcript_mod_conditions,DmodE_object.diff_transcript_mod_samplenames):
        logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")
        
    logger.info("Metadata for transcript expression\n")
    for data_file,condition,sample in zip(DmodE_object.diff_transcript_exp_data_files,DmodE_object.diff_transcript_exp_conditions, DmodE_object.diff_transcript_exp_samplenames):
        logger.info(f"File: {data_file}\\nCondition: {condition}\\nSample: {sample}\\n")
    

    return DmodE_object, counts_df, condition_dict



def transcript_expression_and_modification_analysis(args):
    """
    Perform integrated differential transcript expression and RNA modification analysis.

    This function executes the full workflow for joint differential analysis of 
    transcript expression and RNA modification data across experimental conditions. It 
    initializes the analysis using `initialize_transcript_expression_and_modification_analysis()`,
    generates differential modification volcano plots, extracts and annotates significant
    modification sites, performs differential gene expression testing, and integrates both
    result types into combined correlation and visualization outputs.

    The analysis supports both metadata-driven and direct file-based modes of input.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments or equivalent namespace containing:

        metadata_modkit_file : str
            Path to a tab-separated metadata file describing sample names, file paths,
            and conditions for RNA modification (modkit) data.
        metadata_count_file : str
            Path to a metadata file describing sample names, file paths,
            and conditions for transcript expression (count) data.
        data_count_files : list of str
            List of transcript count matrix file paths if no metadata file is provided.
        data_mod_bed_files : list of str
            List of BED files with per-site RNA modification frequencies.
        conditions : list of str
            Experimental condition labels corresponding to each sample.
        samplenames : list of str
            Sample names for input files.
        reference_levels : list of str
            Condition(s) used as baseline for both expression and modification analysis.
        chrom : str
            Chromosome name to restrict modification plots and site extraction to,
            or empty string `""` to process all chromosomes.
        gtf_file : str
            Path to a GTF file used for mapping modification sites to gene coordinates.
        output_folder : str
            Directory for saving all generated figures and result tables.
        alpha_modification : float
            Significance threshold (p-value) for identifying differentially modified positions.
        alpha_expression : float
            Significance threshold (adjusted p-value) for differential expression testing.
        minimum_coverage : int
            Minimum read coverage threshold for inclusion of modification sites.
        minimum_mod_frequency : float
            Minimum modification frequency required for analysis.
        pvalue_correction : str
            Method for multiple testing correction of modification p-values
            (e.g., "fdr_bh", "bonferroni").

    Workflow
    --------
    1. **Initialization** — Calls 
       `initialize_transcript_expression_and_modification_analysis()` 
       to preprocess count and modification data.
    2. **Modification Analysis**
       - Generates volcano plots of differential modification per condition comparison.
       - Extracts significant modification sites by condition and chromosome.
       - Annotates sites with transcript identifiers using a GTF file.
    3. **Expression Analysis**
       - Performs differential transcript expression analysis for each pairwise comparison.
       - Generates volcano, MA, and heatmap plots for transcript expression data.
       - Creates sample-to-sample correlation plots.
    4. **Integration**
       - Loads generated per-comparison result files.
       - Merges modification, expression, and gene annotation data.
       - Generates combined scatterplots showing expression vs. modification differences.
       - Computes correlation and regression plots where possible.

    Returns
    -------
    None
        Results are written to the specified output folder, including:

        - Volcano and MA plots for both expression and modification analyses.
        - transcript-level heatmaps across conditions.
        - Annotated tables of significant modification sites.
        - Correlation scatterplots comparing gene expression and modification changes.

    Raises
    ------
    ValueError
        If neither metadata files nor explicit file paths and conditions are provided.
    FileNotFoundError
        If required differential analysis output files are missing in the output directory.

    Notes
    -----
    - Each condition pair is processed independently to produce both expression and 
      modification differential results.
    - Chromosome-specific analyses can be restricted using `--chrom <chromosome>`.
    - Integrated correlation analyses require both differential expression and 
      modification results to be successfully generated.

    See Also
    --------
    initialize_transcript_expression_and_modification_analysis : Prepares inputs for this workflow.
    diff_transcript_mod_cond_extract_significant_positions : Extracts significantly modified sites.
    diff_exp_analysis : Performs differential expression testing.
    expression_modification_correlation_scatterplot : Computes and plots gene-level correlations.
    gene_expression_vs_modification_differcence_scatterplot : Visualizes paired expression-modification changes.
    """
    logger.info("Preprocessing combined differential transcript expression and modification analysis between conditions")
    # Create function-specific output folder
    args.output_folder = os.path.join(args.output_folder, "diff_transcript_expression_and_modification")
    os.makedirs(args.output_folder, exist_ok=True)
    
    DmodE_object, counts_df, condition_dict = initialize_transcript_expression_and_modification_analysis(args)
    
        # Check if at least 2 samples per condition are provided, if not, skip UMAP and PCA Plot
    at_least_2_per_con = True
    for condition in DmodE_object.diff_transcript_mod_dataframe_condition_dict.keys():
        if len(DmodE_object.diff_transcript_mod_dataframe_condition_dict[condition].values()) == 1:
            raise ValueError(f"You must provide at least 2 samples per condition to run the differential transcript expression analysis! Condition '{condition}' has only 1 sample.")

    comparison_annotations = _prepare_transcript_annotation_lookup(
        DmodE_object,
        args.gtf_file,
        args.alpha_modification,
        args.output_folder,
    )

    logger.info("Volcano plot differential transcript modification")
    for comparison in DmodE_object.diff_transcript_mod_comparisons:
        comparison_key = f"{comparison[0]}_VS_{comparison[1]}"
        if args.chrom == "":
            fig1 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
                "",
                DmodE_object.diff_transcript_mod_comparisons_dict[comparison_key],
                alpha=args.alpha_modification,
                annotation_df=comparison_annotations.get(comparison_key),
            )
            if args.output_folder != "":
                os.makedirs(args.output_folder, exist_ok=True)
                os.makedirs(f"{args.output_folder}/{comparison_key}", exist_ok=True)
                fig1.savefig(
                    f"{args.output_folder}/{comparison_key}/{comparison_key}_all_transcripts_in_one_volcano_plot.png", dpi=500
                )
                fig1.savefig(
                    f"{args.output_folder}/{comparison_key}/{comparison_key}_all_transcripts_in_one_volcano_plot.svg", format="svg"
                )
    logger.info("Collect significant differentially modified positions")
    for comparison in DmodE_object.diff_transcript_mod_comparisons:
        cond_level = comparison[0]
        reference_level = comparison[1]
        if args.chrom == "":
            diff_transcript_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha_modification,
                chr="",
                output_folder=args.output_folder,
            )
        else:
            diff_transcript_mod_cond_extract_significant_positions(
                dmode_obj=DmodE_object,
                ref_level=reference_level,
                cond_level=cond_level,
                alpha=args.alpha_modification,
                chr=args.chrom,
                output_folder=args.output_folder,
            )
    # RUN PLOTTING FUNCTIONS FOR TRANSCRPT EXPRESSION ANALYSIS
    import numpy as np
    for reference in np.unique(args.reference_levels):
        for condition in np.unique(DmodE_object.diff_transcript_exp_conditions):
            if condition != reference:
                logger.info(f"Comparison level: {condition} VS {reference}")
                result_df = diff_exp_analysis(counts_df = counts_df, ref_level = reference, first_level = condition, condition_dict=condition_dict, alpha = args.alpha_expression, output_folder = args.output_folder, gtf_file=args.gtf_file, operational_level="transcriptome") 
                logger.info("Volcano plot differential gene expression")
                diff_exp_volcano_plot(result_df = result_df, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "transcriptome")
                diff_exp_ma_plot(result_df = result_df, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1, ref_level = reference, first_level = condition, operational_level = "transcriptome")
                logger.info("Heatmap differential gene expression")
                diff_exp_heatmap_pairwise(result_df = result_df, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1, first_level = condition, ref_level = reference, operational_level = "transcriptome")
                diff_exp_heatmap_all_conditions(result_df = result_df, counts_df = counts_df, condition_dict = condition_dict, output_folder = args.output_folder, pvalue_threshold = args.alpha_expression, lfc_threshold = 1,operational_level = "transcriptome")
                logger.info("Sample to sample plot differential gene expression")
                diff_exp_sample_to_sample_plot(counts_df = counts_df, output_folder = args.output_folder, operational_level = "transcriptome")

    # PROCEED WITH COUPLED MODIFICATION AND EXPRESSION ANALYSIS
    comparison_dirs = [d for d in os.listdir(args.output_folder) if os.path.isdir(os.path.join(args.output_folder, d)) and '_VS_' in d]
    if not comparison_dirs:
        logger.error(f"[ERROR] No subdirectories with gene expression / modification testing from dmode found in {args.output_folder}.")
        logger.error("Please run differential gene expression and modification analysis first!")
        return
    comparison_data = {}
    for comp_dir in comparison_dirs:
        comp_path = os.path.join(args.output_folder, comp_dir)
        mod_file = os.path.join(comp_path, f"{comp_dir}_diff_transcript_mod.tsv")
        annotation_file = os.path.join(comp_path, f"{comp_dir}_diff_transcript_mod_transcripts.tsv")
        exp_file = os.path.join(comp_path, f"{comp_dir}_transcriptome_diff_expression_results.tsv")
        mod_df = None
        exp_df = None
        if not os.path.exists(mod_file):
            logger.error(f"[ERROR] No Modification Analysis file found in {comp_dir}")
        else:
            mod_df = pd.read_csv(mod_file, sep='\t')
        if not os.path.exists(exp_file):
            logger.error(f"[ERROR] No differential Transcriptome Expression file found in {comp_dir}")
        else:
            exp_df = pd.read_csv(exp_file, sep='\t')
        if not os.path.exists(annotation_file):
            logger.error(f"[ERROR] No annotated modification sites file found in {comp_dir}")
        else:
            annotation_df = pd.read_csv(annotation_file, sep='\t')    
        comparison_data = {'modification': mod_df, 'expression': exp_df, 'annotation': annotation_df}
        if mod_df is not None and exp_df is not None:
            logger.info(f"Eruption plot")
            gene_expression_vs_modification_differcence_scatterplot(comparison_data, args.output_folder, operational_level="transcriptome")
            try:
                logger.info("Eruption plot with expression/modification correlations")
                expression_modification_correlation_scatterplot(comparison_data, args.output_folder, operational_level="transcriptome")
            except ValueError:
                logger.error("Regression scatterplot could not be calculated from the data.")
                pass
            try:

                logger.info("Top 50 transcripts with most number of dignificant differential modification sites")
                plot_top50_genes_with_most_number_of_significant_modifications(comparison_data, args.output_folder, operational_level="transcriptome")
            except ValueError:
                logger.error("Top 50 transcripts plot could not be calculated from the data.")
                pass
        # Generate HTML report
    _generate_report_if_possible(args.output_folder)


#def QC_report(args):
    # Generate QC report including number of reads barplot
#    generate_numberofreads_barplot(args.qc_metadata_file, args.output_folder)


def main():
    parser = argparse.ArgumentParser(
        prog="dmode",
        description="DModE: A command line tool for differential modification site analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    ## Basic statistics of modification data (Gene level) ##
    gene_basic_statistics_parser = subparsers.add_parser("gene_basic_statistics",
                                                    help="Compute basic statistics of samples at gene level",
                                                    description="""
                                                    Compute and visualize fundamental statistics of gene-level RNA modifications.

                                                    This function performs exploratory analyses on differential gene modification data.
                                                    It identifies all modification types across samples, generates PCA, UMAP,
                                                    barplots, violin plots, and genome-wide modification distributions. Optionally,
                                                    it compares detected sites to known RNA modification databases.

                                                    Input can be provided via metadata files or direct file lists. Results are
                                                    saved to the specified output folder.
                                                    """,
                                                    formatter_class=argparse.RawTextHelpFormatter
                                                )
    
    # Required arguments
    required_gene_basic_stats = gene_basic_statistics_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files
    input_group = required_gene_basic_stats.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata","-m",dest="metadata",type= str,
        help="Path to metadatafile carrying the columns File and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files","-d",dest="data_files",nargs="+",type=str,
        help="Provide every single data file separately here (requires --conditions and --samplenames).",
        default=[]
    )
    required_gene_basic_stats.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_gene_basic_stats.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_gene_basic_stats = gene_basic_statistics_parser.add_argument_group('optional arguments')
    optional_gene_basic_stats.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_gene_basic_stats.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_gene_basic_stats.add_argument(
        "--chromosome","-x",dest="chrom",type=str,
        help="Provide a chromosome of interest occuring in your dataset.",
        default=""
    )
    optional_gene_basic_stats.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_gene_basic_stats.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_gene_basic_stats.add_argument(
        "--rmbase_database_file", "-b", dest="rmbase_database_file",nargs="+",type=str, 
        help="Path to RMBase database file in bed format to compare your modification sites to known modification sites.",
        default=""
    )
    optional_gene_basic_stats.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    gene_basic_statistics_parser.set_defaults(func=gene_basic_statistics)



    ## Metagene Plot on Gene level##
    metagene_plot_parser = subparsers.add_parser("metagene_plot",
                                                    help="Generate Metagene Plot on Gene level",
                                                    description="""
                                                    Compute and visualize a Metagene Plot of gene-level RNA modifications.

                                                    This function generates a Metagene Plot on Genebased Modkit bedfiles, a gtf
                                                    file is required to run this analysis.

                                                    Input can be provided via metadata files or direct file lists. Results are
                                                    saved to the specified output folder.
                                                    """,
                                                    formatter_class=argparse.RawTextHelpFormatter
                                                )
    
    # Required arguments
    required_metagene_plot = metagene_plot_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files
    input_group = required_metagene_plot.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata","-m",dest="metadata",type= str,
        help="Path to metadatafile carrying the columns File and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files","-d",dest="data_files",nargs="+",type=str,
        help="Provide every single data file separately here (requires --conditions and --samplenames).",
        default=[]
    )
    required_metagene_plot.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_metagene_plot.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_metagene_plot = metagene_plot_parser.add_argument_group('optional arguments')
    optional_metagene_plot.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_metagene_plot.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_metagene_plot.add_argument(
        "--p_value_correction","-v",dest="pvalue_correction",type=str,
        help="Select a pvalue correction method. (SLIM (Significance Level-based Iterative Method) or bh (Benjamini-Hochberg))",
        default="SLIM",
    )
    optional_metagene_plot.add_argument(
        "--metagene_min_coverage", dest="metagene_min_coverage",nargs="+",type=str, 
        help="Minimum coverage per site to be included in metagene plot generation",
        default=3
    )
    optional_metagene_plot.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    ) 
    optional_metagene_plot.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    
    optional_metagene_plot.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    
    optional_metagene_plot.add_argument(
        "--gene_id","-x",dest="gene_id",type=str,
        help="Provide a gene id of interest for gene specific metagene plot",
        default=""
    )
    
    optional_metagene_plot.set_defaults(func=metagene_plot)

    ### Argument parser for QC report including number of reads barplot ###
    # qc_report_parser = subparsers.add_parser(
    #     "qc_report",
    #     help="Generate QC report including number of reads barplot",
    #     description="""
    #     Generate a QC report including a barplot of the number of reads per sample.

    #     This function processes a metadata file containing sample information and
    #     generates a QC report with visualizations, including a barplot showing
    #     the number of reads for each sample. The report is saved to the specified
    #     output folder.
    #     """,
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    # qc_report_parser.add_argument(
    #     "--qc_metadata_file", "-q", dest="qc_metadata_file", type=str,
    #     help="Path to QC metadata file.",
    #     required=True
    # )
    # qc_report_parser.add_argument(
    #     "--output_folder", "-o", dest="output_folder", type=str,
    #     help="Path to output folder",
    #     required=True
    # )
    # qc_report_parser.set_defaults(func=QC_report)


    ### Argument parser for basic statistics at transcript level ###
    transcript_basic_statistics_parser = subparsers.add_parser(
        "transcript_basic_statistics",
        help="Perform fundamental analysis on transcript level to get an overview of your data."
    )
    
    # Required arguments - mutually exclusive metadata or data-files
    required_transcript_basic_stats = transcript_basic_statistics_parser.add_argument_group('required arguments')
    metadata_or_datafiles_transcript_stats = required_transcript_basic_stats.add_mutually_exclusive_group(required=True)
    metadata_or_datafiles_transcript_stats.add_argument(
        "--metadata", "-m", dest="metadata", type=str,
        help="Path to metadata file (mutually exclusive with --data-files).",
        default=""
    )
    metadata_or_datafiles_transcript_stats.add_argument(
        "--data-files", "-d", dest="data_files", nargs="+", type=str,
        help="Provide every single data file separately here (requires --conditions and --samplenames).",
        default=[]
    )
    required_transcript_basic_stats.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_transcript_basic_stats.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_transcript_basic_stats = transcript_basic_statistics_parser.add_argument_group('optional arguments')
    optional_transcript_basic_stats.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_transcript_basic_stats.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_transcript_basic_stats.add_argument(
        "--chromosome","-x",dest="chrom",type=str,
        help="Provide a chromosome of interest occuring in your dataset.",
        default=""
    )
    optional_transcript_basic_stats.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_transcript_basic_stats.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_transcript_basic_stats.add_argument(
        "--rmbase_database_file", "-b", dest="rmbase_database_file",nargs="+",type=str, 
        help="Path to RMBase database file in bed format to compare your modification sites to known modification sites.",
        default=""
    )
    optional_transcript_basic_stats.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    transcript_basic_statistics_parser.set_defaults(func=transcript_basic_statistics)

    #### Parser for differential genome based modification analysis between conditions ####
    diff_gene_mod_cond_comparison_parser = subparsers.add_parser("diff_gene_modification", 
                                                                help="Determines differential modification sites between conditions and generates volcano plots.",
                                                                description="""
                                                                Perform differential gene modification analysis across conditions.

                                                                This function identifies significantly modified positions at the gene level
                                                                between experimental conditions. It generates volcano plots (per chromosome
                                                                or genome-wide), extracts significant modification sites, and annotates them
                                                                with gene IDs. Results are saved to the specified output folder for downstream
                                                                interpretation.

                                                                Input can be provided via a metadata file or direct BED files with modification data.
                                                                """,
                                                                formatter_class=argparse.RawTextHelpFormatter)
    
    # Required arguments
    required_gene_mod = diff_gene_mod_cond_comparison_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files. For the latter, conditions and samplenames
    # must be provided (this is validated at runtime in the handler functions).
    input_group = required_gene_mod.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata","-m",dest="metadata",type= str,
        help="Path to metadatafile carrying the columns File and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files","-d",dest="data_files",nargs="+",type=str,
        help="Provide every single data file separately here (requires --conditions and --samplenames).",
        default=[]
    )
    required_gene_mod.add_argument(
        "--reference_levels","-r",dest="reference_levels",nargs="+",type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_gene_mod.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_gene_mod.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_gene_mod = diff_gene_mod_cond_comparison_parser.add_argument_group('optional arguments')
    optional_gene_mod.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_gene_mod.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_gene_mod.add_argument(
        "--alpha", "-a", dest="alpha",type=float, help="Provide a threshold for statistical tests",
        default=0.01
    )
    optional_gene_mod.add_argument(
        "--chromosome","-x",dest="chrom",type=str,
        help="Provide a chromosome of interest occuring in your dataset.",
        default=""
    )
    optional_gene_mod.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_gene_mod.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_gene_mod.add_argument(
        "--p_value_correction","-v",dest="pvalue_correction",type=str,
        help="Select a pvalue correction method. (SLIM (Significance Level-based Iterative Method) or bh (Benjamini-Hochberg))",
        default="SLIM",
    )
    optional_gene_mod.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    diff_gene_mod_cond_comparison_parser.set_defaults(func=differential_gene_modification_condition_comparison)


    ####Parser for differential transcriptome based modification analysis between conditions ####
    diff_transcript_mod_cond_comparison_parser = subparsers.add_parser("diff_transcript_modification", 
                                                                    help="Determines differential modification sites between conditions and generates volcano plots.",
                                                                    description="""
                                                                    Perform differential transcript modification analysis across conditions.

                                                                    This function identifies significantly modified positions at the transcript level
                                                                    between experimental conditions. It generates volcano plots (all transcripts
                                                                    together or per chromosome), extracts significant modification sites, and
                                                                    annotates them with transcript IDs. Results are saved to the specified output
                                                                    folder for downstream interpretation.  
                                                                    Input can be provided via a metadata file or direct BED files with modification data.
                                                                    """,
                                                                    formatter_class=argparse.RawTextHelpFormatter
                                                                    )
    
    # Required arguments
    required_transcript_mod = diff_transcript_mod_cond_comparison_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files. For the latter, conditions and samplenames
    # must be provided (this is validated at runtime in the handler functions).
    input_group = required_transcript_mod.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata","-m",dest="metadata",type= str,
        help="Path to metadatafile carrying the columns File and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files","-d",dest="data_files",nargs="+",type=str,
        help="Provide every single data file separately here (requires --conditions and --samplenames).",
        default=[]
    )
    required_transcript_mod.add_argument(
        "--reference_levels","-r",dest="reference_levels",nargs="+",type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_transcript_mod.add_argument(
        "--gtf-file", "-g", dest="gtf_file", type=str,
        help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_transcript_mod.add_argument(
        "--output_folder", "-o", dest="output_folder", type=str, 
        help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_transcript_mod = diff_transcript_mod_cond_comparison_parser.add_argument_group('optional arguments')
    optional_transcript_mod.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_transcript_mod.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_transcript_mod.add_argument(
        "--alpha", "-a", dest="alpha", type=float, 
        help="Provide a threshold for statistical tests",
        default=0.01
    )
    optional_transcript_mod.add_argument(
        "--gene_of_interest", "-x", dest="chrom", type=str,
        help="Provide a gene of interest occuring in your dataset.",
        default=""
    )
    optional_transcript_mod.add_argument(
        "--coverage_filter", "-f", dest="minimum_coverage", type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_transcript_mod.add_argument(
        "--frequency_filter", "-q", dest="minimum_mod_frequency", type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_transcript_mod.add_argument(
        "--p_value_correction","-v",dest="pvalue_correction",type=str,
        help="Select a pvalue correction method. (SLIM (Significance Level-based Iterative Method) or bh (Benjamini-Hochberg))",
        default="SLIM",
    )
    optional_transcript_mod.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    diff_transcript_mod_cond_comparison_parser.set_defaults(func=differential_transcript_modification_condition_comparison)



    ####Parser for differential gene expression analysis between conditions ####
    diff_gene_exp_cond_comparison_parser = subparsers.add_parser("diff_gene_expression",
                                                                help="Pydeseq2 based differential gene expression analysis between conditions and generates volcano plots, heatmaps and sample to sample plots.",
                                                                description="""
                                                                Perform differential gene expression analysis across conditions.

                                                                This function computes gene-level differential expression using input count
                                                                matrices or metadata files. It performs statistical testing for all pairwise
                                                                comparisons of conditions, generates volcano and MA plots, pairwise and global
                                                                heatmaps, and sample-to-sample correlation plots. Results are saved to the
                                                                specified output folder for downstream interpretation.

                                                                Input can be provided via metadata files or direct count matrices.
                                                                """,
                                                                formatter_class=argparse.RawTextHelpFormatter)
    
    # Required arguments
    required_gene_exp = diff_gene_exp_cond_comparison_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files. For the latter, conditions and samplenames
    # must be provided (this is validated at runtime in the handler functions).
    input_group = required_gene_exp.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata", "-m", dest="metadata", type= str,
        help="Path to metadatafile carrying the columns File, Sample and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files", "-d", dest="data_files", nargs="+", type=str,
        help="Provide count matrix files here (requires --conditions and --samplenames).",
        default=[]
    )
    required_gene_exp.add_argument(
        "--reference_level", "-r", dest="reference_levels", nargs="+", type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_gene_exp.add_argument(
        "--gtf-file", "-g", dest="gtf_file", type=str,
        help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_gene_exp.add_argument(
        "--output_folder", "-o", dest="output_folder", type=str, 
        help="Path to output folder", 
        required=True
    )
    
    # Optional arguments
    optional_gene_exp = diff_gene_exp_cond_comparison_parser.add_argument_group('optional arguments')
    optional_gene_exp.add_argument(
        "--conditions", "-c", dest="conditions", nargs="+", type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_gene_exp.add_argument(
        "--samplenames", "-s", dest="samplenames", nargs="+", type=str,
        help="Samplenames for the provided data files.",
        default=[]
    )
    optional_gene_exp.add_argument(
        "--alpha", "-a", dest="alpha", type=float, 
        help="Provide a threshold for statistical tests",
        default=0.01
    )
    diff_gene_exp_cond_comparison_parser.set_defaults(func=differential_gene_expression_analysis)


    #### Differential transcript expression analysis between conditions ####
    diff_transcript_exp_cond_comparison_parser = subparsers.add_parser("diff_transcript_expression",
                                                                    help="Pydeseq2 based differential transcript expression analysis between conditions and generates volcano plots, heatmaps and sample to sample plots.",
                                                                    description="""
                                                                    Perform differential transcript expression analysis across conditions.

                                                                    This function computes transcript-level differential expression using input
                                                                    count matrices or metadata files. It performs statistical testing for all
                                                                    pairwise comparisons of conditions, generates volcano and MA plots, pairwise
                                                                    and global heatmaps, and sample-to-sample correlation plots. Results are saved
                                                                    to the specified output folder for downstream interpretation.

                                                                    Input can be provided via metadata files or direct count matrices.
                                                                    """,
                                                                    formatter_class=argparse.RawTextHelpFormatter
                                                                    )

    # Required arguments
    required_transcript_exp = diff_transcript_exp_cond_comparison_parser.add_argument_group('required arguments')
    # Input: require either a metadata file OR a list of data files. For the latter, conditions and samplenames
    # must be provided (this is validated at runtime in the handler functions).
    input_group = required_transcript_exp.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata","-m",dest="metadata",type= str,
        help="Path to metadatafile carrying the columns File, Sample and Condition",
        default=""
    )
    input_group.add_argument(
        "--data-files", "-d", dest="data_files", nargs="+", type=str,
        help="Provide count matrix files here (requires --conditions and --samplenames).",
        default=[]
    )
    required_transcript_exp.add_argument(
        "--reference_level", "-r", dest="reference_levels", nargs="+", type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_transcript_exp.add_argument(
        "--gtf-file", "-g", dest="gtf_file", type=str,
        help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_transcript_exp.add_argument(
        "--output_folder", "-o", dest="output_folder", type=str, 
        help="Path to output folder", 
        required=True
    )
    
    # Optional arguments
    optional_transcript_exp = diff_transcript_exp_cond_comparison_parser.add_argument_group('optional arguments')
    optional_transcript_exp.add_argument(
        "--conditions", "-c", dest="conditions", nargs="+", type=str,
        help="Condition labels corresponding to the provided --data-files (required when using --data-files).",
        default=[]
    )
    optional_transcript_exp.add_argument(
        "--samplenames", "-s", dest="samplenames", nargs="+", type=str,
        help="Samplenames for the provided data files.",
        default=[]
    )
    optional_transcript_exp.add_argument(
        "--alpha", "-a", dest="alpha", type=float, help="Provide a threshold for statistical tests",
        default=0.01
    )
    diff_transcript_exp_cond_comparison_parser.set_defaults(func=differential_transcript_expression_analysis)

    
    ## Expression and modification analysis (Genome based)##
    gene_expression_and_modification_analysis_parser = subparsers.add_parser("diff_gene_expression_and_modification", 
                                                                        help="Combined differential gene expression and modification analysis between conditions",
                                                                        description="""Perform integrated differential gene expression and RNA modification analysis.

                                                                        This function executes the full workflow for joint analysis of gene expression
                                                                        and RNA modification across experimental conditions. It generates differential
                                                                        expression and modification results, plots (volcano, MA, heatmaps), and
                                                                        correlation analyses between gene expression changes and modification dynamics.

                                                                        Input can be provided via metadata files or explicit file lists. Results are
                                                                        written to the specified output folder.
                                                                        """,
                                                                        formatter_class=argparse.RawTextHelpFormatter)

    # Required arguments
    required_gene_expr_mod = gene_expression_and_modification_analysis_parser.add_argument_group('required arguments')
    # Input mode 1: Both metadata files (count + modkit)
    required_gene_expr_mod.add_argument(
        "--metadata_count_file","-m",dest="metadata_count_file",type= str,
        help="Path to metadatafile for FeatureCounts tables (columns: File, Sample, Condition). Requires --metadata_modkit_bed_file.",
        default=""
    )
    required_gene_expr_mod.add_argument(
        "--metadata_modkit_bed_file","-n",dest="metadata_modkit_file",type= str,
        help="Path to metadatafile for Modkit bed files (columns: File, Sample, Condition). Requires --metadata_count_file.",
        default=""
    )
    
    # Input mode 2: Both data file lists (count + modkit) with conditions/samplenames
    required_gene_expr_mod.add_argument(
        "--data-count-files","-d",dest="data_count_files",nargs="+",type=str,
        help="List of count files. Requires --data-mod-bed-files, --conditions, and --samplenames.",
        default=[]
    )
    required_gene_expr_mod.add_argument(
        "--data-mod-bed-files","-D",dest="data_mod_bed_files",nargs="+",type=str,
        help="List of modkit bed files. Requires --data-count-files, --conditions, and --samplenames.",
        default=[]
    )
    required_gene_expr_mod.add_argument(
        "--reference_levels","-r",dest="reference_levels",nargs="+",type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_gene_expr_mod.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_gene_expr_mod.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_gene_expr_mod = gene_expression_and_modification_analysis_parser.add_argument_group('optional arguments')
    optional_gene_expr_mod.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels for each sample (required when using --data-count-files and --data-mod-bed-files).",
        default=[]
    )
    optional_gene_expr_mod.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names for each file (required when using --data-count-files and --data-mod-bed-files).",
        default=[]
    )
    optional_gene_expr_mod.add_argument(
        "--alpha_modification", "-a", dest="alpha_modification",type=float, help="Provide a threshold for statistical tests on differential modification",
        default=0.01
    )
    optional_gene_expr_mod.add_argument(
        "--alpha_expression", "-A", dest="alpha_expression",type=float, help="Provide a threshold for statistical tests on differential expression",
        default=0.01
    )
    optional_gene_expr_mod.add_argument(
        "--chromosome","-x",dest="chrom",type=str,
        help="Provide a chromosome of interest occuring in your dataset.",
        default=""
    )
    optional_gene_expr_mod.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_gene_expr_mod.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_gene_expr_mod.add_argument(
        "--p_value_correction","-v",dest="pvalue_correction",type=str,
        help="Select a pvalue correction method for modification detection. (SLIM (Significance Level-based Iterative Method) or bh (Benjamini-Hochberg))",
        default="SLIM",
    )
    optional_gene_expr_mod.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    gene_expression_and_modification_analysis_parser.set_defaults(func=gene_expression_and_modification_analysis)


    ## Expression and modification analysis (Transcriptome based)##
    transcriptome_expression_and_modification_analysis_parser = subparsers.add_parser("diff_transcript_expression_and_modification", 
                                                                        help="Combined differential transcript expression and modification analysis between conditions",
                                                                        description="""Perform integrated differential transcript expression and RNA modification analysis.

                                                                        This function executes the full workflow for joint analysis of transcript expression
                                                                        and RNA modification across experimental conditions. It generates differential
                                                                        expression and modification results, plots (volcano, MA, heatmaps), and
                                                                        correlation analyses between gene expression changes and modification dynamics.

                                                                        Input can be provided via metadata files or explicit file lists. Results are
                                                                        written to the specified output folder.
                                                                        """,
                                                                        formatter_class=argparse.RawTextHelpFormatter)

    # Required arguments
    required_transcript_expr_mod = transcriptome_expression_and_modification_analysis_parser.add_argument_group('required arguments')
    # Input mode 1: Both metadata files (count + modkit)
    required_transcript_expr_mod.add_argument(
        "--metadata_count_file","-m",dest="metadata_count_file",type= str,
        help="Path to metadatafile for count tables (columns: File, Sample, Condition). Requires --metadata_modkit_bed_file.",
        default=""
    )
    required_transcript_expr_mod.add_argument(
        "--metadata_modkit_bed_file","-n",dest="metadata_modkit_file",type= str,
        help="Path to metadatafile for Modkit bed files (columns: File, Sample, Condition). Requires --metadata_count_file.",
        default=""
    )
    
    # Input mode 2: Both data file lists (count + modkit) with conditions/samplenames
    required_transcript_expr_mod.add_argument(
        "--data-count-files","-d",dest="data_count_files",nargs="+",type=str,
        help="List of count files. Requires --data-mod-bed-files, --conditions, and --samplenames.",
        default=[]
    )
    required_transcript_expr_mod.add_argument(
        "--data-mod-bed-files","-D",dest="data_mod_bed_files",nargs="+",type=str,
        help="List of modkit bed files. Requires --data-count-files, --conditions, and --samplenames.",
        default=[]
    )
    required_transcript_expr_mod.add_argument(
        "--reference_levels","-r",dest="reference_levels",nargs="+",type=str,
        help="Define which of the provided conditions should be used as a ctrl sample.",
        required=True
    )
    required_transcript_expr_mod.add_argument(
        "--gtf-file","-g",dest="gtf_file",type=str,help="Path to gtf file of genome the data was aligned to.",
        required=True
    )
    required_transcript_expr_mod.add_argument(
        "--output_folder", "-o", dest="output_folder",type=str,help="Path to output folder",
        required=True
    )
    
    # Optional arguments
    optional_transcript_expr_mod = transcriptome_expression_and_modification_analysis_parser.add_argument_group('optional arguments')
    optional_transcript_expr_mod.add_argument(
        "--conditions","-c",dest="conditions",nargs="+",type=str,
        help="Condition labels for each sample (required when using --data-count-files and --data-mod-bed-files).",
        default=[]
    )
    optional_transcript_expr_mod.add_argument(
        "--samplenames","-s",dest="samplenames",nargs="+",type=str,
        help="Sample names for each file (required when using --data-count-files and --data-mod-bed-files).",
        default=[]
    )
    optional_transcript_expr_mod.add_argument(
        "--alpha_modification", "-a", dest="alpha_modification",type=float, help="Provide a threshold for statistical tests on differential modification",
        default=0.01
    )
    optional_transcript_expr_mod.add_argument(
        "--alpha_expression", "-A", dest="alpha_expression",type=float, help="Provide a threshold for statistical tests on differential expression",
        default=0.01
    )
    optional_transcript_expr_mod.add_argument(
        "--chromosome","-x",dest="chrom",type=str,
        help="Provide a chromosome of interest occuring in your dataset.",
        default=""
    )
    optional_transcript_expr_mod.add_argument(
        "--coverage_filter","-f",dest="minimum_coverage",type=int,
        help="Define a minimum coverage",
        default=20
    )
    optional_transcript_expr_mod.add_argument(
        "--frequency_filter","-q",dest="minimum_mod_frequency",type=int,
        help="Define a minimum modification frequency",
        default=10
    )
    optional_transcript_expr_mod.add_argument(
        "--p_value_correction","-v",dest="pvalue_correction",type=str,
        help="Select a pvalue correction method for modification detection. (SLIM (Significance Level-based Iterative Method) or bh (Benjamini-Hochberg))",
        default="SLIM",
    )
    optional_transcript_expr_mod.add_argument(
        "--modification", dest="modification", nargs="+", type=str,
        help=f"Specify modification type(s) to analyze. Options: {list(BASE_MODIFICATION_TYPE_DICT.keys())}. If not provided, all modification types present in the data will be analyzed.",
        default=None
    )
    transcriptome_expression_and_modification_analysis_parser.set_defaults(func=transcript_expression_and_modification_analysis)

    
    # Parse and run
    args = parser.parse_args()
    
    logger.info(f"Output folder: {args.output_folder}")
    
    if args.output_folder == "":
        if os.path.exists("./dmode_progress.log"):
            with open("./dmode_progress.log", "w") as file:
                file.write("")
        file_stream = logging.FileHandler("./dmode_progress.log")
        file_stream.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_stream)
        logger.setLevel("INFO")
    else:
        if os.path.exists(f"{args.output_folder}/dmode_progress.log"):
            with open(f"{args.output_folder}/dmode_progress.log", "w") as file:
                file.write("")
        file_stream = logging.FileHandler(f"{args.output_folder}/dmode_progress.log")
        file_stream.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_stream)
        logger.setLevel("INFO")        
        
    args.func(args)