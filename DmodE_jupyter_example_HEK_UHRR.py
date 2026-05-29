#%%
from dmode import *
import pandas as pd

gtf_file_path = "gencode.v49.annotation.gtf"
reference_levels = ["UHRR"]
minimum_coverage = 20
minimum_mod_frequency = 10
modification = "m6A"
output_folder = ""
comparison = "HEK293_VS_UHRR"
chromosome = "chr1"
alpha = 0.05


########################################################
# Differential modification analysis at the gene level #
########################################################
#%%

data_files = [
    "UHRR1_modkit.bed", "UHRR2_modkit.bed",
    "HEK293_1_modkit.bed", "HEK293_2_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
DmodE_object.gene_mod_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
)

## Alternatively you can create a tsv table:
"""
#################################################
# Example:
# File                    Condition   Sample
# UHRR1_modkit.bed        UHRR        UHRR1
# UHRR2_modkit.bed        UHRR        UHRR2
# HEK293_1_modkit.bed     HEK293      HEK293_1
# HEK293_2_modkit.bed     HEK293      HEK293_2
#################################################
Then run:
DmodE_object.gene_mod_preprocess_data(metadata=metadata, reference_levels=reference_levels,
                                      coverage_filter=minimum_coverage, mod_freq_filter=minimum_mod_frequency)
"""

diff_gene_mod_comparisons_dict, diff_gene_mod_outer_join_comparisons_dict = \
    diff_gene_mod_cond_extraction(dmode_obj=DmodE_object, correction_method="SLIM", output_folder=output_folder)

# Prepare gene annotation lookup for the comparisons
comparison_annotations_genes = _prepare_gene_annotation_lookup(
    DmodE_object,
    gtf_file_path,
    alpha,
    output_folder,
)

## Generate volcano plot for a single chromosome
fig1 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
    chromosome,
    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
    alpha=alpha,
    modtype="m6A",   # modtype="" will plot all modifications together
    annotation_df=comparison_annotations_genes.get(comparison),
)

## Generate volcano plots for all chromosomes
fig2 = diff_gene_mod_cond_generate_volcano_plots(
    DmodE_object.diff_gene_mod_chr_names,
    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
    alpha=alpha,
    modtype="m6A",   # modtype="" will plot all modifications together
    annotation_df=comparison_annotations_genes.get(comparison),
)

## Extract significant positions
significant_gene_mods_df = diff_gene_mod_cond_extract_significant_positions(
    dmode_obj=DmodE_object,
    ref_level="UHRR",
    cond_level="HEK293",
    alpha=alpha,
    chr="",
    output_folder=output_folder,
)


##############################################################
# Differential modification analysis at the transcript level #
##############################################################
#%%
minimum_coverage = 5
minimum_mod_frequency = 5

data_files = [
    "UHRR1_transcriptome_modkit.bed", "UHRR2_transcriptome_modkit.bed",
    "HEK293_1_transcriptome_modkit.bed", "HEK293_2_transcriptome_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
DmodE_object.transcript_mod_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
)

## Alternatively:
"""
#################################################
# Example:
# File                               Condition   Sample
# UHRR1_transcriptome_modkit.bed     UHRR        UHRR1
# UHRR2_transcriptome_modkit.bed     UHRR        UHRR2
# HEK293_1_transcriptome_modkit.bed  HEK293      HEK293_1
# HEK293_2_transcriptome_modkit.bed  HEK293      HEK293_2
#################################################
Then run:
DmodE_object.transcript_mod_preprocess_data(metadata=metadata, reference_levels=reference_levels,
                                            coverage_filter=minimum_coverage, mod_freq_filter=minimum_mod_frequency)
"""

diff_transcript_mod_comparisons_dict, diff_transcript_mod_outer_join_comparisons_dict = \
    diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object, output_folder=output_folder, correction_method="SLIM")

comparison_annotations_transcripts = _prepare_transcript_annotation_lookup(
    DmodE_object,
    gtf_file_path,
    alpha,
    output_folder,
)

fig4 = diff_transcript_mod_cond_generate_volcano_plot_single_chr(
    chromosome,   # chromosome="" will plot all chromosomes together
    DmodE_object.diff_transcript_mod_comparisons_dict[comparison],
    alpha=alpha,
    modtype=modification,   # modtype="" will plot all modifications together
    annotation_df=comparison_annotations_transcripts.get(comparison),
)

significant_transcript_mods_df = diff_transcript_mod_cond_extract_significant_positions(
    dmode_obj=DmodE_object,
    ref_level="UHRR",
    cond_level="HEK293",
    alpha=alpha,
    chr=chromosome,
    output_folder=output_folder,
)


#########################################
# Differential gene expression analysis #
#########################################
#%%
data_files = [
    "UHRR1_counts.csv", "UHRR2_counts.csv",
    "HEK293_1_counts.csv", "HEK293_2_counts.csv",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    output_folder=output_folder,
)

## Alternatively:
"""
#################################################
# Example:
# File                    Condition   Sample
# UHRR1_counts.csv        UHRR        UHRR1
# UHRR2_counts.csv        UHRR        UHRR2
# HEK293_1_counts.csv     HEK293      HEK293_1
# HEK293_2_counts.csv     HEK293      HEK293_2
#################################################
Then run:
counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(metadata=metadata,
                                                                        reference_levels=reference_levels,
                                                                        output_folder=output_folder)
"""

result_df = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=condition_dict,
    alpha=alpha,
    output_folder=output_folder,
    gtf_file=gtf_file_path,
    operational_level="genome",
)
result_df_temp = pd.DataFrame(result_df)

fig5 = diff_exp_volcano_plot(result_df=result_df_temp, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="genome")
fig6 = diff_exp_ma_plot(result_df=result_df_temp, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="genome")
fig7 = diff_exp_heatmap_pairwise(result_df=result_df_temp, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, first_level="HEK293", ref_level="UHRR", operational_level="genome")
fig8 = diff_exp_heatmap_all_conditions(result_df=result_df_temp, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, operational_level="genome")
fig9 = diff_exp_sample_to_sample_plot(counts_df=counts_df, output_folder=output_folder, operational_level="genome")


###############################################
# Differential transcript expression analysis #
###############################################
#%%
DmodE_object = DmodE()
data_files = [
    "UHRR1_transcript_counts/quant.sf", "UHRR2_transcript_counts/quant.sf",
    "HEK293_1_transcript_counts/quant.sf", "HEK293_2_transcript_counts/quant.sf",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    output_folder=output_folder,
)

## Alternatively:
"""
#################################################
# Example:
# File                          Condition   Sample
# UHRR1_quant.sf                UHRR        UHRR1
# UHRR2_quant.sf                UHRR        UHRR2
# HEK293_1_quant.sf             HEK293      HEK293_1
# HEK293_2_quant.sf             HEK293      HEK293_2
#################################################
Then run:
counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(metadata=metadata,
                                                                               reference_levels=reference_levels,
                                                                               output_folder=output_folder)
"""

result_df = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=condition_dict,
    alpha=alpha,
    output_folder=output_folder,
    gtf_file=gtf_file_path,
    operational_level="transcriptome",
)

fig10 = diff_exp_volcano_plot(result_df=result_df, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="transcriptome")
fig11 = diff_exp_ma_plot(result_df=result_df, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="transcriptome")
fig12 = diff_exp_heatmap_pairwise(result_df=result_df, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, first_level="HEK293", ref_level="UHRR", operational_level="transcriptome")
fig13 = diff_exp_heatmap_all_conditions(result_df=result_df, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha, lfc_threshold=1, operational_level="transcriptome")
fig14 = diff_exp_sample_to_sample_plot(counts_df=counts_df, output_folder=output_folder, operational_level="transcriptome")


#########################################################
# Base stats and exploratory data analysis (gene level) #
#########################################################
#%%
minimum_coverage = 20
minimum_mod_frequency = 10

data_files = [
    "UHRR1_modkit.bed", "UHRR2_modkit.bed",
    "HEK293_1_modkit.bed", "HEK293_2_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
DmodE_object.gene_mod_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
    comparison=False,
)

fig15 = generate_PCA_Plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, annotate_samples=False, operational_level="gene")
fig16 = generate_UMAP_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, annotate_samples=False, operational_level="gene")
fig17 = generate_modsite_barplot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="gene")
fig18 = generate_modification_distribution_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="gene")
fig19 = generate_modification_violin_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="gene")
fig20 = generate_sample_to_sample_correlation_plot(processed_dataframes=DmodE_object.basic_gene_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="gene")


###############################################################
# Base stats and exploratory data analysis (transcript level) #
###############################################################
#%%
minimum_coverage = 5
minimum_mod_frequency = 5

data_files = [
    "UHRR1_transcriptome_modkit.bed", "UHRR2_transcriptome_modkit.bed",
    "HEK293_1_transcriptome_modkit.bed", "HEK293_2_transcriptome_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
DmodE_object.transcript_mod_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
    comparison=False,
)

fig22 = generate_PCA_Plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, annotate_samples=False, operational_level="transcript")
fig23 = generate_UMAP_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, annotate_samples=False, operational_level="transcript")
fig24 = generate_modsite_barplot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="transcript")
fig25 = generate_modification_distribution_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="transcript")
fig26 = generate_modification_violin_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="transcript")
fig27 = generate_sample_to_sample_correlation_plot(processed_dataframes=DmodE_object.basic_transcript_mod_dataframe_condition_dict, modification=modification, output_folder=output_folder, operational_level="transcript")


#################
# Metagene plot #
#################
#%%
minimum_coverage = 20
minimum_mod_frequency = 10
metagene_min_coverage = 10

data_files = [
    "UHRR1_modkit.bed", "UHRR2_modkit.bed",
    "HEK293_1_modkit.bed", "HEK293_2_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
DmodE_object.gene_mod_preprocess_data(
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
    comparison=False,
)

gtf_df, gene_body_df = prepare_gene_body_coverage(gtf_file=gtf_file_path)
interpolated_transcripts_df, feature_distribution_df, fig28, fig29, fig30_with_intronic_regions, fig31_exonic_regions_only = metagene_body_coverage(
    gtf_df=gtf_df,
    gene_body_df=gene_body_df,
    data_files=data_files,
    conditions=conditions,
    samplenames=samplenames,
    normalization="max",
    coverage_filter=metagene_min_coverage,
    mod_type=modification,
    gene_id="",
    output_folder=output_folder,
)


#####################################################################
# Differential expression vs diff modification analysis (gene level)#
#####################################################################
#%%
alpha_modification = 0.05
alpha_expression = 0.05

data_count_files = [
    "UHRR1_counts.csv", "UHRR2_counts.csv",
    "HEK293_1_counts.csv", "HEK293_2_counts.csv",
]
data_mod_bed_files = [
    "UHRR1_modkit.bed", "UHRR2_modkit.bed",
    "HEK293_1_modkit.bed", "HEK293_2_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
counts_df, condition_dict = DmodE_object.diff_gene_exp_preprocess_data(
    data_files=data_count_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    output_folder=output_folder,
)
DmodE_object.gene_mod_preprocess_data(
    data_files=data_mod_bed_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
)
diff_gene_mod_comparisons_dict, diff_gene_mod_outer_join_comparisons_dict = \
    diff_gene_mod_cond_extraction(dmode_obj=DmodE_object, output_folder=output_folder, correction_method="SLIM")

comparison_annotations = _prepare_gene_annotation_lookup(
    DmodE_object,
    gtf_file_path,
    alpha_modification,
    output_folder,
)

fig32 = diff_gene_mod_cond_generate_volcano_plot_single_chr(
    "",
    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
    alpha=alpha_modification,
    annotation_df=comparison_annotations.get(comparison),
)
fig33 = diff_gene_mod_cond_generate_volcano_plots(
    DmodE_object.diff_gene_mod_chr_names,
    DmodE_object.diff_gene_mod_comparisons_dict[comparison],
    alpha=alpha_modification,
    annotation_df=comparison_annotations.get(comparison),
)

diff_gene_mod_cond_extract_significant_positions(
    dmode_obj=DmodE_object,
    ref_level="UHRR",
    cond_level="HEK293",
    alpha=alpha_modification,
    chr="",
    output_folder=output_folder,
)

result_df = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=condition_dict,
    alpha=alpha_expression,
    output_folder=output_folder,
    gtf_file=gtf_file_path,
    operational_level="genome",
)
result_df_temp = pd.DataFrame(result_df)

fig34 = diff_exp_volcano_plot(result_df=result_df_temp, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="genome")
fig35 = diff_exp_ma_plot(result_df=result_df_temp, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="genome")
fig36 = diff_exp_heatmap_pairwise(result_df=result_df_temp, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, first_level="HEK293", ref_level="UHRR", operational_level="genome")
fig37 = diff_exp_heatmap_all_conditions(result_df=result_df_temp, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, operational_level="genome")
fig38 = diff_exp_sample_to_sample_plot(counts_df=counts_df, output_folder=output_folder, operational_level="genome")

comparison_data = {
    "modification": diff_gene_mod_comparisons_dict[comparison],
    "expression": result_df,
    "annotation": comparison_annotations.get(comparison),
}
figures1 = gene_expression_vs_modification_differcence_scatterplot(comparison_data, output_folder, operational_level="genome")
figures2 = expression_modification_correlation_scatterplot(comparison_data, output_folder, operational_level="genome")
figures3 = plot_top50_genes_with_most_number_of_significant_modifications(comparison_data, output_folder, operational_level="genome")


###########################################################################
# Differential expression vs diff modification analysis (transcript level)#
###########################################################################
#%%
alpha_modification = 0.05
alpha_expression = 0.05

data_count_files = [
    "UHRR1_transcript_counts/quant.sf", "UHRR2_transcript_counts/quant.sf",
    "HEK293_1_transcript_counts/quant.sf", "HEK293_2_transcript_counts/quant.sf",
]
data_mod_bed_files = [
    "UHRR1_transcriptome_modkit.bed", "UHRR2_transcriptome_modkit.bed",
    "HEK293_1_transcriptome_modkit.bed", "HEK293_2_transcriptome_modkit.bed",
]
conditions   = ["UHRR", "UHRR", "HEK293", "HEK293"]
samplenames  = ["UHRR1", "UHRR2", "HEK293_1", "HEK293_2"]

DmodE_object = DmodE()
counts_df, condition_dict = DmodE_object.diff_transcript_exp_preprocess_data(
    data_files=data_count_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    output_folder=output_folder,
)
DmodE_object.transcript_mod_preprocess_data(
    data_files=data_mod_bed_files,
    conditions=conditions,
    samplenames=samplenames,
    reference_levels=reference_levels,
    coverage_filter=minimum_coverage,
    mod_freq_filter=minimum_mod_frequency,
)
diff_transcript_mod_comparisons_dict, diff_transcript_mod_outer_join_comparisons_dict = \
    diff_transcript_mod_cond_extraction(dmode_obj=DmodE_object, output_folder=output_folder, correction_method="SLIM")

comparison_annotations = _prepare_transcript_annotation_lookup(
    DmodE_object,
    gtf_file_path,
    alpha_modification,
    output_folder,
)

diff_transcript_mod_cond_extract_significant_positions(
    dmode_obj=DmodE_object,
    ref_level="UHRR",
    cond_level="HEK293",
    alpha=alpha_modification,
    chr="",
    output_folder=output_folder,
)

result_df = diff_exp_analysis(
    counts_df=counts_df,
    ref_level="UHRR",
    first_level="HEK293",
    condition_dict=condition_dict,
    alpha=alpha_expression,
    output_folder=output_folder,
    gtf_file=gtf_file_path,
    operational_level="transcriptome",
)

fig39 = diff_exp_volcano_plot(result_df=result_df, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="transcriptome")
fig40 = diff_exp_ma_plot(result_df=result_df, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, ref_level="UHRR", first_level="HEK293", operational_level="transcriptome")
fig41 = diff_exp_heatmap_pairwise(result_df=result_df, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, first_level="HEK293", ref_level="UHRR", operational_level="transcriptome")
fig42 = diff_exp_heatmap_all_conditions(result_df=result_df, counts_df=counts_df, condition_dict=condition_dict, output_folder=output_folder, pvalue_threshold=alpha_expression, lfc_threshold=1, operational_level="transcriptome")
fig43 = diff_exp_sample_to_sample_plot(counts_df=counts_df, output_folder=output_folder, operational_level="transcriptome")

comparison_data = {
    "modification": diff_transcript_mod_comparisons_dict[comparison],
    "expression": result_df,
    "annotation": comparison_annotations.get(comparison),
}
figures4 = gene_expression_vs_modification_differcence_scatterplot(comparison_data, output_folder, operational_level="transcriptome")
figures5 = expression_modification_correlation_scatterplot(comparison_data, output_folder, operational_level="transcriptome")
figures6 = plot_top50_genes_with_most_number_of_significant_modifications(comparison_data, output_folder, operational_level="transcriptome")