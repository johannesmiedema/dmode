import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind, fisher_exact, hypergeom
import numpy as np
from math import log, exp, isinf
import statsmodels.api as sm
from scipy.stats import f, chi2, rankdata
from tqdm import tqdm
import os
import argparse
from dmode.differential_modification_analysis import (diff_gene_mod_cond_extract_significant_positions,
                                                    diff_gene_mod_cond_extract_significant_genes,
                                                    diff_gene_mod_cond_generate_volcano_plots,
                                                    diff_gene_mod_cond_generate_volcano_plot_single_chr,
                                                    diff_gene_mod_cond_extraction,
                                                    diff_transcript_mod_cond_extract_significant_positions,
                                                    diff_transcript_mod_cond_extract_significant_transcripts,
                                                    diff_transcript_mod_cond_generate_volcano_plot_single_chr,
                                                    diff_transcript_mod_cond_extraction,
                                                    logReg,
                                                    SLIMfunc,
                                                    benjamini_hochberg,
                                                    fisher_exact_test,
                                                    _append_positions_column,
                                                    _prepare_gene_annotation_lookup,
                                                    _prepare_transcript_annotation_lookup,
)


from dmode.differential_expression_analysis import (merge_single_sample_feature_count_tables,
                                                    merge_single_sample_salmon_count_tables,
                                                    prepare_deseq_dataset,
                                                    diff_exp_analysis,
                                                    diff_exp_volcano_plot,
                                                    diff_exp_heatmap_all_conditions,
                                                    diff_exp_ma_plot,
                                                    diff_exp_heatmap_pairwise,
                                                    diff_exp_sample_to_sample_plot,
                                                    diff_exp_pca_plot,
)

from dmode.base_stats import (generate_PCA_Plot,
                              generate_UMAP_plot,
                              generate_modsite_barplot,
                              generate_modification_distribution_plot,
                              generate_modification_violin_plot,
                              generate_sample_to_sample_correlation_plot,
                              barplot_intersections_database)

from dmode.metagene_plot import (prepare_gene_body_coverage,
                                 metagene_body_coverage)
#from dmode.QC import (generate_numberofreads_barplot
#)

from dmode.differential_expression_and_modification_analysis import (
    gene_expression_vs_modification_differcence_scatterplot,
    expression_modification_correlation_scatterplot,
    plot_top50_genes_with_most_number_of_significant_modifications)
from dmode.utility import convert_bed_to_df, gtf_to_df, moddict, moddict_reverse, moddict_mapping, moddict_mapping_reverse
from dmode.dmode import DmodE


__all__ = [
    # differential_modification_analysis
    'diff_gene_mod_cond_extract_significant_positions',
    'diff_gene_mod_cond_extract_significant_genes',
    'diff_gene_mod_cond_generate_volcano_plots',
    'diff_gene_mod_cond_generate_volcano_plot_single_chr',
    'diff_gene_mod_cond_extraction',
    'diff_transcript_mod_cond_extract_significant_positions',
    'diff_transcript_mod_cond_extract_significant_transcripts',
    'diff_transcript_mod_cond_generate_volcano_plot_single_chr',
    'diff_transcript_mod_cond_extraction',
    'logReg',
    'SLIMfunc',
    'benjamini_hochberg',
    'fisher_exact_test',
    '_append_positions_column',
    '_prepare_gene_annotation_lookup',
    '_prepare_transcript_annotation_lookup',
    # differential_expression_analysis
    'merge_single_sample_feature_count_tables',
    'merge_single_sample_salmon_count_tables',
    'prepare_deseq_dataset',
    'diff_exp_analysis',
    'diff_exp_volcano_plot',
    'diff_exp_heatmap_all_conditions',
    'diff_exp_ma_plot',
    'diff_exp_heatmap_pairwise',
    'diff_exp_sample_to_sample_plot',
    'diff_exp_pca_plot',
    # base_stats
    'generate_PCA_Plot',
    'generate_UMAP_plot',
    'generate_modsite_barplot',
    'generate_modification_distribution_plot',
    'generate_modification_violin_plot',
    'generate_sample_to_sample_correlation_plot',
    'barplot_intersections_database',
    # metagene_plot
    'prepare_gene_body_coverage',
    'metagene_body_coverage',
    # differential_expression_and_modification_analysis
    'gene_expression_vs_modification_differcence_scatterplot',
    'expression_modification_correlation_scatterplot',
    'plot_top50_genes_with_most_number_of_significant_modifications',
    # utility
    'convert_bed_to_df',
    'gtf_to_df',
    'moddict',
    'moddict_mapping',
    # dmode
    'DmodE',
]
 