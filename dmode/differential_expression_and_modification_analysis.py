import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import os
import warnings
from scipy.stats import linregress
from adjustText import adjust_text

logger = logging.getLogger(__name__)
  
def gene_expression_vs_modification_differcence_scatterplot(
        data_dict: dict,
        output_dir: str = "",
        log2fc_threshold: float = 0,
        operational_level: str = 'genome'
) -> dict:
    """
    Plots a scatterplot comparing gene expression and modification differences.
    Analysis needed to be done before running this function: 
    1.diff_genome_modification_condition_comparison
    2.diff_gene_expression_analysis
    The output_dir should be the directory where both analysis results are stored.

    Returns
    -------
    dict
        Nested dictionary mapping modification type → plot name → Figure, e.g.:
        {
            'm6A': {
                'combined':  <Figure>,
                'nonsig':    <Figure>,
                'exp_sig':   <Figure>,
                'mod_sig':   <Figure>,
                'all_sig':   <Figure>,
            },
            ...
        }
    """
    # Load data
    modification_df = data_dict.get('modification')
    expression_df = data_dict.get('expression')
    annotation_file = data_dict.get('annotation')

    # Determine entity name column and labels based on operational level
    entity_name_col = 'transcript_name' if operational_level == 'transcriptome' else 'gene_name'
    entity_label    = 'Transcript'      if operational_level == 'transcriptome' else 'Gene'

    available_mod_types = modification_df['mod_type'].unique()

    effective_output_dir = os.path.join(output_dir, 'expression_and_modification_analysis') if output_dir != "" else ""
    if effective_output_dir:
        os.makedirs(effective_output_dir, exist_ok=True)

    figures = {}

    for modtype in available_mod_types:
        modification_df_filtered = modification_df[modification_df['mod_type'] == modtype].copy()

        # Build position tokens
        if operational_level == 'genome':
            annotation_file['positions'] = (
                annotation_file['chrom'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )
        else:  # transcriptome
            annotation_file['positions'] = (
                annotation_file['transcript_id'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )
            modification_df_filtered['positions'] = (
                modification_df_filtered['id'].str.split('|').str[0] + ":" +
                modification_df_filtered['start'].astype(str) + ":" +
                modification_df_filtered['end'].astype(str) + ":" +
                modification_df_filtered['strand'].astype(str) + ":" +
                modification_df_filtered['mod_type'].astype(str)
            )

        # Associate entity names for modification positions
        modification_df_filtered = modification_df_filtered.merge(
            annotation_file[['positions', entity_name_col]],
            on='positions',
            how='left'
        )
        modification_df_filtered = modification_df_filtered.dropna(subset=[entity_name_col])

        if len(modification_df_filtered) == 0:
            logger.warning("No modification data left for %s after merging with annotations", modtype)
            continue

        # Merge with expression data
        merged_df = pd.merge(
            modification_df_filtered,
            expression_df[[entity_name_col, 'log2FoldChange', 'padj']],
            on=entity_name_col,
            how='left',
            suffixes=('_mod', '_exp')
        )

        # Significance flags
        merged_df['all significance'] = (merged_df['padj_mod'] < 0.05) & (merged_df['padj_exp'] < 0.05)
        merged_df['mod significance'] = (merged_df['padj_mod'] < 0.05)
        merged_df['exp significance'] = (merged_df['padj_exp'] < 0.05)

        # Significance subsets
        sub_nonsig  = merged_df[~merged_df['mod significance'] & ~merged_df['exp significance']]
        sub_exp_sig = merged_df[~merged_df['mod significance'] &  merged_df['exp significance']]
        sub_mod_sig = merged_df[ merged_df['mod significance'] & ~merged_df['exp significance']]
        sub_sig     = merged_df[ merged_df['all significance']]

        # ── helper: annotate top-N points per quadrant ──────────────────────
        def _annotate_quadrants(ax, df, name_col, n=15):
            df = df.copy()
            x_std = df['mod_freq_diff'].std()
            y_std = df['log2FoldChange'].std()
            if x_std > 0 and y_std > 0:
                df['distance_from_origin'] = np.sqrt(
                    ((df['mod_freq_diff']  - df['mod_freq_diff'].mean())  / x_std) ** 2 +
                    ((df['log2FoldChange'] - df['log2FoldChange'].mean()) / y_std) ** 2
                )
            else:
                df['distance_from_origin'] = 0.0

            quads = [
                df[(df['mod_freq_diff'] > 0) & (df['log2FoldChange'] > 0)],
                df[(df['mod_freq_diff'] < 0) & (df['log2FoldChange'] > 0)],
                df[(df['mod_freq_diff'] < 0) & (df['log2FoldChange'] < 0)],
                df[(df['mod_freq_diff'] > 0) & (df['log2FoldChange'] < 0)],
            ]
            texts = []
            for quad in quads:
                if not quad.empty:
                    for _, point in quad.nlargest(n, 'distance_from_origin').iterrows():
                        texts.append(ax.text(
                            point['mod_freq_diff'], point['log2FoldChange'],
                            point[name_col], fontsize=7, alpha=0.9,
                        ))
            if texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=0.5))

        # ── helper: shared axis decoration ──────────────────────────────────
        def _decorate_ax(ax, title_suffix=''):
            ax.axhline(0, color='grey', linestyle='--')
            ax.axvline(0, color='grey', linestyle='--')
            ax.set_title(f'{entity_label} Expression vs. {modtype} Frequency Difference{title_suffix}')
            ax.set_xlabel(f'{modtype} Frequency Difference')
            ax.set_ylabel(f'Log2 Fold Change in {entity_label} Expression')

        # ── legend elements (shared) ─────────────────────────────────────────
        significance_legend_elements = [
            mlines.Line2D([], [], color='#888888', marker='o', linestyle='None', markersize=10, label='Non-significant'),
            mlines.Line2D([], [], color='#3A0CE2', marker='o', linestyle='None', markersize=10, label='Mod significant only'),
            mlines.Line2D([], [], color='#E20C3A', marker='o', linestyle='None', markersize=10, label='Exp significant only'),
            mlines.Line2D([], [], color='#000000', marker='o', linestyle='None', markersize=10, label='Significant in both'),
        ]

        if effective_output_dir:
            mod_output_dir = os.path.join(effective_output_dir, modtype)
            os.makedirs(mod_output_dir, exist_ok=True)

        figures[modtype] = {}

        # ── combined plot ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))

        for subset, color, alpha, zorder in [
            (sub_nonsig,  '#888888', 0.15, 1),
            (sub_exp_sig, '#E20C3A', 0.15, 2),
            (sub_mod_sig, '#3A0CE2', 0.15, 2),
            (sub_sig,     '#000000', 0.15, 2),
        ]:
            if not subset.empty:
                ax.scatter(
                    subset['mod_freq_diff'], subset['log2FoldChange'],
                    c=color, alpha=alpha, edgecolors='none', zorder=zorder
                )

        _annotate_quadrants(ax, merged_df, entity_name_col)
        _decorate_ax(ax)

        legend = ax.legend(handles=significance_legend_elements, title='Significances', loc='upper right')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')
        fig.tight_layout()

        # Capture axis limits for consistent scaling across category plots
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        prefix = 'transcript' if operational_level == 'transcriptome' else 'gene'
        if effective_output_dir:
            fig.savefig(os.path.join(mod_output_dir, f'{prefix}_expression_vs_{modtype}_difference_scatterplot_combined.png'))
            fig.savefig(os.path.join(mod_output_dir, f'{prefix}_expression_vs_{modtype}_difference_scatterplot_combined.svg'), format='svg')
        figures[modtype]['combined'] = fig

        # Save merged data for HTML report
        if effective_output_dir:
            merged_df.to_csv(os.path.join(mod_output_dir, f'{prefix}_expression_vs_{modtype}_difference_data.csv'), index=False)

        # ── individual category plots ────────────────────────────────────────
        categories = [
            ('nonsig',  sub_nonsig,  '#888888', 'Non-significant'),
            ('exp_sig', sub_exp_sig, '#E20C3A', 'Expression significant only'),
            ('mod_sig', sub_mod_sig, '#3A0CE2', 'Modification significant only'),
            ('all_sig', sub_sig,     '#000000', 'Significant in both'),
        ]

        for cat_name, cat_data, color, cat_title in categories:
            if cat_data.empty:
                continue

            fig_cat, ax_cat = plt.subplots(figsize=(10, 8))
            ax_cat.scatter(
                cat_data['mod_freq_diff'], cat_data['log2FoldChange'],
                c=color, alpha=0.6, edgecolors='none', zorder=2
            )

            _annotate_quadrants(ax_cat, cat_data, entity_name_col)
            _decorate_ax(ax_cat, title_suffix=f' - {cat_title}')

            ax_cat.set_xlim(x_limits)
            ax_cat.set_ylim(y_limits)
            fig_cat.tight_layout()

            if effective_output_dir:
                fig_cat.savefig(os.path.join(mod_output_dir, f'{prefix}_expression_vs_{modtype}_difference_scatterplot_{cat_name}.png'))
                fig_cat.savefig(os.path.join(mod_output_dir, f'{prefix}_expression_vs_{modtype}_difference_scatterplot_{cat_name}.svg'), format='svg')
            figures[modtype][cat_name] = fig_cat

    return figures            
            

def expression_modification_correlation_scatterplot(
        data_dict: dict,
        output_dir: str = "",
        log2fc_threshold: float = 0,
        operational_level: str = 'genome'
) -> dict:
    """
    Plots a scatterplot comparing expression and modification differences per quadrant,
    only for significant genes/transcripts and modifications (padj < 0.05).
    
    Quadrants:
    1. Mod Down, Expression Up (negative correlation)
    2. Both Up (positive correlation)
    3. Both Down (positive correlation)
    4. Expression Down, Mod Up (negative correlation)

    Parameters
    ----------
    data_dict : dict
        Dictionary containing 'modification', 'expression', and 'annotation' DataFrames
    output_dir : str
        Directory to save output plots
    log2fc_threshold : float
        Log2 fold change threshold (default: 0)
    operational_level : str
        Either 'genome' for gene-level or 'transcriptome' for transcript-level analysis

    Returns
    -------
    dict
        Dictionary mapping modification type strings to their corresponding matplotlib Figure objects,
        e.g. {'m6A': <Figure>, 'psi': <Figure>}
    """
    # Load data
    modification_df = data_dict.get('modification')
    expression_df = data_dict.get('expression')
    annotation_file = data_dict.get('annotation')   

    # Determine entity name column based on operational level
    entity_name_col = 'transcript_name' if operational_level == 'transcriptome' else 'gene_name'
    entity_label = 'Transcript' if operational_level == 'transcriptome' else 'Gene'
    
    # Determine the modification type to filter on based on available types in the modification dataframe
    available_mod_types = modification_df['mod_type'].unique()

    # All plots will be saved in a new directory in the outputdir
    effective_output_dir = os.path.join(output_dir, 'expression_and_modification_analysis') if output_dir != "" else ""
    if effective_output_dir:
        os.makedirs(effective_output_dir, exist_ok=True)

    figures = {}

    # Iterate over available modification types and create separate plots for each
    for modtype in available_mod_types:
        # Filter modification dataframe for specific modification type
        modification_df_filtered = modification_df[modification_df['mod_type'] == modtype].copy()

        # Create position tokens based on operational level
        if operational_level == 'genome':
            # Genome level: use chromosome coordinates
            annotation_file['positions'] = (
                annotation_file['chrom'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )
        else:  # transcriptome
            # Transcript level: use transcript_id coordinates
            annotation_file['positions'] = (
                annotation_file['transcript_id'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )

            # Reformat modification positions to match annotation format
            modification_df_filtered['positions'] = (
                modification_df_filtered['id'].str.split('|').str[0] + ":" +
                modification_df_filtered['start'].astype(str) + ":" +
                modification_df_filtered['end'].astype(str) + ":" +
                modification_df_filtered['strand'].astype(str) + ":" +
                modification_df_filtered['mod_type'].astype(str)
            )

        # Associate entity names (gene_name or transcript_name) for modification positions
        modification_df_filtered = modification_df_filtered.merge(
            annotation_file[['positions', entity_name_col]],
            on='positions',
            how='left'
        )

        # Drop rows where entity_name is NA
        modification_df_filtered = modification_df_filtered.dropna(subset=[entity_name_col])

        # Merge modification and expression data on entity_name
        merged_df = pd.merge(
            modification_df_filtered,
            expression_df[[entity_name_col, 'log2FoldChange', 'padj']],
            on=entity_name_col,
            how='left',
            suffixes=('_mod', '_exp')
        )

        # Add significance column: True if both padj_mod and padj_exp < 0.05
        merged_df['all significance'] = (merged_df['padj_mod'] < 0.05) & (merged_df['padj_exp'] < 0.05)

        # Only keep rows where both expression and modification are significant
        sig_df = merged_df[merged_df['all significance']].copy()

        # Drop rows with missing values in required columns
        sig_df = sig_df.dropna(subset=['log2FoldChange', 'mod_freq_diff'])

        # Define quadrants (using entity_label for titles)
        quadrants = [
            {'name': f'Q1_mod_down_{entity_label.lower()}_up', 
             'cond': (sig_df['log2FoldChange'] > 0) & (sig_df['mod_freq_diff'] < 0), 
             'title': f'Q1: Mod Down, {entity_label} Up'},
            {'name': 'Q2_both_up', 
             'cond': (sig_df['log2FoldChange'] > 0) & (sig_df['mod_freq_diff'] > 0), 
             'title': 'Q2: Both Up'},
            {'name': 'Q3_both_down', 
             'cond': (sig_df['log2FoldChange'] < 0) & (sig_df['mod_freq_diff'] < 0), 
             'title': 'Q3: Both Down'},
            {'name': f'Q4_{entity_label.lower()}_down_mod_up', 
             'cond': (sig_df['log2FoldChange'] < 0) & (sig_df['mod_freq_diff'] > 0), 
             'title': f'Q4: {entity_label} Down, Mod Up'},
        ]

        # Scientific color palette for quadrants (colorblind-friendly)
        colors = ['#0072B2', '#009E73', '#E69F00', '#D55E00']  # Blue, Green, Orange, Red
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all points, colored by quadrant, and perform regression analysis
        for idx, quad in enumerate(quadrants):
            quad_df = sig_df[quad['cond']]
            if not quad_df.empty:
                # Scatter plot for this quadrant
                ax.scatter(
                    quad_df['mod_freq_diff'],
                    quad_df['log2FoldChange'],
                    c=colors[idx],
                    alpha=1.0,
                    label=f'{modtype} ({quad["title"]})',
                    edgecolors='none',
                    zorder=2
                ) 
                
                # Regression analysis for this quadrant (with error handling)
                x = quad_df['mod_freq_diff']
                y = quad_df['log2FoldChange']
                
                # Check if we have enough data points and sufficient variance for regression
                corners = [(0.01, 0.99), (0.99, 0.99), (0.01, 0.01), (0.99, 0.01)]
                va = 'top' if corners[idx][1] > 0.5 else 'bottom'
                ha = 'left' if corners[idx][0] < 0.5 else 'right'

                if len(x) >= 3 and x.std() > 0 and y.std() > 0:
                    try:
                        # Suppress runtime warnings from linregress (handles edge cases internally)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        
                        # Only plot regression line if results are valid
                        if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(r_value):
                            x_vals = np.array([x.min(), x.max()])
                            y_vals = intercept + slope * x_vals
                            ax.plot(x_vals, y_vals, color='black', linestyle='-', linewidth=2, 
                                    label=f'Regression ({quad["title"]})', zorder=3)
                            
                            # Format p-value string
                            p_str = f"{p_value:.5f}" if p_value < 0.01 else f"{p_value:.2f}"
                            
                            ax.text(corners[idx][0], corners[idx][1],
                                    f"{quad['title']}\nr = {r_value:.2f}\np = {p_str}",
                                    transform=ax.transAxes,
                                    verticalalignment=va,
                                    horizontalalignment=ha)
                        else:
                            ax.text(corners[idx][0], corners[idx][1],
                                    f"{quad['title']}\n(insufficient variance)",
                                    transform=ax.transAxes,
                                    verticalalignment=va,
                                    horizontalalignment=ha)
                    except Exception as e:
                        logger.warning("Could not compute regression for %s: %s", quad['title'], e)
                        ax.text(corners[idx][0], corners[idx][1],
                                f"{quad['title']}\n(regression failed)",
                                transform=ax.transAxes,
                                verticalalignment=va,
                                horizontalalignment=ha)
                else:
                    ax.text(corners[idx][0], corners[idx][1],
                            f"{quad['title']}\n(n={len(x)})",
                            transform=ax.transAxes,
                            verticalalignment=va,
                            horizontalalignment=ha)
        
        # Annotate top entities from each quadrant in the correlation plot
        sig_df_copy = sig_df.copy()
        
        # Normalize coordinates to account for different scales
        x_normalized = (sig_df_copy['mod_freq_diff'] - sig_df_copy['mod_freq_diff'].mean()) / sig_df_copy['mod_freq_diff'].std()
        y_normalized = (sig_df_copy['log2FoldChange'] - sig_df_copy['log2FoldChange'].mean()) / sig_df_copy['log2FoldChange'].std()
        sig_df_copy['distance_from_origin'] = np.sqrt(x_normalized**2 + y_normalized**2)
        
        # Define quadrants for annotation and get top 15 entities from each
        quadrants_for_annotation = [
            sig_df_copy[(sig_df_copy['mod_freq_diff'] > 0) & (sig_df_copy['log2FoldChange'] > 0)],  # Q2: both positive
            sig_df_copy[(sig_df_copy['mod_freq_diff'] < 0) & (sig_df_copy['log2FoldChange'] > 0)],  # Q1: mod negative, exp positive
            sig_df_copy[(sig_df_copy['mod_freq_diff'] < 0) & (sig_df_copy['log2FoldChange'] < 0)],  # Q3: both negative
            sig_df_copy[(sig_df_copy['mod_freq_diff'] > 0) & (sig_df_copy['log2FoldChange'] < 0)]   # Q4: mod positive, exp negative
        ]
        
        # Annotate top 15 entities from each quadrant
        texts = []
        for quad_df in quadrants_for_annotation:
            if not quad_df.empty:
                for _, point in quad_df.nlargest(15, 'distance_from_origin').iterrows():
                    texts.append(ax.text(
                        point['mod_freq_diff'], point['log2FoldChange'],
                        point[entity_name_col], fontsize=7, alpha=0.9,
                    ))
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=0.5))
        
        # Add reference lines and labels
        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_title(f'{entity_label} Expression vs. {modtype} Difference - Correlations')
        ax.set_xlabel(f'{modtype} Frequency Difference')
        ax.set_ylabel(f'Log2 Fold Change in {entity_label} Expression')
        
        # Legend for quadrants (colors) - outside plot
        quad_legend = [mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None', 
                                     markersize=10, label=quadrants[i]['title']) for i in range(4)]
        legend = ax.legend(handles=quad_legend, title='Quadrant', loc='upper left', bbox_to_anchor=(1.02, 1))
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_edgecolor('none')
        fig.tight_layout()
        
        # Save the plot
        if effective_output_dir:
            mod_output_dir = os.path.join(effective_output_dir, modtype)
            os.makedirs(mod_output_dir, exist_ok=True)
            fig.savefig(os.path.join(mod_output_dir, f'{operational_level}_expression_{modtype}_correlation_quadrants_single.png'))
            fig.savefig(os.path.join(mod_output_dir, f'{operational_level}_expression_{modtype}_correlation_quadrants_single.svg'), format='svg')
        figures[modtype] = fig

    return figures


def plot_top50_genes_with_most_number_of_significant_modifications(
    data_dict: dict,
    output_dir: str = "",
    operational_level: str = 'genome'
) -> dict:
    """
    Plots the top genes/transcripts with the most number of significant modifications.
    Creates a cleaner visualization focusing on the most important entities.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing 'modification', 'expression', and 'annotation' DataFrames
    output_dir : str
        Directory to save output plots
    operational_level : str
        Either 'genome' for gene-level or 'transcriptome' for transcript-level analysis

    Returns
    -------
    dict
        Dictionary mapping modification type strings to their corresponding matplotlib Figure objects,
        e.g. {'m6A': <Figure>, 'psi': <Figure>}
    """
    # Load data
    modification_df = data_dict.get('modification')
    expression_df = data_dict.get('expression')
    annotation_file = data_dict.get('annotation')   

    # Determine entity name column based on operational level
    entity_name_col = 'transcript_name' if operational_level == 'transcriptome' else 'gene_name'
    entity_label = 'Transcript' if operational_level == 'transcriptome' else 'Gene'
    entity_label_plural = 'Transcripts' if operational_level == 'transcriptome' else 'Genes'
    
    # Determine the modification type to filter on based on available types in the modification dataframe
    available_mod_types = modification_df['mod_type'].unique()

    # All plots will be saved in a new directory in the outputdir
    effective_output_dir = os.path.join(output_dir, 'expression_and_modification_analysis') if output_dir != "" else ""
    if effective_output_dir:
        os.makedirs(effective_output_dir, exist_ok=True)

    figures = {}

    # Iterate over available modification types and create separate plots for each
    for modtype in available_mod_types:
        # Filter modification dataframe for specific modification type
        modification_df_filtered = modification_df[modification_df['mod_type'] == modtype].copy()

        # Create position tokens based on operational level
        if operational_level == 'genome':
            # Genome level: use chromosome coordinates
            annotation_file['positions'] = (
                annotation_file['chrom'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +    
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )
        else:  # transcriptome
            # Transcript level: use transcript_id coordinates
            annotation_file['positions'] = (
                annotation_file['transcript_id'].astype(str) + ":" +
                annotation_file['start'].astype(str) + ":" +
                annotation_file['end'].astype(str) + ":" +
                annotation_file['strand'].astype(str) + ":" +
                annotation_file['mod_type'].astype(str)
            )
            
            # Reformat modification positions to match annotation format
            modification_df_filtered['positions'] = (
                modification_df_filtered['id'].str.split('|').str[0] + ":" +
                modification_df_filtered['start'].astype(str) + ":" +
                modification_df_filtered['end'].astype(str) + ":" +
                modification_df_filtered['strand'].astype(str) + ":" +
                modification_df_filtered['mod_type'].astype(str)
            )
        
        # Associate entity names (gene_name or transcript_name) for modification positions
        modification_df_filtered = modification_df_filtered.merge(
            annotation_file[['positions', entity_name_col]],
            on='positions',
            how='left'
        )
        
        # Drop rows where entity_name is NA
        modification_df_filtered = modification_df_filtered.dropna(subset=[entity_name_col])
        
        # Merge modification and expression data on entity_name
        merged_df = pd.merge(
            modification_df_filtered,
            expression_df[[entity_name_col, 'log2FoldChange', 'padj']],
            on=entity_name_col,
            how='left',
            suffixes=('_mod', '_exp')
        )
        
        # Add significance column: True if both padj_mod and padj_exp < 0.05
        merged_df['all significance'] = (merged_df['padj_mod'] < 0.05) & (merged_df['padj_exp'] < 0.05) 
        
        # Only keep rows where both expression and modification are significant
        sig_df = merged_df[merged_df['all significance']].copy()
        
        # Drop rows with missing values in required columns
        sig_df = sig_df.dropna(subset=['log2FoldChange', 'mod_freq_diff'])
        
        # Count number of significant modifications per entity
        sig_mod_counts = sig_df.groupby(entity_name_col).size().reset_index(name='significant_mod_count')
        
        # Get top 30 entities with most significant modifications for cleaner visualization
        top_entities_info = sig_mod_counts.nlargest(30, 'significant_mod_count')
        top_entities = top_entities_info[entity_name_col]
        top_df = sig_df[sig_df[entity_name_col].isin(top_entities)].copy()
        
        if top_df.empty:
            logger.warning("No significant %s with modifications found for %s", entity_label_plural.lower(), modtype)
            continue 
            
        # Add modification count to the dataframe for plotting
        top_df = top_df.merge(sig_mod_counts, on=entity_name_col, how='left')
        
        # Create a more sophisticated plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a scatter plot with color representing modification count
        # Use a color gradient based on modification count
        scatter = ax.scatter(
            top_df['mod_freq_diff'],
            top_df['log2FoldChange'],
            s=100,  # Fixed point size for all points
            c=top_df['significant_mod_count'],  # Color by modification count
            cmap='viridis',  # Use viridis colormap (perceptually uniform)
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar to show modification count scale
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Number of Significant Modifications', rotation=270, labelpad=20)
        
        # Create a summary table of top entities and show it as text
        top_entities_ranked = top_entities_info.head(10)  # Show top 10 in text
        summary_text = f"Top 10 {entity_label_plural} by Modification Count:\n"
        for idx, (_, row) in enumerate(top_entities_ranked.iterrows(), 1):
            summary_text += f"{idx:2d}. {row[entity_name_col]} ({row['significant_mod_count']} mods)\n"
        
        # Add text box with top entities summary
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Annotate only the top 10 entities to avoid overcrowding
        top_10_entities = top_entities_info.head(10)[entity_name_col]
        texts = []
        for entity in top_10_entities:
            entity_data = top_df[top_df[entity_name_col] == entity]
            if not entity_data.empty:
                entity_data_copy = entity_data.copy()
                x_normalized = (entity_data_copy['mod_freq_diff'] - entity_data_copy['mod_freq_diff'].mean()) / entity_data_copy['mod_freq_diff'].std() if entity_data_copy['mod_freq_diff'].std() > 0 else 0
                y_normalized = (entity_data_copy['log2FoldChange'] - entity_data_copy['log2FoldChange'].mean()) / entity_data_copy['log2FoldChange'].std() if entity_data_copy['log2FoldChange'].std() > 0 else 0
                entity_data_copy['distance_from_origin'] = np.sqrt(x_normalized**2 + y_normalized**2)
                extreme_point = entity_data_copy.loc[entity_data_copy['distance_from_origin'].idxmax()]
                mod_count = int(extreme_point['significant_mod_count'])
                rank = list(top_10_entities).index(entity) + 1
                texts.append(ax.text(
                    extreme_point['mod_freq_diff'], extreme_point['log2FoldChange'],
                    f'#{rank} {entity}\n({mod_count} mods)',
                    fontsize=9, fontweight='bold', color='black',
                ))
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
        
        ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
        ax.set_title(f'Top 30 {entity_label_plural} with Most Significant {modtype} Modifications', fontsize=14, pad=20)
        ax.set_xlabel(f'{modtype} Frequency Difference', fontsize=12)
        ax.set_ylabel(f'Log2 Fold Change in {entity_label} Expression', fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()

        # Save the plot in a directory named after the modification type, create new directory if it does not exist
        if effective_output_dir:
            mod_output_dir = os.path.join(effective_output_dir, modtype)
            os.makedirs(mod_output_dir, exist_ok=True)
            fig.savefig(os.path.join(mod_output_dir, f'{operational_level}_top30_{entity_label_plural.lower()}_most_significant_{modtype}_modifications.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(mod_output_dir, f'{operational_level}_top30_{entity_label_plural.lower()}_most_significant_{modtype}_modifications.svg'), format='svg', bbox_inches='tight')
        figures[modtype] = fig

    return figures