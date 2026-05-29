import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from functools import reduce
import seaborn as sns
import umap
from adjustText import adjust_text
from scipy.stats import pearsonr
from matplotlib.offsetbox import AnchoredText
from dmode.utility import gtf_to_df, moddict, moddict_mapping, moddict_reverse, moddict_mapping_reverse
from tqdm import tqdm
import polars as pl
import pyranges as pr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings


# Use a more scientific color palette (e.g., seaborn's Set2)
def get_condition_palette(conditions):
    unique_conditions = sorted(set(conditions))
    # Use seaborn's tab10 palette for maximum distinguishability
    colors = sns.color_palette('tab10', n_colors=len(unique_conditions))
    palette = {cond: colors[i % len(colors)] for i, cond in enumerate(unique_conditions)}
    return palette


def move_legend_if_exists(ax, *args, **kwargs):
    """Safely reposition the legend only if the axis currently has one."""
    if ax.get_legend() is None:
        return
    sns.move_legend(ax, *args, **kwargs)


def generate_UMAP_plot(
    processed_dataframes: dict,
    modification: str,
    annotate_samples: bool,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a UMAP plot from processed sample dataframes for a specific modification code.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        modification (str): The modification type to filter for.
        annotate_samples (bool): Whether to annotate sample points on the plot.
        output_folder (str): Directory to save the resulting UMAP plot. If empty (""),
            the figure is not saved and no folder is created.

    Returns:
        matplotlib.figure.Figure: The generated UMAP figure.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name
    # Prepare metadata and sample data for UMAP
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Merge all samples on shared position tokens
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="position_token"), sample_dict.values())
    merged_df.set_index("position_token", inplace=True)

    # Transpose so samples are rows, features are columns
    X = merged_df.transpose()
    sample_names = list(X.index)

    # Get condition for each sample
    categories = [meta_dict[sample] for sample in sample_names]
    palette = get_condition_palette(categories)

    Y = pd.DataFrame(sample_names, columns=["Samples"])

    # Perform UMAP
    # Catch warnings when fitting UMAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
    embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df = pd.concat([embedding_df, Y], axis=1)

    modification_name = moddict_mapping_reverse(modification_code)  # Get human-readable name

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = sns.scatterplot(
        data=umap_df, x="UMAP1", y="UMAP2", hue=categories,
        palette=palette, s=100, edgecolor="black", linewidth=0.7, alpha=0.9, marker="o"
    )
    move_legend_if_exists(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.set_aspect("equal", adjustable="datalim")
    # Only show left and bottom axes, hide top/right, and remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')

    # Add text showing number of shared positions
    n_features = X.shape[1]
    text_box = AnchoredText(f'n = {n_features}', frameon=False, loc='upper left', pad=0.5)
    ax.add_artist(text_box)

    # Optionally annotate each sample point
    if annotate_samples:
        texts = []
        for i, sample in enumerate(sample_names):
            texts.append(ax.text(
                umap_df["UMAP1"][i], umap_df["UMAP2"][i],
                sample, fontsize=12, weight="bold", color="black", alpha=0.8
            ))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.tight_layout()

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"UMAP_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"UMAP_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )
    return fig


def generate_PCA_Plot(
    processed_dataframes: dict,
    modification: str,
    annotate_samples: bool,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a PCA plot from processed sample dataframes for a specific modification code.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        modification (str): The modification type to filter for.
        annotate_samples (bool): Whether to annotate sample points on the plot.
        output_folder (str): Directory to save the resulting PCA plot. If empty (""),
            the figure is not saved and no folder is created.

    Returns:
        matplotlib.figure.Figure: The generated PCA figure.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name

    # Prepare metadata and sample data for PCA
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Merge all samples on shared position tokens
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="position_token"), sample_dict.values())
    merged_df.set_index("position_token", inplace=True)
    # Transpose so samples are rows, features are columns
    X = merged_df.transpose()
    sample_names = list(X.index)

    # Get condition/category for each sample
    categories = [meta_dict[sample] for sample in sample_names]
    palette = get_condition_palette(categories)
    Y = pd.DataFrame(sample_names, columns=["Samples"])

    # Perform PCA
    # Catch warnings when fitting PCA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_ * 100  # as percentage
    components_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df = pd.concat([components_df, Y], axis=1)

    modification_name = moddict_mapping_reverse(modification_code)  # Get human-readable name

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue=categories,
        palette=palette, s=100, edgecolor="black", linewidth=0.7, alpha=0.9, marker="o"
    )
    move_legend_if_exists(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}% var)", fontsize=16)
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}% var)", fontsize=16)
    ax.set_aspect("equal", adjustable="datalim")
    # Only show left and bottom axes, hide top/right, and remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')

    # Add text showing number of shared positions
    n_features = X.shape[1]
    text_box = AnchoredText(f'n = {n_features}', frameon=False, loc='upper left', pad=0.5)
    ax.add_artist(text_box)

    # Optionally annotate each sample point
    if annotate_samples:
        texts = []
        for i, sample in enumerate(sample_names):
            texts.append(ax.text(
                pca_df["PC1"][i], pca_df["PC2"][i],
                sample, fontsize=12, weight="bold", color="black", alpha=0.8
            ))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.tight_layout()

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"PCA_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"PCA_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )

    return fig


def generate_modsite_barplot(
    processed_dataframes: dict,
    modification: str,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a barplot of modification site counts per sample.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        modification (str): The modification type to filter for.
        output_folder (str): Directory to save the resulting barplot. If empty (""),
            the figure is not saved and no folder is created.

    Returns:
        matplotlib.figure.Figure: The generated barplot figure.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name

    # Prepare metadata and sample data
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Get number of modification sites per sample
    modsite_counts = {sample: df.shape[0] for sample, df in sample_dict.items()}
    modsite_df = pd.DataFrame(list(modsite_counts.items()), columns=["Sample", "Num_Modification_Sites"])
    modsite_df["Condition"] = modsite_df["Sample"].map(meta_dict)

    modification_name = moddict_mapping_reverse(modification_code)

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    # Use consistent palette for barplot
    palette = get_condition_palette(modsite_df["Condition"].tolist())
    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = sns.barplot(
            data=modsite_df, x="Sample", y="Num_Modification_Sites", hue="Condition",
            palette=palette, edgecolor="black", linewidth=0.7
        )
    move_legend_if_exists(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Sample", fontsize=16)
    ax.set_ylabel("Number of Modification Sites", fontsize=16)
    # Only show left and bottom axes, hide top/right, and remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"Modsite_Barplot_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"Modsite_Barplot_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )

    return fig


def generate_modification_distribution_plot(
    processed_dataframes: dict,
    modification: str,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a KDE plot of modification frequency distributions per condition.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        modification (str): The modification type to filter for.
        output_folder (str): Directory to save the resulting plot. If empty (""),
            the figure is not saved and no folder is created.

    Returns:
        matplotlib.figure.Figure: The generated distribution figure.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name

    # Prepare metadata and sample data
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    modification_name = moddict_mapping_reverse(modification_code)  # Get human-readable name

    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.gca()
    # Color code by condition, not sample
    condition_palette = get_condition_palette(list(set(meta_dict.values())))
    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sample, df in sample_dict.items():
            mod_values = df[sample]
            mod_values = mod_values[mod_values > 0]
            condition = meta_dict[sample]
            if len(mod_values) > 1:
                sns.kdeplot(mod_values, ax=ax, label=condition, color=condition_palette.get(condition, None), linewidth=2, alpha=0.7, clip=(0, 100))
    ax.set_xlabel(f"{modification_name} frequency", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_xlim(0, 100)
    # Only show unique conditions in legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Condition", fontsize=10)
    # Only show left and bottom axes, hide top/right, and remove box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')
    plt.tight_layout()

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"Mod_Distribution_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"Mod_Distribution_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )

    return fig


def generate_modification_violin_plot(
    processed_dataframes: dict,
    modification: str,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a violin plot of modification levels for each condition.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        modification (str): The modification type to filter for.
        output_folder (str): Directory to save the resulting plot. If empty (""),
            the figure is not saved and no folder is created.

    Returns:
        matplotlib.figure.Figure: The generated violin plot figure.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name
    meta_dict = {}
    sample_dict = {}
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            meta_dict[sample] = condition
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Prepare data for violin plot: melt all samples into one DataFrame with columns: Sample, Condition, Value
    violin_data = []
    for sample, df in sample_dict.items():
        mod_values = df[sample]
        mod_values = mod_values[mod_values > 0]
        condition = meta_dict[sample]
        for value in mod_values:
            violin_data.append({"Sample": sample, "Condition": condition, "Modification Level": value})
    violin_df = pd.DataFrame(violin_data)

    modification_name = moddict_mapping_reverse(modification_code)  # Get human-readable name

    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 10, "axes.titlesize": 12, "legend.fontsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.gca()
    # Plot each sample as a violin, color by condition
    condition_palette = get_condition_palette(violin_df["Condition"].unique())
    # Map sample to color by its condition
    sample_colors = [condition_palette[meta_dict[sample]] for sample in violin_df["Sample"].unique()]
    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.violinplot(
            data=violin_df,
            x="Sample",
            y="Modification Level",
            palette=sample_colors,
            ax=ax,
            cut=0,
            inner="box",
            linewidth=1.2
        )
    ax.set_xlabel("Sample", fontsize=16)
    ax.set_ylabel(f"{modification_name} frequency", fontsize=16)
    # Add a legend for conditions
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=cond) for cond, color in condition_palette.items()]
    ax.legend(handles=legend_handles, title="Condition", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')
    plt.tight_layout()

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"Mod_Violin_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"Mod_Violin_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )

    return fig


def generate_sample_to_sample_correlation_plot(
    processed_dataframes: dict,
    modification: str,
    output_folder: str = "",
    operational_level: str = "genome"
):
    """
    Generate a sample-to-sample correlation heatmap for modification ratios.

    Creates a correlation heatmap showing Pearson correlation coefficients
    between all samples based on their modification frequencies at shared positions.

    Parameters
    ----------
    processed_dataframes : dict
        Nested dict of condition -> sample -> DataFrame.
    modification : str
        The modification type to filter for.
    output_folder : str
        Directory to save the resulting correlation heatmap. If empty (""), the
        figure is not saved and no folder is created.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The clustermap object (its underlying figure is accessible via `.fig`).

    Notes
    -----
    Useful for quality control and identifying sample outliers or batch effects.
    Only shared modification positions across all samples are used for correlation.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name

    # Prepare metadata and sample data
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Merge all samples on shared position tokens
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="position_token"), sample_dict.values())
    merged_df.set_index("position_token", inplace=True)

    # Calculate correlation matrix
    correlation_matrix = merged_df.corr()

    modification_name = moddict_mapping_reverse(modification_code)

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})

    # Prepare condition colors for rows/columns and add legend
    categories = [meta_dict[sample] for sample in correlation_matrix.index]
    condition_palette = get_condition_palette(categories)
    # Map each sample to its condition color (list aligned with matrix index order)
    row_colors = [condition_palette[meta_dict[sample]] for sample in correlation_matrix.index]

    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = sns.clustermap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            figsize=(10, 8),
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            dendrogram_ratio=0.15,
            cbar_pos=(0.02, 0.83, 0.03, 0.15),
            row_colors=row_colors,
            col_colors=row_colors
        )

    # Add a legend for condition colors next to the heatmap
    from matplotlib.patches import Patch
    # Preserve the palette order when creating legend entries
    legend_handles = [Patch(facecolor=condition_palette[cond], label=cond) for cond in condition_palette.keys()]
    try:
        ax = fig.ax_heatmap
        ax.legend(handles=legend_handles, title="Condition", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    except Exception:
        # If fig layout differs, attach legend to the figure as a fallback
        fig.fig.legend(handles=legend_handles, title="Condition", bbox_to_anchor=(0.98, 0.9), loc='center left', fontsize=10)

    fig.fig.suptitle(f'Sample-to-Sample Correlation ({modification_name})', fontsize=18, y=1.02)

    # Add text showing number of shared positions
    n_positions = merged_df.shape[0]
    fig.ax_heatmap.text(0.02, 0.98, f'n = {n_positions}', transform=fig.ax_heatmap.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.0))

    # Save the plot only if an output folder was provided
    if output_folder:
        mod_folder = os.path.join(output_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig.savefig(
            os.path.join(mod_folder, f"Sample_Correlation_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(mod_folder, f"Sample_Correlation_{modification_name}.png"),
            format="png",
            bbox_inches="tight"
        )

    return fig


def barplot_intersections_database(
    processed_dataframes: dict,
    rmbase_database: dict,
    modification: str,
    output_folder: str = ""
):
    """
    Generate barplots of modification site intersections with a reference database (e.g., RMBase).

    Two figures are produced: one showing the absolute number of intersecting sites,
    and one showing the proportion of intersecting sites.

    Parameters:
        processed_dataframes (dict): Nested dict of condition -> sample -> DataFrame.
        rmbase_database (dict): key: modification, value: Path to the RMBase database bedfile.
        modification (str): The modification type to filter for.
        output_folder (str): Directory to save the resulting barplots. If empty (""),
            the figures are not saved and no folder is created.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.figure.Figure] | None:
            (count_figure, proportion_figure), or None if the modification is not in
            the RMBase database.
    """
    base_modification_type_dict = moddict()
    modification_code = moddict_mapping(modification)  # Convert to internal code if given a name

    # Load RMBase database
    try:
        rmbase_df = pd.read_csv(rmbase_database[modification_code], sep="\t")
        rmbase_df = rmbase_df[rmbase_df.iloc[:, 6] == moddict_mapping_reverse(modification_code)]
    except KeyError:
        print(f"No RMBase database entry for modification code {modification_code}. Skipping intersection analysis.")
        return None

    # Create position token in RMBase dataframe for intersection by using column indexes
    rmbase_df["position_token"] = (
        rmbase_df.iloc[:, 0].astype(str) + ":" +
        rmbase_df.iloc[:, 1].astype(str) + ":" +
        rmbase_df.iloc[:, 2].astype(str) + ":" +
        rmbase_df.iloc[:, 5].astype(str) + ":" +
        "a"
    )

    rmbase_positions = set(rmbase_df["position_token"].astype(str).tolist())
    # Prepare metadata and sample data for intersection analysis
    meta_dict = {}  # Maps sample name to condition
    sample_dict = {}  # Maps sample name to formatted DataFrame
    for condition in processed_dataframes:
        for sample in processed_dataframes[condition]:
            # Map each sample to its condition
            meta_dict[sample] = condition
            # Filter for the specified modification code
            sample_df = processed_dataframes[condition][sample]
            sample_df_mod = sample_df[sample_df["modified base code"] == modification_code]
            # Keep only position and fraction modified, rename for merging
            sample_df_formatted = sample_df_mod[["position_token", "fraction modified"]].copy()
            sample_df_formatted.columns = ["position_token", sample]
            sample_dict[sample] = sample_df_formatted

    # Calculate intersections and proportions with RMBase for each sample
    intersection_counts = {}
    proportion_counts = {}
    total_counts = {}
    for sample, df in sample_dict.items():
        mod_positions = set(df["position_token"].astype(str).tolist())
        intersection = mod_positions.intersection(rmbase_positions)
        intersection_counts[sample] = len(intersection)
        total_counts[sample] = len(mod_positions)
        proportion_counts[sample] = len(intersection) / len(mod_positions) if len(mod_positions) > 0 else 0

    intersection_df = pd.DataFrame({
        "Sample": list(intersection_counts.keys()),
        "Num_Intersecting_Sites": list(intersection_counts.values()),
        "Proportion_Intersecting": list(proportion_counts.values())
    })
    intersection_df["Condition"] = intersection_df["Sample"].map(meta_dict)

    modification_name = moddict_mapping_reverse(modification_code)

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})

    # ---- Figure 1: Number of intersecting sites ----
    fig_count = plt.figure(figsize=(8, 6), facecolor="white")
    palette = get_condition_palette(intersection_df["Condition"].tolist())
    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = sns.barplot(
            data=intersection_df, x="Sample", y="Num_Intersecting_Sites", hue="Condition",
            palette=palette, edgecolor="black", linewidth=0.7
        )
    move_legend_if_exists(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Sample", fontsize=16)
    ax.set_ylabel("Number of Intersecting Sites with RMBase", fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_facecolor('white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # ---- Figure 2: Proportion of intersecting sites ----
    fig_prop = plt.figure(figsize=(8, 6), facecolor="white")
    # Catch warnings during plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax2 = sns.barplot(
            data=intersection_df, x="Sample", y="Proportion_Intersecting", hue="Condition",
            palette=palette, edgecolor="black", linewidth=0.7
        )
    move_legend_if_exists(ax2, "upper left", bbox_to_anchor=(1, 1))
    ax2.set_xlabel("Sample", fontsize=16)
    ax2.set_ylabel("Proportion of Intersecting Sites with RMBase", fontsize=16)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_linewidth(1.0)
    ax2.spines['bottom'].set_linewidth(1.0)
    ax2.set_facecolor('white')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plots only if an output folder was provided
    if output_folder:
        stats_folder = os.path.join(output_folder, "gene_basic_statistics")
        mod_folder = os.path.join(stats_folder, modification_name)
        os.makedirs(mod_folder, exist_ok=True)
        fig_count.savefig(
            os.path.join(mod_folder, f"Modsite_RMBase_Intersection_{modification_name}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight"
        )
        fig_count.savefig(
            os.path.join(mod_folder, f"Modsite_RMBase_Intersection_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )
        fig_prop.savefig(
            os.path.join(mod_folder, f"Modsite_RMBase_Intersection_Proportion_{modification_name}.png"),
            format="png",
            bbox_inches="tight"
        )
        fig_prop.savefig(
            os.path.join(mod_folder, f"Modsite_RMBase_Intersection_Proportion_{modification_name}.svg"),
            format="svg",
            bbox_inches="tight"
        )
    return fig_count, fig_prop