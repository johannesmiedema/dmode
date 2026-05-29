import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import fisher_exact
from adjustText import adjust_text
from dmode.utility import convert_bed_to_df, gtf_to_df
import numpy as np
from math import log, exp, isinf
import statsmodels.api as sm
from scipy.stats import f, chi2, rankdata
from tqdm import tqdm
from dmode.dmode import DmodE
import math
import warnings
from typing import Optional




def _append_positions_column(df: pd.DataFrame, chrom_column: str = "chrom") -> pd.DataFrame:
    """Ensure a positions column exists for annotation joins."""
    if df is None or df.empty:
        return df
    required_cols = {chrom_column, "start", "end", "strand", "mod_type"}
    if not required_cols.issubset(df.columns):
        return df
    df = df.copy()
    df["positions"] = (
        df[chrom_column].astype(str)
        + ":"
        + df["start"].astype(str)
        + ":"
        + df["end"].astype(str)
        + ":"
        + df["strand"].astype(str)
        + ":"
        + df["mod_type"].astype(str)
    )
    return df


def _prepare_gene_annotation_lookup(dmode_obj, gtf_file, alpha, output_folder):
    """Create per-comparison gene annotation tables for volcano labeling."""
    annotations = {}
    gtf_df = gtf_to_df(gtf_file)
    gtf_df = gtf_df[gtf_df['feature'] == 'gene']

    for comparison_key in dmode_obj.diff_gene_mod_comparisons_dict.keys():
        cond_level,reference_level = comparison_key.split("_VS_")
        annotation_df = diff_gene_mod_cond_extract_significant_genes(
            dmode_obj=dmode_obj,
            gtf_df=pd.DataFrame(gtf_df),
            ref_level=reference_level,
            cond_level=cond_level,
            alpha=alpha,
            genes_only=True,
            output_folder=output_folder,
        )
        annotations[comparison_key] = _append_positions_column(annotation_df)
    return annotations


def _compose_volcano_label(row, fallback_label, identifier_groups):
    """Return fallback label augmented with first available identifier set."""
    for fields in identifier_groups:
        values = []
        for field in fields:
            if field in row.index and pd.notna(row[field]):
                text_value = str(row[field]).strip()
                if text_value:
                    if text_value not in values:
                        values.append(text_value)
        if values:
            identifier = " / ".join(values)
            return f"{identifier} ({fallback_label})"
    return fallback_label


def _collapse_annotation_values(values):
    """Collapse annotation values while keeping order and removing empties."""
    cleaned = [str(val).strip() for val in values if pd.notna(val) and str(val).strip()]
    if not cleaned:
        return np.nan
    unique_ordered = list(dict.fromkeys(cleaned))
    return "; ".join(unique_ordered)


def _merge_annotation_columns(
    data_df: pd.DataFrame,
    annotation_df: Optional[pd.DataFrame],
    key_column: str = "positions",
    value_columns: Optional[list] = None,
):
    """Merge aggregated annotation columns into plotting dataframe."""
    if annotation_df is None or annotation_df.empty:
        return data_df
    if key_column not in data_df.columns or key_column not in annotation_df.columns:
        return data_df
    if value_columns is None:
        value_columns = [col for col in annotation_df.columns if col != key_column]
    available_values = [col for col in value_columns if col in annotation_df.columns]
    if not available_values:
        return data_df

    subset_cols = [key_column] + available_values
    aggregated = (
        annotation_df[subset_cols]
        .groupby(key_column, dropna=False)
        .agg({col: _collapse_annotation_values for col in available_values})
        .reset_index()
    )

    rename_map = {}
    for col in available_values:
        rename_map[col] = f"{col}__annot" if col in data_df.columns else col
    aggregated = aggregated.rename(columns=rename_map)

    merged_df = data_df.copy()
    merged_df = merged_df.merge(aggregated, on=key_column, how="left")

    for col in available_values:
        annot_col = rename_map[col]
        if col in data_df.columns:
            merged_df[col] = merged_df[col].combine_first(merged_df[annot_col])
            merged_df.drop(columns=annot_col, inplace=True)

    return merged_df

def logReg(
    dmode_obj,
    counts_modified_sites,
    counts_unmodified_sites,
    treatment,
    vars_df,
    overdispersion="MN",
    effect="predicted",
    test="F",
    id=None,
    ref_level=None,
    cond_level=None
):
    """
    Fit a logistic regression model for modification data using GLM with a binomial family.

    Parameters
    ----------
    counts_modified_sites : list or array-like of int
        Counts of modified sites (e.g., methylated reads) for each sample.
    counts_unmodified_sites : list or array-like of int
        Counts of unmodified sites (e.g., unmethylated reads) for each sample.
    treatment : list or array-like
        Treatment group labels corresponding to each sample.
        Used as the main explanatory variable.
    vars_df : pandas.DataFrame or None
        Optional additional covariates to include in the regression model.
        If None or empty, only treatment is used.
    overdispersion : {"none", "MN"}
        Method for handling overdispersion:
            - "none" : assume no overdispersion (phi = 1).
            - "MN"   : estimate overdispersion using Pearson residuals
                    and ensure phi ≥ 1.
            # (future: "shrinkMN" could be supported)
    effect : {"wmean", "mean", "predicted"}
        Method for computing effect size between groups:
            - "wmean" : weighted mean methylation per group.
            - "mean" : simple average methylation proportion per group.
            - "predicted" : mean fitted value (from model prediction) per group.
    test : {"F", "Chisq"}
        Type of statistical test for deviance:
            - "F" : F-test (only if overdispersion phi > 1).
            - "Chisq" : Chi-squared test.
    id : str
        Identifier of the modification site of interest.

    Returns
    -------
    result : dict
        Dictionary with keys:
            - "mod_freq_diff" : float
                Difference in methylation percentage (group 2 – group 1).
            - "p.value" : float
                P-value from the chosen test (F or Chi-squared).
            - "q.value" : float
                Same as p-value (placeholder for multiple testing correction).
            - "id" : str
                Identifier passed to the function.
            - "meth_<group>" : float
                Group-specific methylation percentage estimates, where `<group>`
                corresponds to treatment labels.

    Notes
    -----
    - The model is fit using `statsmodels.GLM` with a binomial family,
    weights equal to total read counts per sample.
    - Overdispersion is handled by adjusting the test statistic with `phi`.
    - For >2 treatment groups, `mod_freq_diff` is defined as the difference
    between the maximum and minimum methylation across groups.
    - Intended for analyzing site-level methylation (or other binary event counts)
    between conditions.
    """
    counts = np.array(counts_modified_sites + counts_unmodified_sites)
    treatment = np.array(treatment)
    # correct counts and treatment for NAs in counts

    number_of_all_reads = counts[: len(counts) // 2] + counts[len(counts) // 2 :]
    number_of_modified_reads = counts[: len(counts) // 2]
    proportion_mod_by_all = number_of_modified_reads / number_of_all_reads
    counts = list(counts)
    treatment = list(treatment)
    # full model formula
    if vars_df is None or vars_df.empty:
        vars_combined = pd.DataFrame({"treatment": treatment})
    else:
        vars_combined = pd.concat(
            [
                pd.Series(treatment, name="treatment"),
                vars_df.reset_index(drop=True),
            ],
            axis=1
        )
    vars_combined_dummies = pd.get_dummies(vars_combined, drop_first=True)
    modelMat = sm.add_constant(vars_combined_dummies)
    modelMat = modelMat.astype(float)
    # Catch warnings when fitting the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_full = sm.GLM(proportion_mod_by_all, modelMat, family=sm.families.Binomial(), var_weights=number_of_all_reads)
        logReg_obj = glm_full.fit()
        fitted_values = logReg_obj.fittedvalues
        number_of_parameters = len(logReg_obj.params)
    
    if vars_combined.shape[1] > 1:
        covariat_matrix = sm.add_constant(
            pd.get_dummies(vars_combined.iloc[:, 1:], drop_first=True)
        )
        covariat_matrix = covariat_matrix.astype(float)
        objCov = sm.GLM(proportion_mod_by_all, covariat_matrix, family=sm.families.Binomial(), var_weights=number_of_all_reads)
        objCov = objCov.fit()
        deviance = objCov.deviance - logReg_obj.deviance
        ddf = objCov.df_resid - logReg_obj.df_resid
    else:
        deviance = logReg_obj.null_deviance - logReg_obj.deviance
        ddf = (logReg_obj.nobs - 1) - logReg_obj.df_resid

    # compute overdispersion
    if overdispersion == "none":
        phi = 1
    else:
        uresids = (number_of_modified_reads - number_of_all_reads * fitted_values) / np.sqrt(fitted_values * (number_of_all_reads - number_of_all_reads * fitted_values))
        phi_raw = np.sum(uresids**2) / (len(number_of_all_reads) - number_of_parameters)
        if overdispersion == "MN":
            phi = max(phi_raw, 1)
    # elif overdispersion == "shrinkMN":
    #     df_prior = parShrinkMN.get("df.prior", 0)
    #     var_prior = parShrinkMN.get("var.prior", 1)
    #     df_total = (len(number_of_all_reads)-number_of_parameters) + df_prior
    #     phi = max(((len(number_of_all_reads)-number_of_parameters) * phi_raw + df_prior*var_prior)/df_total, 1)

    if test == "F" and phi > 1:
        test_type = "F"
    else:
        test_type = "Chisq"

    # p-value calculation
    if test_type == "F":
        p_value = f.sf(deviance / phi, ddf, len(number_of_all_reads) - number_of_parameters)
    else:
        p_value = chi2.sf(deviance / phi, 1)

    if np.isnan(p_value):
        p_value = 1

    # effect size calculation
    if effect == "wmean":
        modifications = (
            pd.DataFrame({"number_of_modified_reads": number_of_modified_reads, "number_of_all_reads": number_of_all_reads, "treatment": treatment})
            .groupby("treatment")
            .sum()
        )
        modifications = modifications["number_of_modified_reads"] / modifications["number_of_all_reads"]
    elif effect == "mean":
        modifications = pd.Series(proportion_mod_by_all).groupby(treatment).mean()
    elif effect == "predicted":
        modifications = pd.Series(fitted_values).groupby(treatment).mean()

    # difference
    if len(np.unique(treatment)) > 2:
        mod_freq_diff = modifications.max() - modifications.min()
    else:
        mod_freq_diff = float(modifications[cond_level]) - float(modifications[ref_level])

    result = {"mod_freq_diff": 100 * mod_freq_diff, "pvalue": p_value, "id": id}
    for name, val in modifications.items():
        result[f"mod_freq_{name}"] = 100 * val
    return result

def fisher_exact_test(
    dmode_obj, table: np.array, id: str, ref_level: str, cond_level: str
):
    """
    Perform Fisher's exact test on a 2x2 contingency table and compute methylation difference.

    Parameters
    ----------
    table : np.ndarray of shape (2, 2)
        A 2x2 contingency table with:
            - Rows representing conditions (e.g., Condition A, Condition B)
            - Columns representing counts (e.g., modified vs. unmodified)
    id : str
        Identifier for the modification site of interest.

    Returns
    -------
    dict
        Dictionary containing:
            - "mod_freq_diff" : float
                Difference in modification percentage between Condition B and Condition A
                (in percentage points).
            - "oddsratio" : float
                Odds ratio from Fisher's exact test.
            - "p-value" : float
                P-value from Fisher's exact test.
            - "id" : str
                Passed identifier of the modification site.

    Notes
    -----
    - The function assumes `table` is exactly 2x2.
    - `mod_freq_diff` is computed as:

    (modified_sample / total_sample - modified_ctrl / total_ctrl) * 100
    """
    # Table must have format
    # Perform Fisher's exact test
    oddsratio, p_value = fisher_exact(table)
    mod_freq_diff = (
        (table[1, 0] / table[1, :].sum()) - (table[0, 0] / table[0, :].sum())
    ) * 100
    return {
        "mod_freq_diff": mod_freq_diff,
        # "oddsratio":oddsratio,
        "pvalue": p_value,
        "id": id,
        f"mod_freq_{ref_level}": (table[1, 0] / table[1, :].sum()) * 100,
        f"mod_freq_{cond_level}": (table[0, 0] / table[0, :].sum()) * 100,
    }


def QValuesfun(pvals, pi0):
            number_of_pvalues_ = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_pvals = pvals[sorted_idx]
            qvals = pi0 * sorted_pvals * number_of_pvalues_ / np.arange(1, number_of_pvalues_ + 1)
            qvals = np.minimum.accumulate(qvals[::-1])[::-1]
            qvals_out = np.empty_like(qvals)
            qvals_out[sorted_idx] = qvals
            return qvals_out

def SLIMfunc(dmode_obj, raw_pvalues, STA=0.1, Divi=10, Pz=0.05, B=100, Bplot=True,output_folder=""):
    """
    Perform SLIM (Significance Level-based Iterative Method) multiple testing correction.
    
    This method implements a more sophisticated alternative to Benjamini-Hochberg
    correction by estimating the proportion of true null hypotheses (π₀) and
    adjusting the false discovery rate accordingly.
    
    Parameters
    ----------
    raw_pvalues : array-like
        List/array of raw p-values to be corrected.
    STA : float, default 0.1
        Starting threshold for π₀ estimation.
    Divi : int, default 10
        Number of divisions for linear fitting in π₀ estimation.
    Pz : float, default 0.05
        Target false discovery rate threshold.
    B : int, default 100
        Number of quantiles to explore for optimization.
    Bplot : bool, default False
        Whether to generate diagnostic plots.
        
    Returns
    -------
    np.ndarray
        Array of adjusted p-values (q-values) in the same order as input.
        
    Notes
    -----
    The SLIM method:
    1. Estimates π₀ (proportion of true nulls) using linear regression
    2. Searches over quantiles to find optimal π₀ estimate
    3. Computes q-values using the estimated π₀
    4. Controls FDR at the specified level (Pz)
    
    This method can be more powerful than standard Benjamini-Hochberg when
    the proportion of true alternatives is high.
    """
    raw_pvalues = np.array(raw_pvalues)
    number_of_pvalues = len(raw_pvalues)

    # -------------------------
    # 1. Observed points
    lambda_ga = np.arange(0, 1.001, 0.001)

    def f1(lambda_, raw_pvalues):
        return np.sum(raw_pvalues < lambda_) / len(raw_pvalues)

    gamma_ga = np.array([f1(lambda_i, raw_pvalues) for lambda_i in lambda_ga])
    #Gamma = gamma_ga.copy()
    #alpha_mtx = gamma_ga[np.where(lambda_ga == 0.05)]

    # -------------------------
    # 2. Estimation of pi0 by linear model
    pi0_mtx = []
    itv = (1 - STA) / Divi
    for i in range(1, Divi + 1):
        cutoff = STA + (i / Divi) * (1 - STA)
        lambda_seq = np.linspace(cutoff - itv, cutoff, int(10) + 1)
        gamma_seq = np.array([f1(l, raw_pvalues) for l in lambda_seq])
        # simple linear regression
        slope, intercept = np.polyfit(lambda_seq, gamma_seq, 1)
        pi0_mtx.append(slope)

    # -------------------------
    # 3. Searching over quantiles
    if B <= 1:
        B = 100
    quapoint_mtx = np.linspace(0.01, 0.99, B)

    pi0s_est_COM = []
    P_pi1_mtx = []
    pi1_act_mtx = []
    n_com = []
    Num_mtx = []
    maxFDR_mtx = []

    for qua_point in quapoint_mtx:
        pi0_combLR = min(np.quantile(pi0_mtx, qua_point), 1)
        pi0_est = pi0_combLR

        pi0s_est_COM.append(pi0_est)

        # condition for P_pi1
        idx = max(int(len(raw_pvalues) * (1 - pi0_est)), 1) - 1
        P_pi1 = np.sort(raw_pvalues)[idx]
        P_pi1_mtx.append(P_pi1)

        maxFDR = Pz * pi0_est / (1 - (1 - Pz) * pi0_est)
        maxFDR_mtx.append(maxFDR)

        # compute q-values
        # simple BH-like q-value approximation
        qvalues_combLR = QValuesfun(raw_pvalues, pi0_est)
        #print(qvalues_combLR)
        selected = np.where(qvalues_combLR < maxFDR)[0]
        n_com.append(selected)
        Num_mtx.append(len(selected))
        pi1_act_mtx.append(len(selected) / len(raw_pvalues))

    # -------------------------
    # 4. Judging by max FDR
    pi1s_est_COM = 1 - np.array(pi0s_est_COM)
    Diff = np.sum(raw_pvalues <= Pz) / len(raw_pvalues) - np.array(pi1_act_mtx)

    loc = np.argmin(np.abs(Diff))
    Diff_loc = Diff[loc]
    selQuantile = quapoint_mtx[loc]
    pi0_Est = min(1, pi0s_est_COM[loc])
    maxFDR_Pz = Pz * pi0_Est / (1 - (1 - Pz) * pi0_Est)

    # -------------------------
    # 5. Optional plotting
    if Bplot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(raw_pvalues, bins=50)
        plt.title("Histogram of p-values")

        plt.subplot(1, 2, 2)
        plt.plot(lambda_ga, gamma_ga, label="CPD of p-value")
        qValues = QValuesfun(raw_pvalues, pi0_Est)
        gammaq_ga = np.array([f1(l, qValues) for l in lambda_ga])
        plt.plot(lambda_ga, gammaq_ga, "--", color="blue", label="CPD of q-value")
        plt.axvline(Pz, color="black", linestyle=":", label=f"Pmax={Pz}")
        plt.axvline(
            maxFDR_Pz, color="blue", linestyle=":", label=f"FDRmax={maxFDR_Pz:.2f}"
        )
        plt.title("Relationship of p- and q-value")
        plt.legend()

        #plt.show()
        if output_folder != "":
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(f"{output_folder}/Bplot_p_and_qvalues.png")
            plt.savefig(f"{output_folder}/Bplot_p_and_qvalues.svg")

    #m = len(raw_pvalues)
    #sorted_idx = np.argsort(raw_pvalues)
    #sorted_pvals = raw_pvalues[sorted_idx]

    #qvals = pi0_Est * sorted_pvals * m / np.arange(1, m + 1)
    #qvals = np.minimum.accumulate(qvals[::-1])[::-1]  # ensure monotonicity
    qvals = QValuesfun(raw_pvalues, pi0_Est)
    padj = qvals

    #padj = np.empty_like(qvals)
    #padj[sorted_idx] = qvals
    return padj

def benjamini_hochberg(dmode_obj, pvals):
    """
    Perform Benjamini-Hochberg multiple testing correction.
    
    This method implements the standard Benjamini-Hochberg procedure for
    controlling the false discovery rate (FDR) in multiple hypothesis testing.

    Parameters
    ----------
    pvals : array-like
        List or array of raw p-values to be corrected.

    Returns
    -------
    np.ndarray
        Array of adjusted p-values (FDR-corrected) in the same order as input.
        
    Notes
    -----
    The Benjamini-Hochberg procedure:
    1. Sort p-values in ascending order
    2. For the i-th smallest p-value, compute: p_i * m / i
    3. Enforce monotonicity by taking cumulative minimum from largest to smallest
    4. Return adjusted p-values in original order
    
    This is the most commonly used FDR correction method and is appropriate
    when you want to control the expected proportion of false discoveries.
    """
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH adjustment
    adjusted = sorted_pvals * n / (np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]  # ensure monotonicity

    # Return in original order
    bh_pvals = np.empty_like(adjusted)
    bh_pvals[sorted_idx] = adjusted
    return bh_pvals


def outer_join(df1:pd.DataFrame, df2:pd.DataFrame, sample2:str):
    if df1.empty:
        df_merged = df2[["position_token","Nvalid_cov","Nmod","fraction modified"]].sort_values(by="position_token", ascending=True)
        df_merged.columns = ["position_token",f"coverage_{sample2}",f"n_modified_{sample2}",f"fraction_modified_{sample2}"]
        df_merged = df_merged.sort_values(by="position_token", ascending=True)
    else:
        df1 = df1.sort_values(by="position_token", ascending=True)
        df2_short = df2[["position_token","Nvalid_cov","Nmod","fraction modified"]]
        df2_short.columns = ["position_token",f"coverage_{sample2}",f"n_modified_{sample2}",f"fraction_modified_{sample2}"]
        df2_short = df2_short.sort_values(by="position_token", ascending=True)
        df_merged = pd.merge(df1,df2_short, how="outer", on="position_token")
        df_merged = df_merged.sort_values(by="position_token", ascending=True)
    return df_merged


def position_token_conversion(position_token, base_modification_type_dict):
    token_split = position_token.split(":")
    chrom = token_split[0]
    start = token_split[1]
    end = token_split[2]
    strand = token_split[3]
    try:
        mod_type = base_modification_type_dict[str(token_split[4])]
        positions = f"{chrom}:{start}:{end}:{strand}:{mod_type}"
    except KeyError:
        mod_type = token_split[4]
        positions = f"{chrom}:{start}:{end}:{strand}:{mod_type}"
    return chrom, start, end, strand, mod_type, positions
                    
def filter_and_sort(df, intersecting_positions):
            df = df[df["position_token"].isin(intersecting_positions)]
            df = df.sort_values(by="position_token", ascending=True)
            return df

### Genome based differential modification analysis ###

def diff_gene_mod_cond_extraction(dmode_obj, correction_method="SLIM", output_folder=""):
    """
    Perform differential modification analysis between all specified comparisons.
    
    This is the main analysis method that compares modification levels between
    reference and condition groups for all overlapping genomic positions.
    Statistical testing is performed using either Fisher's exact test (for 
    single replicates) or logistic regression (for multiple replicates).
    
    Parameters
    ----------
    correction_method : {"SLIM", "bh"}, default "SLIM"
        Method for multiple testing correction:
        - "SLIM" : Use SLIM method for FDR control
        - "bh" : Use Benjamini-Hochberg correction
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    dict
        Dictionary mapping comparison names (format: "cond_VS_ref") to 
        pandas DataFrames containing:
        - Genomic coordinates (chrom, start, end, strand, mod_type)
        - Statistical test results (pvalue, padj, mod_freq_diff)
        - Condition-specific modification frequencies
        - Negative log10 adjusted p-values for visualization
        
    Notes
    -----
    - Only positions present in ALL samples of both conditions are analyzed
    - For single replicates per condition: Fisher's exact test is used
    - For multiple replicates: Logistic regression with binomial GLM is used
    - Results are saved as tab-separated files if output_folder is specified
    - Progress is shown with tqdm progress bars
    
    The method updates dmode_obj.diff_gene_mod_comparisons_dict with results.
    """
    comparisons_dict = {}
    outer_join_comparisons_dict = {}
    base_modification_type_dict={
        "a":"m6A",
        "69426": "Am",
        "17596": "Ino",
        "17802": "pseU",
        "19227": "Um",
        "19229": "Gm",
        "19228": "Cm",
        "m": "m5C"
    }
    for idx, (cond_level, ref_level) in enumerate(dmode_obj.diff_gene_mod_comparisons):
        output_df = pd.DataFrame()
        reference_dfs = [df for df in list(dmode_obj.diff_gene_mod_dataframe_condition_dict[ref_level].values())]
        condition_dfs = [df for df in list(dmode_obj.diff_gene_mod_dataframe_condition_dict[cond_level].values())]
        
        outer_join_df = pd.DataFrame()
        for df, sample in zip(dmode_obj.diff_gene_mod_dataframe_condition_dict[ref_level].values(),dmode_obj.diff_gene_mod_dataframe_condition_dict[ref_level].keys()):
            outer_join_df = outer_join(df1 = outer_join_df,df2 =  df,sample2 = sample)
        
        for df, sample in zip(dmode_obj.diff_gene_mod_dataframe_condition_dict[cond_level].values(),dmode_obj.diff_gene_mod_dataframe_condition_dict[cond_level].keys()):
            outer_join_df = outer_join(df1 = outer_join_df,df2 =  df,sample2 = sample)
            
        outer_join_df = outer_join_df.fillna(value = 0)
        sorted_columns = sorted(list(outer_join_df.columns))
        outer_join_df = outer_join_df[sorted_columns]
        position_list = []
        for entry in outer_join_df["position_token"]:
            chrom, start, end, strand, mod_type, positions = position_token_conversion(position_token=entry,base_modification_type_dict=base_modification_type_dict)
            position_list.append(positions)
        outer_join_df["positions"] = position_list

        intersection_lists = []
        for df_element in reference_dfs + condition_dfs:
            intersection_lists.append(df_element["position_token"])
        intersecting_position_tokens = set(intersection_lists[0]).intersection(
            *intersection_lists[1:]
        )
        
        def filter_and_sort(df, intersecting_positions):
            df = df[df["position_token"].isin(intersecting_positions)]
            df = df.sort_values(by="position_token", ascending=True)
            return df

        reference_dfs = [
            filter_and_sort(df, intersecting_position_tokens)
            for df in list(dmode_obj.diff_gene_mod_dataframe_condition_dict[ref_level].values())
        ]
        condition_dfs = [
            filter_and_sort(df, intersecting_position_tokens)
            for df in list(dmode_obj.diff_gene_mod_dataframe_condition_dict[cond_level].values())
        ]
        intersecting_position_tokens = sorted(intersecting_position_tokens)
        


        results_list = []
        for idx, position_token in tqdm(
            enumerate(intersecting_position_tokens),
            total=len(intersecting_position_tokens),
        ):
            if len(reference_dfs) == 1 and len(condition_dfs) == 1:
                ref_temp_df = reference_dfs[0].iloc[idx, :].to_dict()
                cond_temp_df = condition_dfs[0].iloc[idx, :].to_dict()
                ref_modified_counts = ref_temp_df["Nmod"]
                ref_unmodifed_counts = (
                    ref_temp_df["Nvalid_cov"] - ref_temp_df["Nmod"]
                )
                cond_modified_counts = cond_temp_df["Nmod"]
                cond_unmodified_counts = (
                    cond_temp_df["Nvalid_cov"] - cond_temp_df["Nmod"]
                )
                fisher_exact_test_table = np.array(
                    [
                        [ref_modified_counts, ref_unmodifed_counts],
                        [cond_modified_counts, cond_unmodified_counts],
                    ]
                )
                test_results = fisher_exact_test(
                    dmode_obj=dmode_obj,
                    table=fisher_exact_test_table,
                    id=position_token,
                    ref_level=ref_level,
                    cond_level=cond_level,
                )
                test_results["positions"] = position_token
            else:
                counts_modified = []
                counts_unmodified = []
                treatment = []
                for reference_df in reference_dfs:
                    ref_temp_df = reference_df.iloc[idx, :].to_dict()
                    counts_modified.append(ref_temp_df["Nmod"])
                    counts_unmodified.append(
                        ref_temp_df["Nvalid_cov"] - ref_temp_df["Nmod"]
                    )
                    treatment.append(ref_level)

                for condition_df in condition_dfs:
                    cond_temp_df = condition_df.iloc[idx, :].to_dict()
                    counts_modified.append(cond_temp_df["Nmod"])
                    counts_unmodified.append(
                        (cond_temp_df["Nvalid_cov"] - cond_temp_df["Nmod"])
                    )
                    treatment.append(cond_level)

                test_results = logReg(
                    dmode_obj=dmode_obj,
                    counts_modified_sites = counts_modified,
                    counts_unmodified_sites = counts_unmodified,
                    treatment = treatment,
                    vars_df=pd.DataFrame(),
                    id=position_token,
                    ref_level=ref_level,
                    cond_level=cond_level
                )
                test_results["positions"] = position_token

            chrom, start, end, strand, mod_type, positions = position_token_conversion(position_token=position_token,base_modification_type_dict=base_modification_type_dict)
            test_results["chrom"] = chrom
            test_results["start"] = start
            test_results["end"] = end
            test_results["strand"] = strand
            test_results["mod_type"] = mod_type
            test_results["positions"] = positions
            results_list.append(test_results)
        output_df = pd.concat([output_df, pd.DataFrame(results_list)], axis=0)
        valid_mask = output_df["pvalue"] < 1.0
        output_df["padj"] = np.nan
        if correction_method == "SLIM":
            output_df.loc[valid_mask, "padj"] = SLIMfunc(dmode_obj=dmode_obj,raw_pvalues=output_df.loc[valid_mask,"pvalue"],output_folder=output_folder)
        if correction_method == "bh":
            output_df.loc[valid_mask, "padj"] = benjamini_hochberg(dmode_obj=dmode_obj,pvals=output_df.loc[valid_mask,"pvalue"])
        output_df["neg_log_p"] = -np.log10(output_df["padj"])
        comparison = f"{cond_level}_VS_{ref_level}"
        comparisons_dict[comparison] = pd.DataFrame(output_df)
        outer_join_comparisons_dict[comparison] = outer_join_df
    dmode_obj.diff_gene_mod_comparisons_dict = comparisons_dict
    dmode_obj.diff_gene_mod_outer_join_comparisons_dict = outer_join_comparisons_dict
    if output_folder != "":
        os.makedirs(output_folder, exist_ok=True)
        for comparison in comparisons_dict.keys():
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            dmode_obj.diff_gene_mod_comparisons_dict[comparison].to_csv(f"{output_folder}/{comparison}/{comparison}_diff_gene_mod.tsv",sep="\t",index=False)
            dmode_obj.diff_gene_mod_outer_join_comparisons_dict[comparison].to_csv(f"{output_folder}/{comparison}/{comparison}_diff_gene_modification_raw_count_table.tsv",sep="\t",index=False)
    return dmode_obj.diff_gene_mod_comparisons_dict, dmode_obj.diff_gene_mod_outer_join_comparisons_dict


def diff_gene_mod_cond_extract_significant_positions(
    dmode_obj, ref_level:str, cond_level:str, alpha:float = 0.05, chr:str="", output_folder:str=""
):
    """
    Extract genomic positions with significant differential modification.
    
    This method filters the results from differential modification analysis
    to identify positions that meet the statistical significance threshold.
    
    Parameters
    ----------
    ref_level : str
        Name of the reference condition level.
    cond_level : str
        Name of the comparison condition level.
    alpha : float, default 0.05
        Significance threshold for adjusted p-values.
    chr : str, default ""
        Specific chromosome to filter for. If empty, all chromosomes included.
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing significant positions with columns:
        - chrom : chromosome name
        - start : start position
        - end : end position  
        - strand : strand information
        - mod_type : modification type
        
    Notes
    -----
    - Positions are filtered based on padj < alpha
    - Results can be optionally filtered by chromosome
    - Output is saved as tab-separated file if output_folder is specified
    - This creates a BED-like format suitable for genomic analysis tools
    """
    if chr == "":
        filtered_df = dmode_obj.diff_gene_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
        filtered_df = filtered_df[(filtered_df["padj"] < alpha)]
        # Create BED DataFrame
        bed_df = pd.DataFrame(
            filtered_df, columns=["chrom", "start", "end", "strand", "mod_type"]
        )
        if output_folder != "":
            comparison = f"{cond_level}_VS_{ref_level}"
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            bed_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_significant_diff_gene_mod.tsv",sep="\t",index=False)
        return bed_df
    else:
        filtered_df = dmode_obj.diff_gene_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
        filtered_df = filtered_df[
            (filtered_df["padj"] < alpha) & (filtered_df["chrom"] == chr)
        ]
        # Create BED DataFrame
        bed_df = pd.DataFrame(
            filtered_df, columns=["chrom", "start", "end", "strand", "mod_type"]
        )
        if output_folder != "":
            comparison = f"{cond_level}_VS_{ref_level}"
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            bed_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_significant_diff_gene_mod.tsv",sep="\t",index=False)
        return bed_df

def diff_gene_mod_cond_extract_significant_genes(
    dmode_obj, gtf_df:pd.DataFrame, ref_level, cond_level, alpha: float = 0.05, genes_only:bool=True, output_folder:str=""
):
    """
    Identify genes containing significantly differentially modified positions.
    
    This method intersects significantly modified genomic positions with gene
    annotations from a GTF file to identify genes that may be affected by
    differential modification.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file containing gene coordinates.
    ref_level : str
        Name of the reference condition level.
    cond_level : str
        Name of the comparison condition level.
    alpha : float, default 0.05
        Significance threshold for adjusted p-values.
    genes_only : bool, default True
        Whether to return only gene-level information (currently not implemented).
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing genes with significant modifications:
        - gene_name : gene symbol/name
        - gene_id : gene identifier
        - chrom : chromosome name
        - start : modification start position
        - end : modification end position
        - strand : strand information
        - mod_type : modification type
        
    Notes
    -----
    - Uses genomic overlap to match modification positions to genes
    - Requires strand-specific matching between modifications and genes
    - Multiple modifications per gene will result in multiple rows
    - Progress is shown with tqdm progress bar
    - Results saved as tab-separated file if output_folder is specified
    
    The GTF file should contain gene annotations with gene_id and gene_name
    attributes in the standard GTF format.
    """
    filtered_df = dmode_obj.diff_gene_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
    filtered_df = filtered_df[
        ["chrom", "start", "end", "strand", "mod_type"]
    ]

    
    significant_genes_list = []
    #Find all intersecting genes between differentially modified sites and genes in gtf file
    for chrom_i,start_i,end_i,strand_i,mod_type_i in tqdm(zip(filtered_df["chrom"],filtered_df["start"],filtered_df["end"],filtered_df["strand"],filtered_df["mod_type"]), total=filtered_df.shape[0]):
        temp_gtf_df = gtf_df[(gtf_df['seqname'] == chrom_i) & (gtf_df['start'] <= int(start_i)) & (gtf_df['end'] >= int(end_i)) & (gtf_df['strand'] == strand_i)]
        # if temp_gtf_df.empty:
        #     list_element = {"chrom":chrom_i,"start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"gene_id":"","gene_name":""}
        #     significant_genes_list.append(list_element)
        if isinstance(temp_gtf_df, pd.Series):
            list_element = {"chrom":chrom_i,"start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"gene_id":gtf_df["gene_id"],"gene_name":gtf_df["gene_name"]}
            significant_genes_list.append(list_element)
        else:
            for gene_id_j,gene_name_j in zip(temp_gtf_df["gene_id"],temp_gtf_df["gene_name"]):
                list_element = {"chrom":chrom_i,"start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"gene_id":gene_id_j,"gene_name":gene_name_j}
                significant_genes_list.append(list_element)
    
    #Create dataframe of extracted combinations
    significant_genes_df = pd.DataFrame(significant_genes_list)
    significant_genes_df = significant_genes_df[["gene_name","gene_id","chrom","start","end","strand","mod_type"]]
    #Write dataframe
    if output_folder != "":
        comparison = f"{cond_level}_VS_{ref_level}"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
        significant_genes_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_diff_gene_mod_genes.tsv",sep="\t",index=False)
    return significant_genes_df



# //////// Generate a volcano plot for mod. differetial expression and for each chromosome ////

def diff_gene_mod_cond_generate_volcano_plots(
    Chr_names: list,
    Data_analyzed: pd.DataFrame,
    alpha: float,
    modtype: str = None,
    annotation_df: Optional[pd.DataFrame] = None,
):
    """
    Generate multi-panel volcano plots showing differential modification across chromosomes.
    
    This function creates a grid of volcano plots, one for each chromosome, displaying
    the relationship between modification frequency difference and statistical significance.
    Points are colored based on significance and effect size thresholds.
    
    Parameters
    ----------
    Chr_names : list of str
        List of chromosome names to plot (e.g., ['chr1', 'chr2', ..., 'chrY']).
    Data_analyzed : pandas.DataFrame
        DataFrame containing differential modification analysis results with columns:
        - 'chrom': chromosome name
        - 'mod_freq_diff': modification frequency difference (%)
        - 'padj': adjusted p-values
        - 'neg_log_p': negative log10 of adjusted p-values
        - 'positions': genomic position identifiers
    alpha : float
        Significance threshold for adjusted p-values (e.g., 0.05).
    annotation_df : pandas.DataFrame, optional
        Optional annotation table containing a `positions` column along with
        identifier columns (e.g., `gene_name`, `gene_id`) used to enrich labels.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the multi-panel volcano plots.
        
    Notes
    -----
    - Creates a 4-column grid with as many rows as needed
    - Color coding:
        - Gray (#BDBDBD): not significant or effect size < 4%
        - Blue (#4575B4): significant decrease
        - Red (#D73027): significant increase
    - Horizontal line at -log10(alpha) for significance threshold
    - Vertical lines at ±4% modification difference
    - X-axis range: -100% to +100%
    - Scientific styling with minimal decorations
    - Top hits annotated with gene identifiers (chr:start:strand) when available
    
    The plot helps visualize both statistical significance and biological relevance
    of modification changes across the genome.
    """
    import seaborn as sns
    
    #filter data_analyzed by modtype if provided
    working_df = Data_analyzed if modtype is None else Data_analyzed[Data_analyzed["mod_type"] == modtype]
    working_df = working_df.copy()

    working_df = _merge_annotation_columns(
        working_df,
        annotation_df,
        key_column="positions",
        value_columns=["gene_name", "gene_id"],
    )

    # Define the number (X) of plots to create
    # Plots will then be shown in X rows * 4
    number_of_chromosomes = len(Chr_names)
    number_of_subplots = int(number_of_chromosomes)
    while number_of_subplots % 4 != 0:
        number_of_subplots += 1

    num_rows, num_cols = math.ceil(number_of_subplots // 4) , 4
    
    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.0, rc={"axes.labelsize": 12, "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows), facecolor="white")
    axes = axes.flatten()  # To easily index subplots in a 1D loop

    # Define colors matching scientific palette
    color_decrease = '#4575B4'  # Blue for decrease
    color_increase = '#D73027'  # Red for increase
    color_ns = '#BDBDBD'  # Gray for not significant

    for i in range(number_of_chromosomes):
        ax = axes[i]
        final_analysis_df = working_df[working_df["chrom"] == Chr_names[i]]

        # Create masks
        blue_mask = (final_analysis_df['mod_freq_diff'] <= -4) & (final_analysis_df['padj'] < alpha)
        red_mask = (final_analysis_df['mod_freq_diff'] >= 4) & (final_analysis_df['padj'] < alpha)
        gray_mask = ~(blue_mask | red_mask)

        # Plot gray points (not significant)
        ax.scatter(
            final_analysis_df.loc[gray_mask, 'mod_freq_diff'],
            final_analysis_df.loc[gray_mask, 'neg_log_p'],
            color=color_ns, alpha=0.4, s=20, edgecolor='none'
        )

        # Plot blue points (significant decrease)
        ax.scatter(
            final_analysis_df.loc[blue_mask, 'mod_freq_diff'],
            final_analysis_df.loc[blue_mask, 'neg_log_p'],
            color=color_decrease, alpha=0.8, s=30, edgecolor='black', linewidth=0.3
        )

        # Plot red points (significant increase)
        ax.scatter(
            final_analysis_df.loc[red_mask, 'mod_freq_diff'],
            final_analysis_df.loc[red_mask, 'neg_log_p'],
            color=color_increase, alpha=0.8, s=30, edgecolor='black', linewidth=0.3
        )

        # Annotate top 10 most significant points with simplified labels
        significant_df = final_analysis_df[blue_mask | red_mask].copy()
        if len(significant_df) > 0:
            # Sort by significance and take top 10
            top_sig = significant_df.nlargest(min(10, len(significant_df)), 'neg_log_p')
            texts = []
            for _, row in top_sig.iterrows():
                color = color_decrease if row['mod_freq_diff'] < 0 else color_increase
                pos_parts = str(row['positions']).split(':')
                strand_token = row['strand'] if 'strand' in row.index and pd.notna(row['strand']) else (pos_parts[3] if len(pos_parts) >= 4 else None)
                if len(pos_parts) >= 2:
                    base_label = f"{pos_parts[0]}:{pos_parts[1]}"
                    if strand_token:
                        base_label = f"{base_label}:{strand_token}"
                else:
                    base_label = str(row['positions'])
                label = _compose_volcano_label(row, base_label, [["gene_name"], ["gene_id"]])
                texts.append(ax.text(row['mod_freq_diff'], row['neg_log_p'], label,
                                     fontsize=6, color=color))
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        # Decorations - scientific style
        ax.set_xlabel('Modification frequency\ndifference (%)', fontsize=11)
        ax.set_ylabel('-log$_{10}$(p-adjusted)', fontsize=11)
        ax.set_title(Chr_names[i], fontsize=13)
        
        # Threshold lines
        ax.axhline(y=-np.log10(alpha), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=4, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-4, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_xlim(-100, 100)
        ax.set_facecolor('white')
        
        # Clean axes - only show left and bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

    # Hide empty subplots
    for i in range(number_of_chromosomes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig
    


    
def diff_gene_mod_cond_generate_volcano_plot_single_chr(
    chr: str,
    Data_analyzed: pd.DataFrame,
    alpha: float,
    modtype: str = None,
    annotation_df: Optional[pd.DataFrame] = None,
):
    """
    Generate a single volcano plot for one chromosome or all chromosomes combined.
    
    This function creates a volcano plot showing the relationship between modification
    frequency difference and statistical significance, with optional chromosome filtering.
    
    Parameters
    ----------
    chr : str
        Chromosome name to plot (e.g., 'chr1'). If empty string, plots all chromosomes.
    Data_analyzed : pandas.DataFrame
        DataFrame containing differential modification analysis results with columns:
        - 'chrom': chromosome name
        - 'mod_freq_diff': modification frequency difference (%)
        - 'padj': adjusted p-values
        - 'neg_log_p': negative log10 of adjusted p-values
        - 'positions': genomic position identifiers
    alpha : float
        Significance threshold for adjusted p-values (e.g., 0.05).
    annotation_df : pandas.DataFrame, optional
        Optional annotation table with `positions` and identifier columns used to
        enrich point labels.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the volcano plot.
        
    Notes
    -----
    - Creates a single figure with scientific styling
    - Color coding:
        - Gray (#BDBDBD): not significant or effect size < 4%
        - Blue (#4575B4): significant decrease
        - Red (#D73027): significant increase
    - Reference lines:
        - Horizontal at -log10(alpha) for significance threshold
        - Vertical at ±4% modification difference
    - X-axis range: -100% to +100%
    - Clean, publication-quality appearance
    - Top 10 most significant positions are annotated with gene identifiers (chr:start:strand) when available
    
    This function is useful for detailed examination of individual chromosomes
    or creating publication-quality figures of specific genomic regions.
    """
    import seaborn as sns
    
    #filter data_analyzed by modtype if provided
    working_df = Data_analyzed if modtype is None else Data_analyzed[Data_analyzed["mod_type"] == modtype]
    working_df = working_df.copy()

    working_df = _merge_annotation_columns(
        working_df,
        annotation_df,
        key_column="positions",
        value_columns=["gene_name", "gene_id"],
    )

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    
    if chr != "":
        final_analysis_df = working_df[working_df["chrom"] == chr]
    else:
        final_analysis_df = working_df

    def _compose_label_no_paren(row, fallback_label, identifier_groups):
        """Return identifier:first_available + fallback, joined with ':' instead of parentheses.

        Also trims pipe-delimited tokens to their first segment to avoid verbose headers.
        """
        for fields in identifier_groups:
            values = []
            for field in fields:
                if field in row.index and pd.notna(row[field]):
                    text_value = str(row[field]).strip()
                    if text_value:
                        cleaned = text_value.split("|")[0]
                        if cleaned not in values:
                            values.append(cleaned)
            if values:
                identifier = " / ".join(values)
                return f"{identifier}:{fallback_label}"
        return fallback_label
    
    # Create masks
    blue_mask = (final_analysis_df['mod_freq_diff'] <= -4) & (final_analysis_df['padj'] < alpha)
    red_mask = (final_analysis_df['mod_freq_diff'] >= 4) & (final_analysis_df['padj'] < alpha)
    gray_mask = ~(blue_mask | red_mask)

    # Define colors matching scientific palette
    color_decrease = '#4575B4'  # Blue for decrease
    color_increase = '#D73027'  # Red for increase
    color_ns = '#BDBDBD'  # Gray for not significant

    # Plot gray points (not significant)
    ax.scatter(
        final_analysis_df.loc[gray_mask, 'mod_freq_diff'],
        final_analysis_df.loc[gray_mask, 'neg_log_p'],
        color=color_ns, alpha=0.4, s=30, edgecolor='none', label='Not significant'
    )

    # Plot blue points (significant decrease)
    ax.scatter(
        final_analysis_df.loc[blue_mask, 'mod_freq_diff'],
        final_analysis_df.loc[blue_mask, 'neg_log_p'],
        color=color_decrease, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant decrease'
    )

    # Plot red points (significant increase)
    ax.scatter(
        final_analysis_df.loc[red_mask, 'mod_freq_diff'],
        final_analysis_df.loc[red_mask, 'neg_log_p'],
        color=color_increase, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant increase'
    )

    # Annotate top 10 most significant points with simplified labels
    significant_df = final_analysis_df[blue_mask | red_mask].copy()
    if len(significant_df) > 0:
        # Sort by significance and take top 10
        top_sig = significant_df.nlargest(min(10, len(significant_df)), 'neg_log_p')
        texts = []
        for _, row in top_sig.iterrows():
            pos_parts = str(row['positions']).split(':')
            strand_token = row['strand'] if 'strand' in row.index and pd.notna(row['strand']) else (pos_parts[3] if len(pos_parts) >= 4 else None)
            if len(pos_parts) >= 2:
                base_label = f"{pos_parts[0]}:{pos_parts[1]}"
                if strand_token:
                    base_label = f"{base_label}:{strand_token}"
            else:
                base_label = str(row['positions'])
            label = _compose_volcano_label(row, base_label, [["gene_name"], ["gene_id"]])
            texts.append(ax.text(row['mod_freq_diff'], row['neg_log_p'], label, fontsize=6, color='black'))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    # Decorations - scientific style
    ax.set_xlabel('Modification frequency difference (%)', fontsize=16)
    ax.set_ylabel('-log$_{10}$(p-adjusted)', fontsize=16)

    # Threshold lines
    ax.axhline(y=-np.log10(alpha), color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=4, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=-4, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    
    ax.set_xlim(-100, 100)
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
    return fig




### Transcriptome based differential modification analysis ###


def _prepare_transcript_annotation_lookup(dmode_obj, gtf_file, alpha, output_folder):
    """Create per-comparison transcript annotation tables for volcano labeling."""
    annotations = {}
    gtf_df = gtf_to_df(gtf_file)
    gtf_df = gtf_df[gtf_df['feature'] == 'transcript']

    for comparison_key in dmode_obj.diff_transcript_mod_comparisons_dict.keys():
        cond_level,reference_level = comparison_key.split("_VS_")
        annotation_df = diff_transcript_mod_cond_extract_significant_transcripts(
            dmode_obj=dmode_obj,
            gtf_df=pd.DataFrame(gtf_df),
            ref_level=reference_level,
            cond_level=cond_level,
            alpha=alpha,
            genes_only=True,
            output_folder=output_folder,
        )
        if annotation_df is None or annotation_df.empty:
            annotations[comparison_key] = annotation_df
            continue
        annotation_df = annotation_df.copy()
        transcript_ids = annotation_df["transcript_id"].fillna("").astype(str)
        chrom_values = annotation_df["chrom"].fillna("").astype(str)
        combined = transcript_ids.copy()
        mask = transcript_ids != ""
        combined.loc[mask] = transcript_ids.loc[mask] + "|" + chrom_values.loc[mask]
        combined.loc[~mask] = chrom_values.loc[~mask]
        annotation_df["chrom_token"] = combined

        primary_annotations = _append_positions_column(
            annotation_df,
            chrom_column="chrom_token",
        )

        alternate_annotations = annotation_df[mask].copy()
        alternate_annotations = _append_positions_column(
            alternate_annotations,
            chrom_column="transcript_id",
        )

        chromosome_annotations = _append_positions_column(
            annotation_df,
            chrom_column="chrom",
        )

        merged_annotations = pd.concat(
            [primary_annotations, alternate_annotations, chromosome_annotations],
            ignore_index=True,
        ).drop_duplicates(subset="positions")

        debug_cols = [
            col
            for col in ["positions", "transcript_id", "transcript_name", "gene_name"]
            if col in merged_annotations.columns
        ]
        annotations[comparison_key] = merged_annotations
    return annotations


def diff_transcript_mod_cond_extraction(dmode_obj, correction_method="SLIM", output_folder=""):
    """
    Perform differential modification analysis between all specified comparisons.
    
    This is the main analysis method that compares modification levels between
    reference and condition groups for all overlapping genomic positions.
    Statistical testing is performed using either Fisher's exact test (for 
    single replicates) or logistic regression (for multiple replicates).
    
    Parameters
    ----------
    correction_method : {"SLIM", "bh"}, default "SLIM"
        Method for multiple testing correction:
        - "SLIM" : Use SLIM method for FDR control
        - "bh" : Use Benjamini-Hochberg correction
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    dict
        Dictionary mapping comparison names (format: "cond_VS_ref") to 
        pandas DataFrames containing:
        - Genomic coordinates (chrom, start, end, strand, mod_type)
        - Statistical test results (pvalue, padj, mod_freq_diff)
        - Condition-specific modification frequencies
        - Negative log10 adjusted p-values for visualization
        
    Notes
    -----
    - Only positions present in ALL samples of both conditions are analyzed
    - For single replicates per condition: Fisher's exact test is used
    - For multiple replicates: Logistic regression with binomial GLM is used
    - Results are saved as tab-separated files if output_folder is specified
    - Progress is shown with tqdm progress bars
    
    The method updates dmode_obj.diff_transcript_mod_comparisons_dict with results.
    """
    comparisons_dict = {}
    outer_join_comparisons_dict = {}
    base_modification_type_dict={
        "a":"m6A",
        "69426": "Am",
        "17596": "Ino",
        "17802": "pseU",
        "19227": "Um",
        "19229": "Gm",
        "19228": "Cm",
        "m": "m5C"
    }
    for idx, (cond_level, ref_level) in enumerate(dmode_obj.diff_transcript_mod_comparisons):
        output_df = pd.DataFrame()
        reference_dfs = [df for df in list(dmode_obj.diff_transcript_mod_dataframe_condition_dict[ref_level].values())]
        condition_dfs = [df for df in list(dmode_obj.diff_transcript_mod_dataframe_condition_dict[cond_level].values())]
        # combined_df = pd.merge(reference_df,condition_df,how="inner",suffixes=('_ref', '_cond'))
        
        outer_join_df = pd.DataFrame()
        for df, sample in zip(dmode_obj.diff_transcript_mod_dataframe_condition_dict[ref_level].values(),dmode_obj.diff_transcript_mod_dataframe_condition_dict[ref_level].keys()):
            outer_join_df = outer_join(df1 = outer_join_df,df2 =  df,sample2 = sample)
        
        for df, sample in zip(dmode_obj.diff_transcript_mod_dataframe_condition_dict[cond_level].values(),dmode_obj.diff_transcript_mod_dataframe_condition_dict[cond_level].keys()):
            outer_join_df = outer_join(df1 = outer_join_df,df2 =  df,sample2 = sample)    
            
        outer_join_df = outer_join_df.fillna(value = 0)
        sorted_columns = sorted(list(outer_join_df.columns))
        outer_join_df = outer_join_df[sorted_columns]
        
        outer_join_df = outer_join_df.fillna(value = 0)
        sorted_columns = sorted(list(outer_join_df.columns))
        outer_join_df = outer_join_df[sorted_columns]
        transcript_positions = []
        for entry in outer_join_df["position_token"]:
            chrom, start, end, strand, mod_type, positions = position_token_conversion(position_token=entry,base_modification_type_dict=base_modification_type_dict)
            if "|" in chrom:
                transcript_id_token, chrom_clean = chrom.split("|", 1)
                transcript_positions.append(f"{transcript_id_token}|{chrom_clean}:{start}:{end}:{strand}:{mod_type}")
            else:
                transcript_positions.append(f"{chrom}:{start}:{end}:{strand}:{mod_type}")
        outer_join_df["positions"] = transcript_positions
        
        intersection_lists = []
        for df_element in reference_dfs + condition_dfs:
            intersection_lists.append(df_element["position_token"])
        intersecting_position_tokens = set(intersection_lists[0]).intersection(
            *intersection_lists[1:]
        )

        reference_dfs = [
            filter_and_sort(df, intersecting_position_tokens)
            for df in list(dmode_obj.diff_transcript_mod_dataframe_condition_dict[ref_level].values())
        ]
        condition_dfs = [
            filter_and_sort(df, intersecting_position_tokens)
            for df in list(dmode_obj.diff_transcript_mod_dataframe_condition_dict[cond_level].values())
        ]
        intersecting_position_tokens = sorted(intersecting_position_tokens)

        results_list = []
        for idx, position_token in tqdm(
            enumerate(intersecting_position_tokens),
            total=len(intersecting_position_tokens),
        ):
            if len(reference_dfs) == 1 and len(condition_dfs) == 1:
                ref_temp_df = reference_dfs[0].iloc[idx, :].to_dict()
                cond_temp_df = condition_dfs[0].iloc[idx, :].to_dict()
                ref_modified_counts = ref_temp_df["Nmod"]
                ref_unmodifed_counts = (
                    ref_temp_df["Nvalid_cov"] - ref_temp_df["Nmod"]
                )
                cond_modified_counts = cond_temp_df["Nmod"]
                cond_unmodified_counts = (
                    cond_temp_df["Nvalid_cov"] - cond_temp_df["Nmod"]
                )
                fisher_exact_test_table = np.array(
                    [
                        [ref_modified_counts, ref_unmodifed_counts],
                        [cond_modified_counts, cond_unmodified_counts],
                    ]
                )
                test_results = fisher_exact_test(
                    dmode_obj=dmode_obj,
                    table=fisher_exact_test_table,
                    id=position_token,
                    ref_level=ref_level,
                    cond_level=cond_level,
                )
                chrom, start, end, strand, mod_type, positions = position_token_conversion(position_token=position_token,base_modification_type_dict=base_modification_type_dict)
                if "|" in chrom:
                    transcript_id_token, chrom_clean = chrom.split("|", 1)
                    composed_position = f"{transcript_id_token}|{chrom_clean}:{start}:{end}:{strand}:{mod_type}"
                else:
                    transcript_id_token = ""
                    chrom_clean = chrom
                    composed_position = f"{chrom_clean}:{start}:{end}:{strand}:{mod_type}"

                test_results["chrom"] = chrom
                test_results["start"] = start
                test_results["end"] = end
                test_results["strand"] = strand
                test_results["mod_type"] = mod_type
                test_results["positions"] = composed_position
            else:
                counts_modified = []
                counts_unmodified = []
                treatment = []
                for reference_df in reference_dfs:
                    ref_temp_df = reference_df.iloc[idx, :].to_dict()
                    counts_modified.append(ref_temp_df["Nmod"])
                    counts_unmodified.append(
                        ref_temp_df["Nvalid_cov"] - ref_temp_df["Nmod"]
                    )
                    treatment.append(ref_level)

                for condition_df in condition_dfs:
                    cond_temp_df = condition_df.iloc[idx, :].to_dict()
                    counts_modified.append(cond_temp_df["Nmod"])
                    counts_unmodified.append(
                        (cond_temp_df["Nvalid_cov"] - cond_temp_df["Nmod"])
                    )
                    treatment.append(cond_level)

                test_results = logReg(
                    dmode_obj=dmode_obj,
                    counts_modified_sites = counts_modified,
                    counts_unmodified_sites = counts_unmodified,
                    treatment = treatment,
                    vars_df=pd.DataFrame(),
                    id=position_token,
                    ref_level=ref_level,
                    cond_level=cond_level
                )
                chrom, start, end, strand, mod_type, positions = position_token_conversion(position_token=position_token,base_modification_type_dict=base_modification_type_dict)
                if "|" in chrom:
                    transcript_id_token, chrom_clean = chrom.split("|", 1)
                    composed_position = f"{transcript_id_token}|{chrom_clean}:{start}:{end}:{strand}:{mod_type}"
                else:
                    transcript_id_token = ""
                    chrom_clean = chrom
                    composed_position = f"{chrom_clean}:{start}:{end}:{strand}:{mod_type}"

                test_results["chrom"] = chrom
                test_results["start"] = start
                test_results["end"] = end
                test_results["strand"] = strand
                test_results["mod_type"] = mod_type
                test_results["positions"] = composed_position
            results_list.append(test_results)
        output_df = pd.concat([output_df, pd.DataFrame(results_list)], axis=0)
        valid_mask = output_df["pvalue"] < 1.0
        output_df["padj"] = np.nan
        if correction_method == "SLIM":
            output_df.loc[valid_mask, "padj"] = SLIMfunc(dmode_obj=dmode_obj,raw_pvalues=output_df.loc[valid_mask,"pvalue"],output_folder=output_folder)
        if correction_method == "bh":
            output_df.loc[valid_mask, "padj"] = benjamini_hochberg(dmode_obj=dmode_obj,pvals=output_df.loc[valid_mask,"pvalue"])
        output_df["neg_log_p"] = -np.log10(output_df["padj"])
        comparisons_dict[f"{cond_level}_VS_{ref_level}"] = pd.DataFrame(output_df)
        outer_join_comparisons_dict[f"{cond_level}_VS_{ref_level}"] = outer_join_df
    dmode_obj.diff_transcript_mod_comparisons_dict = comparisons_dict
    dmode_obj.diff_transcript_mod_outer_join_comparisons_dict = outer_join_comparisons_dict
    if output_folder != "":
        os.makedirs(output_folder, exist_ok=True)
        for comparison in comparisons_dict.keys():
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            dmode_obj.diff_transcript_mod_comparisons_dict[comparison].to_csv(f"{output_folder}/{comparison}/{comparison}_diff_transcript_mod.tsv",sep="\t",index=False)
            dmode_obj.diff_transcript_mod_outer_join_comparisons_dict[comparison].to_csv(f"{output_folder}/{comparison}/{comparison}_diff_transcript_modification_raw_count_table.tsv",sep="\t",index=False)
    return dmode_obj.diff_transcript_mod_comparisons_dict, dmode_obj.diff_transcript_mod_outer_join_comparisons_dict


def diff_transcript_mod_cond_extract_significant_positions(
    dmode_obj, ref_level:str, cond_level:str, alpha:float = 0.05, chr:str="", output_folder:str=""
):
    """
    Extract genomic positions with significant differential modification.
    
    This method filters the results from differential modification analysis
    to identify positions that meet the statistical significance threshold.
    
    Parameters
    ----------
    ref_level : str
        Name of the reference condition level.
    cond_level : str
        Name of the comparison condition level.
    alpha : float, default 0.05
        Significance threshold for adjusted p-values.
    chr : str, default ""
        Specific chromosome to filter for. If empty, all chromosomes included.
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing significant positions with columns:
        - chrom : chromosome name
        - start : start position
        - end : end position  
        - strand : strand information
        - mod_type : modification type
        
    Notes
    -----
    - Positions are filtered based on padj < alpha
    - Results can be optionally filtered by chromosome
    - Output is saved as tab-separated file if output_folder is specified
    - This creates a BED-like format suitable for genomic analysis tools
    """
    if chr == "":
        filtered_df = dmode_obj.diff_transcript_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
        filtered_df = filtered_df[(filtered_df["padj"] < alpha)]
        # Create BED DataFrame
        bed_df = pd.DataFrame(
            filtered_df, columns=["chrom", "start", "end", "strand", "mod_type"]
        )
        if output_folder != "":
            comparison = f"{cond_level}_VS_{ref_level}"
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            bed_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_significant_diff_transcript_mod.tsv",sep="\t",index=False)
        return bed_df
    else:
        filtered_df = dmode_obj.diff_transcript_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
        filtered_df = filtered_df[
            (filtered_df["padj"] < alpha) & (filtered_df["chrom"] == chr)
        ]
        # Create BED DataFrame
        bed_df = pd.DataFrame(
            filtered_df, columns=["chrom", "start", "end", "strand", "mod_type"]
        )
        if output_folder != "":
            comparison = f"{cond_level}_VS_{ref_level}"
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
            bed_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_significant_diff_transcript_mod.tsv",sep="\t",index=False)
        return bed_df

def diff_transcript_mod_cond_extract_significant_transcripts(
    dmode_obj, gtf_df:pd.DataFrame, ref_level, cond_level, alpha: float = 0.05, genes_only:bool=True, output_folder:str=""
):
    """
    Identify genes containing significantly differentially modified positions.
    
    This method intersects significantly modified genomic positions with gene
    annotations from a GTF file to identify genes that may be affected by
    differential modification.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file containing gene coordinates.
    ref_level : str
        Name of the reference condition level.
    cond_level : str
        Name of the comparison condition level.
    alpha : float, default 0.05
        Significance threshold for adjusted p-values.
    genes_only : bool, default True
        Whether to return only gene-level information (currently not implemented).
    output_folder : str, default ""
        Directory path to save results. If empty, no files are written.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing genes with significant modifications:
        - gene_name : gene symbol/name
        - gene_id : gene identifier
        - chrom : chromosome name
        - start : modification start position
        - end : modification end position
        - strand : strand information
        - mod_type : modification type
        
    Notes
    -----
    - Uses genomic overlap to match modification positions to genes
    - Requires strand-specific matching between modifications and genes
    - Multiple modifications per gene will result in multiple rows
    - Progress is shown with tqdm progress bar
    - Results saved as tab-separated file if output_folder is specified
    
    The GTF file should contain gene annotations with gene_id and gene_name
    attributes in the standard GTF format.
    """
    filtered_df = dmode_obj.diff_transcript_mod_comparisons_dict[f"{cond_level}_VS_{ref_level}"]
    filtered_df = filtered_df[
        ["chrom", "start", "end", "strand", "mod_type"]
    ]
    new_chrom_names = [element.split("|")[0] for element in filtered_df["chrom"]]
    filtered_df.loc[:,"chrom"] = new_chrom_names
    significant_genes_list = []
    #Find all intersecting genes between differentially modified sites and genes in gtf file
    for transcript_i,start_i,end_i,strand_i,mod_type_i in tqdm(zip(filtered_df["chrom"],filtered_df["start"],filtered_df["end"],filtered_df["strand"],filtered_df["mod_type"]), total=filtered_df.shape[0]):
        temp_gtf_df = gtf_df[(gtf_df['transcript_id'] == transcript_i)]
        # if temp_gtf_df.empty:
        #     list_element = {"chrom":"","start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"transcript_id":transcript_i,"transcript_name":"","gene_name":""}
        #     significant_genes_list.append(list_element)
        if isinstance(temp_gtf_df, pd.Series):
            list_element = {"chrom":temp_gtf_df["seqname"],"start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"transcript_id":gtf_df["transcript_id"],"transcript_name":gtf_df["transcript_name"],"gene_name":gtf_df["gene_name"]}
            significant_genes_list.append(list_element)
        else:
            for transcript_id_j,transcript_name_j,gene_id_j,gene_name_j in zip(temp_gtf_df["transcript_id"],temp_gtf_df["transcript_name"],temp_gtf_df["gene_id"],temp_gtf_df["gene_name"]):
                chromosome = list(temp_gtf_df["seqname"])[0]
                list_element = {"chrom":chromosome,"start":start_i,"end":end_i,"strand":strand_i,"mod_type":mod_type_i,"transcript_id":transcript_id_j,"transcript_name":transcript_name_j,"gene_id":gene_id_j,"gene_name":gene_name_j}
                significant_genes_list.append(list_element)
    
    #Create dataframe of extracted combinations
    significant_genes_df = pd.DataFrame(significant_genes_list)
    if significant_genes_df.empty:
        return significant_genes_df
    significant_genes_df = significant_genes_df[["transcript_id","transcript_name","gene_name","gene_id","chrom","start","end","strand","mod_type"]]
    #Write dataframe
    if output_folder != "":
        comparison = f"{cond_level}_VS_{ref_level}"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f"{output_folder}/{comparison}", exist_ok=True)
        significant_genes_df.to_csv(f"{output_folder}/{comparison}/{cond_level}_VS_{ref_level}_diff_transcript_mod_transcripts.tsv",sep="\t",index=False)
    return significant_genes_df



# //////// Generate a volcano plot for mod. differetial expression and for each chromosome ////
    
def diff_transcript_mod_cond_generate_volcano_plot_single_chr(
    chr: str,
    Data_analyzed: pd.DataFrame,
    alpha: float,
    modtype: str = None,
    annotation_df: Optional[pd.DataFrame] = None,
):
    """
    Generate a single volcano plot for one chromosome or all transcripts combined.
    
    This function creates a volcano plot showing the relationship between modification
    frequency difference and statistical significance, with optional chromosome filtering.
    
    Parameters
    ----------
    chr : str
        Chromosome name to plot (e.g., 'chr1'). If empty string, plots all chromosomes.
    Data_analyzed : pandas.DataFrame
        DataFrame containing differential modification analysis results with columns:
        - 'chr': name of the gene of interest
        - 'mod_freq_diff': modification frequency difference (%)
        - 'padj': adjusted p-values
        - 'neg_log_p': negative log10 of adjusted p-values
        - 'positions': genomic position identifiers
    alpha : float
        Significance threshold for adjusted p-values (e.g., 0.05).
    annotation_df : pandas.DataFrame, optional
        Optional annotation table containing `positions` plus identifier columns
        such as `transcript_name`, `transcript_id`, or `gene_name` for
        annotating significant hits.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the volcano plot.
        
    Notes
    -----
    - Creates a single figure with scientific styling
    - Color coding:
        - Gray (#BDBDBD): not significant or effect size < 4%
        - Blue (#4575B4): significant decrease
        - Red (#D73027): significant increase
    - Reference lines:
        - Horizontal at -log10(alpha) for significance threshold
        - Vertical at ±4% modification difference
    - X-axis range: -100% to +100%
    - Clean, publication-quality appearance
        - Top 10 most significant positions are annotated with transcript names (fallback to IDs) when available
    
    This function is useful for detailed examination of individual chromosomes
    or creating publication-quality figures of specific genomic regions.
    """
    import seaborn as sns
    
    #filter data_analyzed by modtype if provided
    working_df = Data_analyzed if modtype is None else Data_analyzed[Data_analyzed["mod_type"] == modtype]
    working_df = working_df.copy()
    # Derive transcript_id from the chrom token (e.g. "ENST0001|chr1") so labels can use it
    if "transcript_id" not in working_df.columns and "chrom" in working_df.columns:
        working_df["transcript_id"] = working_df["chrom"].apply(
            lambda v: str(v).split("|")[0] if pd.notna(v) else np.nan
        )
    if "positions" not in working_df.columns:
        required_cols = {"chrom", "start", "end", "strand", "mod_type"}
        if required_cols.issubset(working_df.columns):
            working_df["positions"] = (
                working_df["chrom"].astype(str)
                + ":"
                + working_df["start"].astype(str)
                + ":"
                + working_df["end"].astype(str)
                + ":"
                + working_df["strand"].astype(str)
                + ":"
                + working_df["mod_type"].astype(str)
            )
    working_df = _merge_annotation_columns(
        working_df,
        annotation_df,
        key_column="positions",
        value_columns=["transcript_name", "transcript_id", "gene_name", "gene_id"],
    )

    # Fill missing transcript_name from transcript_id mapping if provided in annotation_df
    if (
        annotation_df is not None
        and not annotation_df.empty
        and "transcript_id" in annotation_df.columns
        and "transcript_name" in annotation_df.columns
    ):
        id_to_name = (
            annotation_df.dropna(subset=["transcript_id", "transcript_name"])
            .drop_duplicates(subset=["transcript_id"])
            .set_index("transcript_id")["transcript_name"]
        )
        working_df["transcript_name"] = working_df["transcript_name"].combine_first(
            working_df["transcript_id"].map(id_to_name)
        )

    # Scientific plotting style
    sns.set_theme(style="white", font_scale=1.3, rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    
    if chr != "":
        final_analysis_df = working_df[working_df["chrom"] == chr]
    else:
        final_analysis_df = working_df

    def _compose_label_no_paren(row, fallback_label, identifier_groups):
        """Return identifier:first_available + fallback, joined with ':' instead of parentheses."""
        for fields in identifier_groups:
            values = []
            for field in fields:
                if field in row.index and pd.notna(row[field]):
                    text_value = str(row[field]).strip()
                    if text_value:
                        if text_value not in values:
                            values.append(text_value)
            if values:
                identifier = " / ".join(values)
                return f"{identifier}:{fallback_label}"
        return fallback_label
    
    # Create masks
    blue_mask = (final_analysis_df['mod_freq_diff'] <= -4) & (final_analysis_df['padj'] < alpha)
    red_mask = (final_analysis_df['mod_freq_diff'] >= 4) & (final_analysis_df['padj'] < alpha)
    gray_mask = ~(blue_mask | red_mask)

    # Define colors matching scientific palette
    color_decrease = '#4575B4'  # Blue for decrease
    color_increase = '#D73027'  # Red for increase
    color_ns = '#BDBDBD'  # Gray for not significant

    # Plot gray points (not significant)
    ax.scatter(
        final_analysis_df.loc[gray_mask, 'mod_freq_diff'],
        final_analysis_df.loc[gray_mask, 'neg_log_p'],
        color=color_ns, alpha=0.4, s=30, edgecolor='none', label='Not significant'
    )

    # Plot blue points (significant decrease)
    ax.scatter(
        final_analysis_df.loc[blue_mask, 'mod_freq_diff'],
        final_analysis_df.loc[blue_mask, 'neg_log_p'],
        color=color_decrease, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant decrease'
    )

    # Plot red points (significant increase)
    ax.scatter(
        final_analysis_df.loc[red_mask, 'mod_freq_diff'],
        final_analysis_df.loc[red_mask, 'neg_log_p'],
        color=color_increase, alpha=0.8, s=50, edgecolor='black', linewidth=0.5, label='Significant increase'
    )

    # Annotate top 10 most significant points with simplified labels
    significant_df = final_analysis_df[blue_mask | red_mask].copy()
    if len(significant_df) > 0:
        # Sort by significance and take top 10
        top_sig = significant_df.nlargest(min(10, len(significant_df)), 'neg_log_p')
        texts = []
        for _, row in top_sig.iterrows():
            pos_parts = str(row['positions']).split(':')
            chrom_token = pos_parts[0] if len(pos_parts) >= 1 else str(row['positions'])
            chrom_parts = chrom_token.split('|')
            chrom_clean = chrom_parts[-1] if chrom_parts else chrom_token
            coord_token = pos_parts[1] if len(pos_parts) >= 2 else pos_parts[-1] if pos_parts else ""
            base_label = f"{chrom_clean}:{coord_token}" if coord_token else chrom_clean
            label = _compose_label_no_paren(
                row,
                base_label,
                [["transcript_name"], ["gene_name"], ["transcript_id"], ["gene_id"]],
            )
            texts.append(ax.text(row['mod_freq_diff'], row['neg_log_p'], label, fontsize=6, color='black'))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Decorations - scientific style
    ax.set_xlabel('Modification frequency difference (%)', fontsize=16)
    ax.set_ylabel('-log$_{10}$(p-adjusted)', fontsize=16)
    
    # Threshold lines
    ax.axhline(y=-np.log10(alpha), color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=4, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axvline(x=-4, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    
    ax.set_xlim(-100, 100)
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
    return fig