import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from functools import reduce
import seaborn as sns
import umap

logger = logging.getLogger(__name__)
from scipy.stats import pearsonr
from matplotlib.offsetbox import AnchoredText
from dmode.utility import gtf_to_df, moddict, moddict_reverse, moddict_mapping, moddict_mapping_reverse
from tqdm import tqdm 
import polars as pl
import pyranges as pr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings


def prepare_gene_body_coverage(gtf_file):
    # --- Step 1: Load GTF ---
    gtf_df = gtf_to_df(gtf_file)
    if not isinstance(gtf_df, pl.DataFrame):
        gtf_df = pl.from_pandas(gtf_df)

    gtf_df = gtf_df.with_columns(
    ((pl.col("end") - pl.col("start")).abs() + 1).alias("length"))

    logger.info("Preparing gene body datasets")
    gtf_genes = gtf_df.filter(pl.col("feature") == "gene")
    gtf_trans = gtf_df.filter(pl.col("feature") == "transcript")
    gtf_CDS = gtf_df.filter(pl.col("feature") == "CDS")
    gtf_UTRs = gtf_df.filter(pl.col("feature") == "UTR")

    # --- Step 2: MANE and protein coding selection ---
    logger.info("Finding MANE transcripts and protein-coding genes")

    # MANE_Select transcripts
    gtf_trans_mane = gtf_trans.filter(
        pl.col("tags").str.contains("MANE_Select")
    )

    # Protein coding genes
    genes_pc = gtf_genes.filter(pl.col("gene_type") == "protein_coding")

    # Find protein-coding genes without MANE transcript
    mane_gene_ids = set(gtf_trans_mane["gene_id"].to_list())
    missing_genes = genes_pc.filter(~pl.col("gene_id").is_in(mane_gene_ids))

    # Collect missing genes and add canonical or first transcript
    new_mane_rows = []
    for gid in tqdm(missing_genes["gene_id"].to_list()):
        tset = gtf_trans.filter(pl.col("gene_id") == gid)
        if tset.height == 0:
            continue

        canon = tset.filter(pl.col("tags").str.contains("Ensembl_canonical"))
        new_mane_rows.append(canon if canon.height > 0 else tset.head(1))

    if new_mane_rows:
        gtf_trans_mane = pl.concat([gtf_trans_mane, *new_mane_rows], how="vertical")

    # --- Step 3: Merge CDS and UTRs for selected transcripts ---
    merged_CDS = gtf_CDS.filter(pl.col("transcript_id").is_in(gtf_trans_mane["transcript_id"]))
    merged_UTR = gtf_UTRs.filter(pl.col("transcript_id").is_in(gtf_trans_mane["transcript_id"]))

    logger.info("Annotating 5' and 3' UTRs")

    # Combine CDS + UTR
    all_feats = pl.concat([merged_CDS, merged_UTR], how="vertical").with_columns(
        pl.col("feature").fill_null("UTR")
    )

    # --- Step 4: Annotate transcripts efficiently ---
    def annotate_transcript(df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            # Return empty DataFrame with expected schema
            schema = {
                "seqname": pl.Utf8,
                "feature": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "gene_id": pl.Utf8,
                "gene_name": pl.Utf8,
                "gene_type": pl.Utf8,
                "transcript_id": pl.Utf8,
                "transcript_name": pl.Utf8,
                "strand": pl.Utf8,
                "exon_id": pl.Utf8,
                "exon_number": pl.Utf8,
                "tags": pl.Utf8,
                "length": pl.Int64,
                "feature_end": pl.Int64,
                "feature_start": pl.Int64
            }
            return pl.DataFrame(schema=schema)

        strand = df["strand"][0]
        CDS = df.filter(pl.col("feature") == "CDS")
        UTR = df.filter(pl.col("feature") == "UTR")

        if CDS.height == 0 or UTR.height == 0:
            # Still return empty DataFrame with all columns
            schema = {k: df.schema[k] for k in df.schema.keys()}
            schema["feature_end"] = pl.Int64
            schema["feature_start"] = pl.Int64
            return pl.DataFrame(schema=schema)

        CDS_start = CDS["start"].min()
        CDS_end = CDS["end"].max()

        # Assign 5'/3' UTRs
        if strand == "+":
            UTR = UTR.with_columns(
                pl.when(pl.col("end") <= CDS_start)
                .then(pl.lit("5'UTR"))
                .when(pl.col("start") >= CDS_end)
                .then(pl.lit("3'UTR"))
                .otherwise(pl.col("feature"))
                .alias("feature")
            )
        else:
            UTR = UTR.with_columns(
                pl.when(pl.col("end") <= CDS_start)
                .then(pl.lit("3'UTR"))
                .when(pl.col("start") >= CDS_end)
                .then(pl.lit("5'UTR"))
                .otherwise(pl.col("feature"))
                .alias("feature")
            )

        comp = pl.concat([CDS, UTR], how="vertical").sort("start")
        comp = comp.with_columns([
            pl.col("length").cum_sum().alias("feature_end"),
            (pl.col("length").cum_sum() - pl.col("length") + 1).alias("feature_start")
        ])
        return comp

    # Apply annotation per transcript
    logger.info("Concatenating and annotating transcript features")
    gene_body_df = (
    all_feats
    .group_by("transcript_id", maintain_order=True)
    .map_groups(annotate_transcript)
    )
    return gtf_df, gene_body_df


def metagene_stacked_barplot(df):
    #Set ticklabel format to prevent scietific numbering on y axis
    #plt.ticklabel_format(style='plain', axis='both')
    # Catch warnings during plotting and calculations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # --- Create consistent color mapping for features ---
        features = np.unique(df["Feature"])
        cmap = cm.get_cmap("tab20", len(features))  # categorical colormap
        color_map = {feature: cmap(i) for i, feature in enumerate(features)}

        # --------------------------
        # 1. Per-sample barplots
        # --------------------------
        sample_df = df.with_columns((pl.col("count") / pl.sum("count").over("Sample")).alias("Relative_count"))
        bottom = {sample: 0 for sample in np.unique(sample_df["Sample"])}
        relative_bottom = {sample: 0 for sample in np.unique(sample_df["Sample"])}
        
        sample_df = sample_df.sort("Feature")

        fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

        seen_features = set()

        for sample, feature, count, relative_count in zip(sample_df["Sample"], sample_df["Feature"], sample_df["count"], sample_df["Relative_count"]):
            color = color_map[feature]
            label = feature if feature not in seen_features else None  # only label once
            ax.bar(sample, count, label=label, bottom=bottom[sample], color=color)
            ax2.bar(sample, relative_count, label=label, bottom=relative_bottom[sample], color=color)
            bottom[sample] += count
            relative_bottom[sample] += relative_count
            seen_features.add(feature)

        ax2.legend(bbox_to_anchor=(1.01, 1))
        ax.set_ylabel("Absolute modification site abundance")
        ax2.set_ylabel("Relative modification site abundance")
        ax.set_xlabel("Sample")
        ax2.set_xlabel("Sample")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='minor', labelsize=10)
        # --------------------------
        # 2. Per-condition barplots
        # --------------------------
        # First aggregate per group
        condition_df = (
            df.group_by(["Condition", "Feature"])
            .agg(pl.col("count").sum().alias("count"))
        )

        # Then compute relative counts by Condition
        condition_df = condition_df.with_columns(
            (pl.col("count") / pl.sum("count").over("Condition")).alias("Relative_count")
        )

        condition_df = condition_df.sort("Feature")
        bottom = {cond: 0 for cond in np.unique(condition_df["Condition"])}
        relative_bottom = {cond: 0 for cond in np.unique(condition_df["Condition"])}

        fig2, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

        seen_features = set()

        for condition, feature, count, relative_count in zip(condition_df["Condition"], condition_df["Feature"], condition_df["count"], condition_df["Relative_count"]):
            color = color_map[feature]
            label = feature if feature not in seen_features else None
            ax.bar(condition, count, label=label, bottom=bottom[condition], color=color)
            ax2.bar(condition, relative_count, label=label, bottom=relative_bottom[condition], color=color)
            bottom[condition] += count
            relative_bottom[condition] += relative_count
            seen_features.add(feature)

        ax2.legend(bbox_to_anchor=(1.01, 1))
        ax.set_ylabel("Absolute modification site abundance")
        ax2.set_ylabel("Relative modification site abundance")
        ax.set_xlabel("Condition")
        ax2.set_xlabel("Condition")
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='minor', labelsize=10)
        plt.tight_layout()
        return fig, fig2

def metagene_modification_distribution_plot(df, gene_body_df, conditions, UTR5_dim = 500, UTR3_dim = 500, CDS_dim = 1000, normalization = "max", gene_id = "", mod_type="", coverage_filter = 0, count_type="inclusive"):
    df = df.sort("Gene_id")
    if gene_id != "":
        df = df.filter(pl.col("Gene_id") == gene_id)
    if mod_type != "":
        mod_type = moddict_mapping(mod_type)
        df = df.filter(pl.col("Modified_base_code") == mod_type)
    df = df.filter((pl.col("Nvalid_cov") + pl.col("Nnocall")) >= coverage_filter)
    fig,(ax, ax2) = plt.subplots(nrows = 2, ncols= 1,figsize=(10,10))
    for condition in tqdm(np.unique(conditions), total=len(np.unique(conditions))):
        all_counter = 0
        all_combined_counter = np.zeros(UTR5_dim + CDS_dim + UTR3_dim)
        all_combined_coverage_counter = np.zeros(UTR5_dim + CDS_dim + UTR3_dim)
        for gene_id in tqdm(np.unique(df["Gene_id"]),total = len(np.unique(df["Gene_id"]))):
            transcript_id = df.filter(pl.col("Gene_id") == gene_id)["Transcript_id"][0]
            strandedness = df.filter(pl.col("Gene_id") == gene_id)["Strand"][0]
            condition_df = df.filter((pl.col("Condition") == condition) & (pl.col("Gene_id") == gene_id))
            temp_gene_body = gene_body_df.filter(pl.col("Transcript_id") == transcript_id).sort("Start")
            try:
                if strandedness == "-":
                    UTR5_length = int(temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("Start").min())[0,0] + 1)
                    CDS_length = int(temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("Start").min())[0,0] + 1)
                    UTR3_length = int(temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("Start").min())[0,0] + 1)

                    minuend_UTR5 = int(temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("End").max())[0,0])
                    minuend_CDS = int(temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("End").max())[0,0])
                    minuend_UTR3 = int(temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("End").max())[0,0])
                    exon_UTR5_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "5'UTR")
                            .select([
                                (((minuend_UTR5 - (pl.col("End"))) / UTR5_length) * UTR5_dim).alias("utr5_end"),
                                (((minuend_UTR5 - (pl.col("Start"))) / UTR5_length) * UTR5_dim).alias("utr5_start"),
                            ])
                            .rows()
                    )               
                    exon_CDS_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "CDS")
                            .select([
                                ((((minuend_CDS - (pl.col("End"))) / CDS_length) * CDS_dim) + UTR5_dim).alias("cds_end"),
                                ((((minuend_CDS - (pl.col("Start"))) / CDS_length) * CDS_dim) + UTR5_dim).alias("cds_start"),
                            ])
                            .rows()               
                    )
                    exon_UTR3_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "3'UTR")
                            .select([
                                ((((minuend_UTR3 - (pl.col("End"))) / UTR3_length) * UTR3_dim) + UTR5_dim + CDS_dim).alias("utr3_end"),
                                ((((minuend_UTR3 - (pl.col("Start"))) / UTR3_length) * UTR3_dim) + UTR5_dim + CDS_dim).alias("utr3_start"),
                            ])
                            .rows()               
                    )      

                elif strandedness == "+":
                    UTR5_length = int(temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("Start").min())[0,0] + 1)
                    CDS_length = int(temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("Start").min())[0,0] + 1)
                    UTR3_length = int(temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("End").max())[0,0] - temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("Start").min())[0,0] + 1)
                    
                    subtrahend_UTR5 = int(temp_gene_body.filter(pl.col("Feature") == "5'UTR").select(pl.col("Start").min())[0,0])
                    subtrahend_CDS = int(temp_gene_body.filter(pl.col("Feature") == "CDS").select(pl.col("Start").min())[0,0])
                    subtrahend_UTR3 = int(temp_gene_body.filter(pl.col("Feature") == "3'UTR").select(pl.col("Start").min())[0,0])

                    exon_UTR5_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "5'UTR")
                            .select([
                                ((((pl.col("Start") - subtrahend_UTR5)) / UTR5_length) * UTR5_dim).alias("utr5_start"),
                                ((((pl.col("End") - subtrahend_UTR5)) / UTR5_length) * UTR5_dim).alias("utr5_end"),
                            ])
                            .rows()
                    )               
                    exon_CDS_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "CDS")
                            .select([
                                (((((pl.col("Start") - subtrahend_CDS)) / CDS_length) * CDS_dim) + UTR5_dim).alias("cds_start"),
                                (((((pl.col("End") - subtrahend_CDS)) / CDS_length) * CDS_dim) + UTR5_dim).alias("cds_end"),
                            ])
                            .rows()               
                    )
                    exon_UTR3_boundaries = (
                        temp_gene_body
                            .filter(pl.col("Feature") == "3'UTR")
                            .select([
                                (((((pl.col("Start") - subtrahend_UTR3)) / UTR3_length) * UTR3_dim) + UTR5_dim + CDS_dim).alias("utr3_start"),
                                (((((pl.col("End") - subtrahend_UTR3)) / UTR3_length) * UTR3_dim) + UTR5_dim + CDS_dim).alias("utr3_end"),
                            ])
                            .rows()
                    )
            except TypeError:
                continue
            #Collect unique exon boundaries and sort by ascending start positions
            exon_UTR5_boundaries = [(int(i),int(j)) for i,j in sorted(set(exon_UTR5_boundaries), key=lambda x: x[0])]
            exon_CDS_boundaries = [(int(i),int(j)) for i,j in sorted(set(exon_CDS_boundaries), key=lambda x: x[0])]
            exon_UTR3_boundaries = [(int(i),int(j)) for i,j in sorted(set(exon_UTR3_boundaries), key=lambda x: x[0])]
            # print(strandedness)
            # print("UTR5")
            # print(exon_UTR5_boundaries)
            # print("CDS")
            # print(exon_CDS_boundaries)
            # print("UTR3")
            # print(exon_UTR3_boundaries)
            if condition_df.height == 0:
                continue
            single_counter = np.zeros(UTR5_dim + CDS_dim + UTR3_dim)
            single_coverage_counter = np.zeros(UTR5_dim + CDS_dim + UTR3_dim)
            last_sample = str(condition_df["Sample"][0])
            for row in condition_df.iter_rows(named=True):
                #Extract relevant info from row
                feature = row["Feature"]
                start_mod = row["Start"]
                start_gene = row["Start_gene"]
                end_gene = row["End_gene"]
                start_full_body = row["Start_full_body"]
                end_full_body = row["End_full_body"]
                nvalid = row["Nvalid_cov"]
                nnocall = row["Nnocall"]
                sample = row["Sample"]
                fail = row["Nfail"]
                strand = row["Strand"]
                coverage = nvalid + nnocall + fail 
                if strand == "-":
                    ##########################################################
                    ## Example of reverse strand adjustment                  #
                    ## Reads are mapped 5' to 3' on the forward strand       # 
                    ## 3'UTR has lower coordinates than CDS and 5'UTR        #
                    ## Thus, need to invert positions                        # 
                    ## The end of 5'UTR must become position 0 and           #
                    ## the start of 3'UTR the full length of the gene body   #
                    ## 950 -- 5'UTR -- mod at 1000 -- 1200                   #
                    ## Mod position -> 1200 - 1000 == 200                    #
                    ## 5'UTR start -> 0 or (1200 - 1200)                     #
                    ## 5'UTR end -> 1200 - 950 == 250                        #
                    ##########################################################
                    #Prepare copies of values
                    end_full_body_temp = int(end_full_body)
                    start_full_body_temp = int(start_full_body)
                    start_mod_temp = int(start_mod)
                    start_gene_temp = int(start_gene)
                    end_gene_temp = int(end_gene)
                    #Change values
                    end_full_body = end_full_body_temp - start_full_body_temp
                    start_full_body = 0
                    start_mod = end_full_body_temp - start_mod_temp
                    start_gene = end_full_body_temp - start_gene_temp
                    end_gene = end_full_body_temp - end_gene_temp
                    length = end_full_body_temp - start_full_body_temp
                    
                elif strand == "+":
                    ######################################################
                    ## Example of forward strand adjustment              #
                    ## Reads are mapped 5' to 3' on the forward strand   #
                    ## 5'UTR has lower coordinates than CDS and 3'UTR    #
                    ## Thus, need to shift positions to start at 0       #
                    ## 100 -- mod at 300 -- 5'UTR -- 350                 #
                    ## Mod position -> 300 - 100 == 200                  #
                    ## 5'UTR start -> 0 or (100 - 100)                   # 
                    ## 5'UTR end -> 350 - 100 == 250                     #
                    ######################################################
                    #Prepare copies of values
                    end_full_body_temp = int(end_full_body)
                    start_full_body_temp = int(start_full_body)
                    start_mod_temp = int(start_mod)
                    start_gene_temp = int(start_gene)
                    end_gene_temp = int(end_gene)
                    end_full_body = end_full_body_temp - start_full_body_temp
                    #Change values
                    start_full_body = 0
                    start_mod = start_mod_temp - start_full_body_temp
                    start_gene = start_gene_temp - start_full_body_temp
                    end_gene = end_gene_temp - start_full_body_temp
                    length = end_full_body_temp - start_full_body_temp

                #Positions were adjusted, now proceed as normal
                rel_position = ((start_mod - start_full_body) / (length))
                rel_exon_start = ((start_gene - start_full_body) / (length))
                rel_exon_end = ((end_gene - start_full_body) / (length))

                #Define shift for concatenated array (5'UTR, CDS, 3'UTR)
                #if count_type == "inclusive":
                if feature == "5'UTR":
                    shift = 0
                    dim = UTR5_dim
                elif feature == "CDS":
                    shift = UTR5_dim
                    dim = CDS_dim
                elif feature == "3'UTR":
                    shift = UTR5_dim + CDS_dim
                    dim = UTR3_dim
                
                # Augment modification counter arrays
                single_counter[shift + min(max(int(rel_position * dim),0), dim-1)] += 1
                if single_coverage_counter[shift + min(max(int(rel_position * dim),0), dim-1)] == 0 or last_sample != sample:
                    last_sample = sample
                    single_coverage_counter[shift + min(max(int(rel_position * dim),0), dim-1)] += coverage
                exon_start = shift + min(max(int(rel_exon_start * dim),0), dim-1)
                exon_end = shift + min(max(int(rel_exon_end * dim), 0), dim-1)
  
            if count_type == "exclusive":
                UTR5_split_counter = []
                UTR5_split_coverage_counter = []
                for exon in exon_UTR5_boundaries:
                    exon_start = exon[0]
                    exon_end = exon[1]
                    UTR5_split_counter  = UTR5_split_counter + list(single_counter[exon_start:exon_end + 1])
                    UTR5_split_coverage_counter = UTR5_split_coverage_counter + list(single_coverage_counter[exon_start:exon_end + 1])
                original_size = np.linspace(0, 1, num=len(UTR5_split_counter))
                target_size = np.linspace(0, 1, num=UTR5_dim)
                if len(UTR5_split_counter) == 0:
                    UTR5_split_counter = [0] * UTR5_dim
                    UTR5_split_coverage_counter = [0] * UTR5_dim
                    single_counter[0:UTR5_dim] = UTR5_split_counter
                    single_coverage_counter[0:UTR5_dim] = UTR5_split_coverage_counter 
                else:
                    UTR5_split_counter  = np.interp(target_size, original_size, UTR5_split_counter)
                    UTR5_split_coverage_counter = np.interp(target_size, original_size, UTR5_split_coverage_counter)
                    single_counter[0:UTR5_dim] = UTR5_split_counter 
                    single_coverage_counter[0:UTR5_dim] = UTR5_split_coverage_counter 
                    
                CDS_split_counter = []
                CDS_split_coverage_counter = []
                for exon in exon_CDS_boundaries:
                    exon_start = exon[0]
                    exon_end = exon[1]
                    CDS_split_counter = CDS_split_counter + list(single_counter[exon_start:exon_end + 1])
                    CDS_split_coverage_counter = CDS_split_coverage_counter + list(single_coverage_counter[exon_start:exon_end + 1])
                    
                original_size = np.linspace(0, 1, num=len(CDS_split_counter))
                target_size = np.linspace(0, 1, num=CDS_dim)
                if len(CDS_split_counter) == 0:
                    CDS_split_counter = [0]*CDS_dim
                    CDS_split_coverage_counter = [0]*CDS_dim
                    single_counter[UTR5_dim:UTR5_dim + CDS_dim] = CDS_split_counter
                    single_coverage_counter[UTR5_dim:UTR5_dim + CDS_dim] = CDS_split_coverage_counter 
                else:
                    CDS_split_counter = np.interp(target_size, original_size, CDS_split_counter)
                    CDS_split_coverage_counter = np.interp(target_size, original_size, CDS_split_coverage_counter)
                    single_counter[UTR5_dim:UTR5_dim + CDS_dim] = CDS_split_counter
                    single_coverage_counter[UTR5_dim:UTR5_dim + CDS_dim] = CDS_split_coverage_counter     
                    
                UTR3_split_counter = []
                UTR3_split_coverage_counter = []
                if len(exon_UTR3_boundaries) == 0:
                    continue
                for exon in exon_UTR3_boundaries:
                    exon_start = exon[0]
                    exon_end = exon[1]
                    UTR3_split_counter  = UTR3_split_counter  + list(single_counter[exon_start:exon_end + 1])
                    UTR3_split_coverage_counter = UTR3_split_coverage_counter + list(single_coverage_counter[exon_start:exon_end + 1])
                original_size = np.linspace(0, 1, num=len(UTR3_split_counter ))
                target_size = np.linspace(0, 1, num=UTR3_dim)
                if len(UTR3_split_counter) == 0:
                    UTR3_split_counter = [0]*UTR3_dim
                    UTR3_split_coverage_counter = [0]*UTR3_dim
                    single_counter[UTR5_dim + CDS_dim:UTR5_dim + CDS_dim + UTR3_dim] = UTR3_split_counter 
                    single_coverage_counter[UTR5_dim + CDS_dim:UTR5_dim + CDS_dim + UTR3_dim] = UTR3_split_coverage_counter  
                else:
                    UTR3_split_counter  = np.interp(target_size, original_size, UTR3_split_counter )
                    UTR3_split_coverage_counter = np.interp(target_size, original_size, UTR3_split_coverage_counter)
                    single_counter[UTR5_dim + CDS_dim:UTR5_dim + CDS_dim + UTR3_dim] = UTR3_split_counter 
                    single_coverage_counter[UTR5_dim + CDS_dim:UTR5_dim + CDS_dim + UTR3_dim] = UTR3_split_coverage_counter    

            # Normalize single counters
            if normalization == "max":
                single_counter_max = max(single_counter)
                single_coverage_counter_max = max(single_coverage_counter)
            elif normalization == "sum":
                single_counter_max = sum(single_counter)
                single_coverage_counter_max = sum(single_coverage_counter)
            elif normalization == "mean":
                single_counter_max = np.mean(np.array(single_counter)) 
                single_coverage_counter_max = np.mean(np.array(single_coverage_counter))   

            if not single_counter_max == 0 and not single_coverage_counter_max == 0:
                relative_combined_counter = np.array(single_counter) / single_counter_max
                relative_combined_coverage_counter = np.array(single_coverage_counter) / single_coverage_counter_max
                all_combined_counter = np.array(all_combined_counter) + np.array(relative_combined_counter)
                all_combined_coverage_counter = np.array(all_combined_coverage_counter) + np.array(relative_combined_coverage_counter)
                all_counter += 1
            else:
                continue

        relative_all_combined_counter = all_combined_counter / all_counter  
        relative_all_combined_coverage_counter = all_combined_coverage_counter / all_counter
        ax.plot([i for i in range(len(relative_all_combined_counter))],relative_all_combined_counter, label = condition)
        ax2.plot([i for i in range(len(relative_all_combined_coverage_counter))],relative_all_combined_coverage_counter, label = condition)

    
    
    # Vertical lines for boundaries
    ax.axvline(x=UTR5_dim - 1, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=UTR5_dim + CDS_dim - 1, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=UTR5_dim - 1, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=UTR5_dim + CDS_dim - 1, color='black', linestyle='--', linewidth=1)

    # Add region labels on x-axis
    ax.set_xticks([
        UTR5_dim / 2,                             # middle of 5′UTR
        UTR5_dim + CDS_dim / 2,                   # middle of CDS
        UTR5_dim + CDS_dim + UTR3_dim / 2         # middle of 3′UTR
    ])
    ax2.set_xticks([
        UTR5_dim / 2,                             # middle of 5′UTR
        UTR5_dim + CDS_dim / 2,                   # middle of CDS
        UTR5_dim + CDS_dim + UTR3_dim / 2         # middle of 3′UTR
    ])
    
    ax.set_xticklabels(["5′UTR", "CDS", "3′UTR"])
    ax2.set_xticklabels(["5′UTR", "CDS", "3′UTR"])

    # Optional: extend x-limits to fit the labels nicely
    total_length = UTR5_dim + CDS_dim + UTR3_dim
    #if normalization == "max" or normalization == "sum":
    #    ax.set_ylim(0,1)
    #    ax2.set_ylim(0,1)
    ax.set_xlim(0,total_length)
    ax2.set_xlim(0, total_length)
    ax.set_ylabel("Relative modification site abundance")
    ax2.set_ylabel("Relative coverage at modification sites")

    # Legend
    ax2.legend(bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    return fig


def metagene_body_coverage(gtf_df, gene_body_df, data_files, conditions, samplenames, normalization="max", coverage_filter=0, mod_type="", gene_id="", output_folder=""):
    # Keep only necessary columns to reduce memory
    gene_body_df = gene_body_df.select([
        "seqname", "start", "end", "feature", "strand",
        "gene_id", "gene_name", "gene_type", "transcript_id","feature_start", "feature_end"
    ])
    
    gene_body_df.columns = [
        "Chromosome", "Start", "End", "Feature", "Strand",
        "Gene_id", "Gene_name", "Gene_type", "Transcript_id", "Feature_start", "Feature_end"
    ]

    # Convert gene body data to PyRanges (efficient interval representation)
    gene_body_pr = pr.PyRanges(
        gene_body_df.to_pandas()

    )

    #For forward strand genes, get min start and max end per gene
    full_gene_body_df = (
        gene_body_df.group_by(["Gene_id","Feature"])
        .agg(
            [
                pl.col("Chromosome").first().alias("Chromosome"),
                pl.col("Start").min(),
                pl.col("End").max()
            ]
        )
    )
    
    
    full_gene_body_pr = pr.PyRanges(
        full_gene_body_df.to_pandas()
    )


    results = []
    feature_distributions = []
    logger.info("Extracting modkit data and intersecting with gene body features")
    for data_file, condition, sample in tqdm(zip(data_files, conditions, samplenames), total= len(data_files)):
        modkit_df = pl.read_csv(
            data_file,
            separator="\t",
            has_header=False,
            new_columns=[
                "Chromosome", "Start", "End", "Modified_base_code",
                "Score", "Strand", "Start2", "End2", "Color",
                "Nvalid_cov", "Fraction_modified", "Nmod",
                "Ncanonical", "Nother_mod", "Ndelete",
                "Nfail", "Ndiff", "Nnocall"
            ],
        ).drop("Start2","End2","Color","Score","Ncanonical","Nother_mod","Ndelete","Ndiff")
        # Convert modification data to PyRanges
        modkit_pr = pr.PyRanges(
            modkit_df.to_pandas()
        )

        # Efficient genomic overlap join
        overlap_pr = modkit_pr.join(gene_body_pr, suffix="_gene")
        overlap_pr = overlap_pr.join(full_gene_body_pr, suffix="_full_body")

        # Convert back to Polars DataFrame
        overlap_df = pl.DataFrame(overlap_pr.df)

        # Compute relative positions on template
        overlap_df = overlap_df.with_columns([
            (pl.col("Start") - pl.col("Start_gene")).alias("Start_on_template"),
            (pl.col("End") - pl.col("Start_gene")).alias("End_on_template"),
            pl.lit(condition).alias("Condition"),
            pl.lit(sample).alias("Sample")
        ])
        
        
        feature_value_counts_df = overlap_df.with_columns(
            (pl.col("Chromosome") + ":" +
            pl.col("Start").cast(pl.Utf8) + ":" +
            pl.col("End").cast(pl.Utf8) + ":" +
            pl.col("Modified_base_code")
            ).alias("Position_token")
        )
        feature_value_counts_df = feature_value_counts_df.unique(subset=["Position_token"], keep="first")
        feature_value_counts_df = feature_value_counts_df.group_by(["Sample","Feature"]).count()
        feature_value_counts_df = feature_value_counts_df.with_columns(pl.lit(condition).alias("Condition"))
        feature_distributions.append(feature_value_counts_df)
        results.append(overlap_df)

    # Concatenate all results
    results = [df.with_columns([df[col].cast(pl.Utf8) for col in df.columns if df[col].dtype == pl.Categorical]) for df in results]
    feature_distributions = [df.with_columns([df[col].cast(pl.Utf8) for col in df.columns if df[col].dtype == pl.Categorical]) for df in feature_distributions]
    interpolated_transcripts_df = pl.concat(results)
    interpolated_transcripts_df = interpolated_transcripts_df.with_columns(
         (pl.col("Chromosome") + ":" +
          pl.col("Start").cast(pl.Utf8) + ":" +
          pl.col("End").cast(pl.Utf8) + ":" +
          pl.col("Modified_base_code")
         ).alias("Position_token")
     )

    feature_distribution_df = pl.concat(feature_distributions)
    fig, fig2 = metagene_stacked_barplot(feature_distribution_df)
    #Intronic regions included
    fig3 = metagene_modification_distribution_plot(interpolated_transcripts_df, gene_body_df, conditions, UTR5_dim = 50, UTR3_dim = 50, CDS_dim = 100, normalization = normalization, coverage_filter= coverage_filter, gene_id=gene_id, mod_type=mod_type, count_type="inclusive")
    #Exonic regions only
    fig4 = metagene_modification_distribution_plot(interpolated_transcripts_df, gene_body_df, conditions, UTR5_dim = 50, UTR3_dim = 50, CDS_dim = 100, normalization = normalization, coverage_filter= coverage_filter, gene_id=gene_id, mod_type=mod_type, count_type="exclusive")
    if output_folder != "":  
        if not os.path.exists(os.path.join(output_folder,f"statistics")):
            os.makedirs(os.path.join(output_folder,f"statistics"), exist_ok=True)

        mod_type_converted = str(moddict_mapping_reverse(mod_type))

        if mod_type != "":
            os.makedirs(os.path.join(output_folder,f"statistics",mod_type_converted), exist_ok=True)

        if gene_id != "" and mod_type != "":
            output_name = os.path.join(output_folder,f"statistics",mod_type_converted,f"{gene_id}_{mod_type_converted}")
        elif gene_id != "":
            output_name = os.path.join(output_folder,f"statistics",mod_type_converted,f"{gene_id}_{mod_type_converted}")
        elif mod_type != "":
            output_name = os.path.join(output_folder,f"statistics",mod_type_converted,f"all_genes_{mod_type_converted}")
        else:
            output_name = os.path.join(output_folder,f"statistics",f"all_genes_all_modtypes")
        fig.savefig(f"{output_name}_absolute_stacked_barplot_samples.svg", format="svg")
        fig.savefig(f"{output_name}_relative_stacked_barplot_samples.png", format="png")
        fig2.savefig(f"{output_name}_absolute_stacked_barplot_conditions.svg", format="svg")
        fig2.savefig(f"{output_name}_relative_stacked_barplot_conditions.png", format="png")
        fig3.savefig(f"{output_name}_metagene_plot_including_intronic_regions.svg", format="svg")
        fig3.savefig(f"{output_name}_metagene_plot_including_intronic_regions.png", format="png")
        fig4.savefig(f"{output_name}_metagene_plot_exonic_regions.svg", format="svg")
        fig4.savefig(f"{output_name}_metagene_plot_exonic_regions.png", format="png")

    return interpolated_transcripts_df, feature_distribution_df, fig, fig2, fig3, fig4