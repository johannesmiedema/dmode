import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import polars as pl



def convert_bed_to_df(bed_file):
    """
    Convert BED file intersected with GTF annotations to pandas DataFrame.
    
    This function processes the output from genomic intersection tools (like bedtools)
    that combine BED format genomic positions with GTF annotation information,
    parsing the combined data into a structured DataFrame.
    
    Parameters
    ----------
    bed_file : object
        File object or iterator containing intersected BED+GTF features.
        Each feature should have a .fields attribute containing the combined data.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with parsed BED and GTF information containing:
        - BED columns: 'chrom', 'start', 'end', 'strand', 'mod_type'
        - GTF columns: 'gtf_chrom', 'source', 'feature', 'gtf_start', 'gtf_end',
                      'score', 'strand', 'frame', 'attributes'
        - Parsed attributes: 'gene_id', 'gene_name' (and other GTF attributes)
        
    Notes
    -----
    - Assumes input contains 5 BED fields + 9 standard GTF fields
    - Parses GTF attributes field to extract gene_id, gene_name, etc.
    - GTF attributes are expected in format: key "value"; key "value"; ...
    - Useful for analyzing genomic modifications in the context of gene annotations
    - Handles the complex parsing of GTF attribute strings automatically
    
    Examples
    --------
    >>> # After running: bedtools intersect -a modifications.bed -b genes.gtf
    >>> intersected_df = Convert_bed_to_df(intersected_file)
    >>> print(intersected_df[['chrom', 'start', 'gene_name']].head())
    
    This function is typically used in genomic analysis pipelines where you need
    to combine modification data with gene annotation information.
    """
    # Convert each line (feature) into a list of fields
    data = [feature.fields for feature in bed_file]

    # If you know how many columns were in your original BED input,
    # you can separate your fields into: BED fields + GTF fields

    # Example: Original BED had 3 columns: chrom, start, end
    # GTF has 9 standard columns
    bed_cols = ['chrom', 'start', 'end','strand','mod_type']
    gtf_cols = [
        'gtf_chrom', 'source', 'feature', 'gtf_start', 'gtf_end',
        'score', 'strand', 'frame', 'attributes'
    ]

    # Combine into one DataFrame
    df = pd.DataFrame(data, columns=bed_cols + gtf_cols)

    # Optional: parse the GTF attributes to extract gene_id, gene_name
    def parse_attributes(attr_str):
        attrs = {}
        for item in attr_str.strip().split(";"):
            if item.strip():
                key, value = item.strip().split(" ", 1)
                attrs[key] = value.strip('"')
        return pd.Series(attrs)

    # Add gene_id, gene_name columns
    attributes_df = df['attributes'].apply(parse_attributes)
    df = pd.concat([df, attributes_df], axis=1)

    return df


def parse_gtf(filepath_or_buffer, split_attributes=True, features=None):
    """
    Parse GTF file into a pandas DataFrame, using Polars for fast I/O and processing.
    """

    required_columns = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attributes"
    ]

    # --- Read the file efficiently with Polars ---
    df = pl.read_csv(
        filepath_or_buffer,
        separator="\t",
        has_header=False,
        new_columns=required_columns,
        comment_prefix="#",
        null_values=".",
        infer_schema_length=1000,
        dtypes={
            "seqname": pl.Utf8,
            "source": pl.Utf8,
            "feature": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "score": pl.Float64,
            "strand": pl.Utf8,
            "frame": pl.Utf8,
            "attributes": pl.Utf8
        },
        low_memory=True
    )
    # Fill NA frame values with "0"
    df = df.with_columns(pl.col("frame").fill_null("0"))

    # Optional feature filtering
    if features is not None:
        df = df.filter(pl.col("feature").is_in(features))

    # --- Split GTF attributes manually (vectorized loop) ---
    if split_attributes:

        gene_ids = []
        transcript_ids = []
        transcript_names = []
        gene_names = []
        gene_types = []
        exon_ids = []
        exon_numbers = []
        tags = []

        for feature, attr in tqdm(
            zip(df["feature"].to_list(), df["attributes"].to_list()),
            total=df.height
        ):
            # Split and initialize
            if not isinstance(attr, str):
                attr = ""
            parts = attr.split(";")
            tag_str = ""

            gene_id = transcript_id = transcript_name = gene_name = gene_type = ""
            exon_id = exon_number = ""

            for el in parts:
                el = el.strip()
                if not el or " " not in el:
                    continue
                key, val = el.split(" ", 1)
                val = val.replace('"', "").strip()

                if key == "gene_id":
                    gene_id = val
                elif key == "transcript_id":
                    transcript_id = val
                elif key == "transcript_name":
                    transcript_name = val
                elif key == "gene_name":
                    gene_name = val
                elif key == "gene_type":
                    gene_type = val
                elif key == "exon_id":
                    exon_id = val
                elif key == "exon_number":
                    exon_number = val
                elif key == "tag":
                    tag_str += f"tag:{val}:"

            # Fill missing fields depending on feature type
            if feature == "gene":
                transcript_id = transcript_name = exon_id = exon_number = ""
            elif feature in ["transcript", "Selenocysteine"]:
                exon_id = exon_number = ""

            gene_ids.append(gene_id)
            transcript_ids.append(transcript_id)
            transcript_names.append(transcript_name)
            gene_names.append(gene_name)
            gene_types.append(gene_type)
            exon_ids.append(exon_id)
            exon_numbers.append(exon_number)
            tags.append(tag_str)

        df = df.with_columns([
            pl.Series("gene_id", gene_ids),
            pl.Series("transcript_id", transcript_ids),
            pl.Series("transcript_name", transcript_names),
            pl.Series("gene_type", gene_types),
            pl.Series("gene_name", gene_names),
            pl.Series("exon_id", exon_ids),
            pl.Series("exon_number", exon_numbers),
            pl.Series("tags", tags)
        ])

    df = df.drop("attributes")

    # Return pandas DataFrame for compatibility
    return df.to_pandas()


def gtf_to_df(gtf_file: str):
    """
    Parse a GTF annotation file into a pandas DataFrame using Polars internally.
    """
    gtf_file = os.path.abspath(gtf_file)
    gtf_df = parse_gtf(gtf_file)

    # Select relevant columns
    gtf_df = gtf_df[
        [
            "seqname", "feature", "start", "end",
            "gene_id", "gene_name", "gene_type",
            "transcript_id", "transcript_name",
            "strand", "exon_id", "exon_number", "tags"
        ]
    ]
    return gtf_df


def moddict():
    BASE_MODIFICATION_TYPE_DICT = {
    "m6A": "a",
    "Am": "69426", 
    "Ino": "17596",
    "pseU": "17802",
    "Um": "19227",
    "Gm": "19229",
    "Cm": "19228",
    "m5C": "m",
    "a": "a",
    "69426": "69426",
    "17596": "17596",
    "17802": "17802",
    "19227": "19227",
    "19229": "19229",
    "19228": "19228",
    "m": "m"
    }
    return BASE_MODIFICATION_TYPE_DICT

def moddict_reverse():
    BASE_MODIFICATION_TYPE_DICT = {
    "a": "m6A",
    "69426": "Am", 
    "17596": "Ino",
    "17802": "pseU",
    "19227": "Um",
    "19229": "Gm",
    "19228": "Cm",
    "m": "m5C",
    "m6A": "m6A",
    "Am": "Am",
    "Ino": "Ino",
    "pseU": "pseU",
    "Um": "Um",
    "Gm": "Gm",
    "Cm": "Cm",
    "m5C": "m5C"
    }
    return BASE_MODIFICATION_TYPE_DICT

def moddict_reverse():
    BASE_MODIFICATION_TYPE_DICT = {
    "a": "m6A",
    "69426": "Am", 
    "17596": "Ino",
    "17802": "pseU",
    "19227": "Um",
    "19229": "Gm",
    "19228": "Cm",
    "m": "m5C",
    "m6A": "m6A",
    "Am": "Am",
    "Ino": "Ino",
    "pseU": "pseU",
    "Um": "Um",
    "Gm": "Gm",
    "Cm": "Cm",
    "m5C": "m5C"
    }
    return BASE_MODIFICATION_TYPE_DICT

def moddict_mapping(key:str):
    return moddict().get(key, key)

def moddict_mapping_reverse(key:str):
    return moddict_reverse().get(key, key)