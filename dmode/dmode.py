import logging

import pandas as pd
from tqdm import tqdm
from dmode.differential_expression_analysis import (merge_single_sample_feature_count_tables,
                                                    merge_single_sample_salmon_count_tables)

logger = logging.getLogger(__name__)

class DmodE:
    """
    Differential Modification Expression (DmodE) analysis tool for genomic modification data.
    
    This class provides functionality for analyzing differential DNA/RNA modifications 
    (such as methylation) between different experimental conditions. It supports 
    preprocessing of modification data, statistical testing, and identification of 
    significantly modified genomic positions and genes.
    
    Attributes
    ----------
    data_files : list
        List of file paths containing modification data.
    conditions : list
        List of condition labels corresponding to each data file.
    reference_levels : list
        List of reference condition levels for comparisons.
    comparisons : list of tuples
        All pairwise comparisons between reference and condition levels.
    chr_names : list
        List of chromosome names (chr1-chr22, chrX, chrY).
    dataframe_condition_dict : dict
        Dictionary mapping conditions to their processed dataframes.
    comparisons_dict : dict
        Dictionary containing results of differential modification analysis.
        
    Examples
    --------
    >>> # Initialize with metadata file
    >>> dme = DmodE(metadata="samples.tsv", reference_levels=["control"])
    
    >>> # Or initialize manually
    >>> dme = DmodE(
    ...     data_files=["sample1.bed", "sample2.bed"],
    ...     conditions=["control", "treatment"], 
    ...     reference_levels=["control"]
    ... )
    
    >>> # Preprocess data and run analysis
    >>> dme.Preprocess_Data(coverage_filter=5, mod_freq_filter=0.1)
    >>> results = dme.diff_gene_mod_expression(correction_method="SLIM")
    """
    def __init__(
        self
    ):
        return None


    def gene_mod_preprocess_data(self, 
        metadata: str = None,
        data_files: str = None,
        conditions: list = None,
        samplenames: list = None,
        reference_levels: list = None,
        coverage_filter:int = 20, 
        mod_freq_filter:float = 10,
        comparison: bool = True,
        ) -> dict:
        """
        Load and preprocess modification data files with filtering criteria.
        
        This method reads all modification data files, applies coverage and modification
        frequency filters, and organizes the data by experimental condition. Each
        genomic position is assigned a unique position token for cross-sample matching.
        
        Parameters
        ----------
        metadata : str, optional
            Path to tab-separated metadata file containing 'File' and 'Condition' columns.
            If provided, data_files and conditions are read from this file.
        data_files : list of str, optional
            List of file paths to modification data files. Required if metadata is None.
        conditions : list of str, optional
            List of condition labels for each data file. Required if metadata is None.
        samplenames : list of str, optional
            List of sample names for each data file. Required if metadata is None.
        reference_levels : list of str, optional
            List of condition labels to use as reference levels in comparisons.
            Required only if comparison is True.
        coverage_filter : int
            Minimum read coverage required for a position to be included in analysis.
        mod_freq_filter : float
            Minimum modification frequency (0-1) required for inclusion.
        comparison : bool, default=True
            If True, sets up reference levels and comparisons for differential analysis.
            If False, only preprocesses data without comparison setup.
            
        Returns
        -------
        self
            Returns self with processed data stored in attributes.
            
        Notes
        -----
        Input files are expected to be tab-separated with the following columns:
        - chrom, start position, end position, modified base code, score, strand,
        - start position 2, end position 2, color, Nvalid_cov, fraction modified,
        - Nmod, Ncanonical, Nother_mod, Ndelete, Nfail, Ndiff, Nnocall
        
        The method adds a 'position_token' column combining genomic coordinates,
        strand, and modification type for unique position identification.
        
        Attributes set depend on comparison parameter:
        - If comparison=True: Uses 'diff_gene_mod_' prefix
        - If comparison=False: Uses 'basic_gene_mod_' prefix
        """
        prefix = 'diff_gene_mod_' if comparison else 'basic_gene_mod_'
        
        if metadata == None:
            setattr(self, f'{prefix}data_files', data_files)
            setattr(self, f'{prefix}conditions', conditions)
            setattr(self, f'{prefix}samplenames', samplenames)
        else:
            metadata_df = pd.read_csv(metadata, sep="\t", header=0)
            setattr(self, f'{prefix}data_files', list(metadata_df["File"]))
            setattr(self, f'{prefix}conditions', [str(i) for i in metadata_df["Condition"]])
            setattr(self, f'{prefix}samplenames', [str(i) for i in metadata_df["Sample"]])
            
        data_files_attr = getattr(self, f'{prefix}data_files')
        conditions_attr = getattr(self, f'{prefix}conditions')
        samplenames_attr = getattr(self, f'{prefix}samplenames')
        
        # Setup comparisons only if comparison mode is enabled
        if comparison:
            setattr(self, f'{prefix}reference_levels', reference_levels)
            comparisons = []
            for ref_level in reference_levels:
                for cond_level in list(set(conditions_attr)):
                    if not ref_level == cond_level:
                        comparisons.append((cond_level, ref_level))
            setattr(self, f'{prefix}comparisons', comparisons)

        # Read all files with progress feedback
        all_dataframes = {}
        bed_desc = "Preprocessing gene modification BED files"
        for fp in tqdm(data_files_attr, desc=bed_desc, unit="file"):
            logger.info(f"Preprocessing BED file: {fp}")
            df = pd.read_csv(fp, delimiter="\t", header=None)
            df.columns = [
                "chrom",
                "start position",
                "end position",
                "modified base code",
                "score",
                "strand",
                "start position 2",
                "end position 2",
                "color",
                "Nvalid_cov",
                "fraction modified",
                "Nmod",
                "Ncanonical",
                "Nother_mod",
                "Ndelete",
                "Nfail",
                "Ndiff",
                "Nnocall",
            ]
            all_dataframes[fp] = df
        dataframe_condition_dict = {}
        for idx, (df_Data, condition, sample) in enumerate(
            zip(all_dataframes, conditions_attr, samplenames_attr)
        ):
            dataframe_condition_dict[condition] = {}
        for idx, (df_Data, condition, sample) in enumerate(
            zip(all_dataframes, conditions_attr, samplenames_attr)
        ):  
            #Filter dataframe by coverage and modification frequency
            df_Data = all_dataframes[df_Data]
            df_Data_filtered = df_Data[
                (df_Data["Nvalid_cov"] >= coverage_filter)
                & (df_Data["fraction modified"] >= mod_freq_filter)
            ]
            df_Data_filtered = df_Data_filtered.copy()
            df_Data_filtered["position_token"] = [
                f"{chrom}:{start}:{end}:{strand}:{mod_type}"
                for chrom, start, end, strand, mod_type in zip(
                    df_Data_filtered["chrom"],
                    df_Data_filtered["start position"],
                    df_Data_filtered["end position"],
                    df_Data_filtered["strand"],
                    df_Data_filtered["modified base code"],
                )
            ]
            dataframe_condition_dict[condition][sample]=df_Data_filtered
        setattr(self, f'{prefix}dataframe_condition_dict', dataframe_condition_dict)

        # Chromosome processing
        possible_chromosomes = []
        for chr_df in list(all_dataframes.values()):
            possible_chromosomes = possible_chromosomes + list(chr_df["chrom"])
        setattr(self, f'{prefix}chr_names', list(reversed(sorted(list(set(possible_chromosomes))))))
        return self
    
    def transcript_mod_preprocess_data(self, 
        metadata: str = None,
        data_files: str = None,
        conditions: list = None,
        samplenames: list = None,
        reference_levels: list = None,
        coverage_filter:int = 20, 
        mod_freq_filter:float = 10,
        comparison: bool = True) -> dict:
        """
        Load and preprocess modification data files with filtering criteria.
        
        This method reads all modification data files, applies coverage and modification
        frequency filters, and organizes the data by experimental condition. Each
        genomic position is assigned a unique position token for cross-sample matching.
        
        Parameters
        ----------
        metadata : str, optional
            Path to tab-separated metadata file containing 'File' and 'Condition' columns.
            If provided, data_files and conditions are read from this file.
        data_files : list of str, optional
            List of file paths to modification data files. Required if metadata is None.
        conditions : list of str, optional
            List of condition labels for each data file. Required if metadata is None.
        samplenames : list of str, optional
            List of sample names for each data file. Required if metadata is None.
        reference_levels : list of str, optional
            List of condition labels to use as reference levels in comparisons.
            Required only if comparison is True.
        coverage_filter : int
            Minimum read coverage required for a position to be included in analysis.
        mod_freq_filter : float
            Minimum modification frequency (0-1) required for inclusion.
        comparison : bool, default=True
            If True, sets up reference levels and comparisons for differential analysis.
            If False, only preprocesses data without comparison setup.
            
        Returns
        -------
        self
            Returns self with processed data stored in attributes.
            
        Notes
        -----
        Input files are expected to be tab-separated with the following columns:
        - chrom, start position, end position, modified base code, score, strand,
        - start position 2, end position 2, color, Nvalid_cov, fraction modified,
        - Nmod, Ncanonical, Nother_mod, Ndelete, Nfail, Ndiff, Nnocall
        
        The method adds a 'position_token' column combining genomic coordinates,
        strand, and modification type for unique position identification.
        
        Attributes set depend on comparison parameter:
        - If comparison=True: Uses 'diff_transcript_mod_' prefix
        - If comparison=False: Uses 'basic_transcript_mod_' prefix
        """
        prefix = 'diff_transcript_mod_' if comparison else 'basic_transcript_mod_'
        
        if metadata == None:
            setattr(self, f'{prefix}data_files', data_files)
            setattr(self, f'{prefix}conditions', conditions)
            setattr(self, f'{prefix}samplenames', samplenames)
        else:
            metadata_df = pd.read_csv(metadata, sep="\t", header=0)
            setattr(self, f'{prefix}data_files', list(metadata_df["File"]))
            setattr(self, f'{prefix}conditions', [str(i) for i in metadata_df["Condition"]])
            setattr(self, f'{prefix}samplenames', [str(i) for i in metadata_df["Sample"]])
            
        data_files_attr = getattr(self, f'{prefix}data_files')
        conditions_attr = getattr(self, f'{prefix}conditions')
        samplenames_attr = getattr(self, f'{prefix}samplenames')

        # Setup comparisons only if comparison mode is enabled
        if comparison:
            setattr(self, f'{prefix}reference_levels', reference_levels)
            comparisons = []
            for ref_level in reference_levels:
                for cond_level in list(set(conditions_attr)):
                    if not ref_level == cond_level:
                        comparisons.append((cond_level, ref_level))
            setattr(self, f'{prefix}comparisons', comparisons)

        # Read all files with progress feedback
        all_dataframes = {}
        bed_desc = "Preprocessing transcript modification BED files"
        for fp in tqdm(data_files_attr, desc=bed_desc, unit="file"):
            logger.info(f"Preprocessing BED file: {fp}")
            df = pd.read_csv(fp, delimiter="\t", header=None)
            df.columns = [
                "chrom",
                "start position",
                "end position",
                "modified base code",
                "score",
                "strand",
                "start position 2",
                "end position 2",
                "color",
                "Nvalid_cov",
                "fraction modified",
                "Nmod",
                "Ncanonical",
                "Nother_mod",
                "Ndelete",
                "Nfail",
                "Ndiff",
                "Nnocall",
            ]
            all_dataframes[fp] = df
        dataframe_condition_dict = {}
        for idx, (df_Data, condition, sample) in enumerate(
            zip(all_dataframes, conditions_attr, samplenames_attr)
        ):
            dataframe_condition_dict[condition] = {}
        for idx, (df_Data, condition, sample) in enumerate(
            zip(all_dataframes, conditions_attr, samplenames_attr)
        ):  
            #Filter dataframe by coverage and modification frequency
            df_Data = all_dataframes[df_Data]
            df_Data_filtered = df_Data[
                (df_Data["Nvalid_cov"] >= coverage_filter)
                & (df_Data["fraction modified"] >= mod_freq_filter)
            ]
            df_Data_filtered = df_Data_filtered.copy()
            df_Data_filtered["position_token"] = [
                f"{chrom}:{start}:{end}:{strand}:{mod_type}"
                for chrom, start, end, strand, mod_type in zip(
                    df_Data_filtered["chrom"],
                    df_Data_filtered["start position"],
                    df_Data_filtered["end position"],
                    df_Data_filtered["strand"],
                    df_Data_filtered["modified base code"],
                )
            ]
            dataframe_condition_dict[condition][sample]=df_Data_filtered
        setattr(self, f'{prefix}dataframe_condition_dict', dataframe_condition_dict)

        # Chromosome processing
        possible_chromosomes = []
        for chr_df in list(all_dataframes.values()):
            possible_chromosomes = possible_chromosomes + list(chr_df["chrom"])
        setattr(self, f'{prefix}chr_names', list(reversed(sorted(list(set(possible_chromosomes))))))
        return self

    def diff_gene_exp_preprocess_data(self, 
                                metadata: str = None,
                                data_files: str = None,
                                conditions: list = None,
                                samplenames: list = None,
                                reference_levels: list = None,
                                output_folder:str="") -> dict:
        if metadata == None:
            self.diff_gene_exp_data_files = data_files
            self.diff_gene_exp_conditions = conditions
            self.diff_gene_exp_samplenames = samplenames
        else:
            self.diff_gene_exp_data_files = list(pd.read_csv(metadata, sep="\t", header=0)["File"])
            self.diff_gene_exp_conditions = list(
                pd.read_csv(metadata, sep="\t", header=0)["Condition"]
            )
            self.diff_gene_exp_conditions = [str(i) for i in self.diff_gene_exp_conditions]
            self.diff_gene_exp_samplenames = list(
                pd.read_csv(metadata, sep="\t", header=0)["Sample"]
            )
            self.diff_gene_exp_samplenames = [str(i) for i in self.diff_gene_exp_samplenames]

            
        self.diff_gene_exp_reference_levels = reference_levels
        self.diff_gene_exp_comparisons = []
        for ref_level in self.diff_gene_exp_reference_levels:
            for cond_level in list(set(self.diff_gene_exp_conditions)):
                if not ref_level == cond_level:
                    self.diff_gene_exp_comparisons.append((cond_level, ref_level))
                else:
                    continue
        # Read all files
        self.diff_gene_exp_count_df, self.diff_gene_exp_conditions_df = merge_single_sample_feature_count_tables(input_files = self.diff_gene_exp_data_files, 
                                                                                                                conditions = self.diff_gene_exp_conditions, 
                                                                                                                samplenames = self.diff_gene_exp_samplenames, 
                                                                                                                output_folder=output_folder) 
        return self.diff_gene_exp_count_df, self.diff_gene_exp_conditions_df
        
    def diff_transcript_exp_preprocess_data(self, 
                                metadata: str = None,
                                data_files: str = None,
                                conditions: list = None,
                                samplenames: list = None,
                                reference_levels: list = None,
                                output_folder:str="") -> dict:
        if metadata == None:
            self.diff_transcript_exp_data_files = data_files
            self.diff_transcript_exp_conditions = conditions
            self.diff_transcript_exp_samplenames = samplenames

        else:
            self.diff_transcript_exp_data_files = list(pd.read_csv(metadata, sep="\t", header=0)["File"])
            self.diff_transcript_exp_conditions = list(
                pd.read_csv(metadata, sep="\t", header=0)["Condition"]
            )
            self.diff_transcript_exp_conditions = [str(i) for i in self.diff_transcript_exp_conditions]
            self.diff_transcript_exp_samplenames = list(
                pd.read_csv(metadata, sep="\t", header=0)["Sample"]
            )
            self.diff_transcript_exp_samplenames = [str(i) for i in self.diff_transcript_exp_samplenames]
            

        self.diff_transcript_exp_reference_levels = reference_levels
        self.diff_transcript_exp_comparisons = []
        for ref_level in self.diff_transcript_exp_reference_levels:
            for cond_level in list(set(self.diff_transcript_exp_conditions)):
                if not ref_level == cond_level:
                    self.diff_transcript_exp_comparisons.append((cond_level, ref_level))
                else:
                    continue
        # Folder with your files
        # data_list = os.listdir(self.data_folder

        # Read all files
        self.diff_transcript_exp_count_df, self.diff_transcript_exp_conditions_df = merge_single_sample_salmon_count_tables(input_files = self.diff_transcript_exp_data_files,
                                                                                                                            conditions = self.diff_transcript_exp_conditions, 
                                                                                                                            samplenames = self.diff_transcript_exp_samplenames, 
                                                                                                                            output_folder=output_folder) 
        return self.diff_transcript_exp_count_df, self.diff_transcript_exp_conditions_df
    

