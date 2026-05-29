#%%
from dmode import *

metadata="./test_samplesheet.tsv"
gtf_file_path = "../gencode.v48.basic.annotation.gtf/gencode.v48.basic.annotation.gtf"

# load DmodE self,metadata:str=None,data_files:str=None,conditions:list=None,reference_levels:list=None)
#DmodE_object = DmodE(data_files = data_paths, conditions=vector_age, reference_levels=["10"] )
DmodE_object = DmodE(metadata=metadata, reference_levels=["10"] )

# preprocess the data
DmodE_object.Preprocess_Data(20, 10)

# # analyze the data
output_dict = DmodE_object.Diff_mod_expression()


#%%
print(output_dict["10_VS_14"].columns)
print(output_dict["10_VS_14"][output_dict["10_VS_14"]["padj"] <= 0.05])


#%%
# Load GTF annotation (can be local file or downloaded GENCODE .gtf)
gtf_df = gtf_to_df(gtf_file_path)
significant_genes_df = DmodE_object.Extract_significant_genes(gtf_df=gtf_df, ref_level="10", cond_level="14", alpha=0.05, genes_only = True)
significant_positions_df = DmodE_object.Extract_significant_positions(ref_level="10", cond_level="14", alpha=0.05, chr="")
print(significant_genes_df)
print(significant_positions_df)

#%%
print(significant_genes_df)
print(significant_positions_df)

#%% generate the volcano DmodE volcano plot for each chromosome

headers = list(output_dict.keys())
chr_names = DmodE_object.chr_names

Generate_volcano_plots(DmodE_object.chr_names, output_dict[headers[0]], alpha=0.05)
Generate_volcano_plot_single_chr("", output_dict[headers[0]],alpha=0.05)
