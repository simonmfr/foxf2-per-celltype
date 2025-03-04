import os
import warnings
import pandas as pd
import scanpy as sc
from datetime import date

def summarize_gene_expression(adata, gene, groupby, study_name, organism, layer="normalized", 
                              output_dir="output", export=False):
    """
    Summarizes gene expression per cluster, including mean expression and fraction of expressing cells.
    
    Parameters:
    - adata: AnnData object
    - gene: str, gene of interest
    - study_name: str, source of the data
    - organism: str, organism name
    - layer: str, adata.X layer (default: normalized)
    - output_dir: str, directory for output files
    - export: bool, whether to save the output as a CSV file (default: False)
    
    Returns:
    - DataFrame summarizing mean expression, fraction of cells, and metadata
    """
    
    # Check if the gene exists in adata.var_names
    if gene not in adata.var_names:
        warnings.warn(f"Gene '{gene}' not found in adata.var_names. Skipping this gene.")
        return None
    
    # Compute mean expression per group
    mean_expr = sc.pl.matrixplot(adata, [gene], groupby=groupby, standard_scale="var",
                                 layer=layer, return_fig=True).values_df
    mean_expr.columns = ['mean_expression']

    # dotplot dot size = fraction of cells in group expressing Foxf2
    frac_cells = sc.pl.dotplot(adata, [gene], groupby=groupby, standard_scale="var",
                               layer=layer, return_fig=True).dot_size_df
    frac_cells.columns = ['fraction_of_cells']

    # Combine results
    summary = pd.concat([mean_expr, frac_cells], axis=1)
    summary['gene'] = gene
    summary['source'] = study_name
    summary['organism'] = organism
    summary['cell_number'] = adata.obs[groupby].value_counts().reindex(summary.index)

    if export:
        filename = os.path.join(f"{output_dir}",f"{date.today().strftime('%Y%m%d')}_{gene}_{study_name}_layer{layer}.csv")
        summary.to_csv(filename, sep=';')
        print(f"Exported to {filename}")

    return summary


def cluster_small_multiples(adata, clust_key, size=1.5, frameon=False, legend_loc=None, **kwargs):
    """
    Generates a UMAP facet plot for each cluster at a time.
    """
    tmp = adata.copy()
    for i, clust in enumerate(tmp.obs[clust_key].cat.categories):
        tmp.obs[clust] = tmp.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", tmp.uns[clust_key + "_colors"][i]]
    sc.pl.umap(
        tmp, groups=tmp.obs[clust].cat.categories[1:].values,
        color=tmp.obs[clust_key].cat.categories.tolist(),
        size=size,
        frameon=frameon, legend_loc=legend_loc, **kwargs,
    )