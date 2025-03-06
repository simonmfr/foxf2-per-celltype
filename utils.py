import warnings
import os
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc_context
import scanpy as sc

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

def clean_and_standardize_data(df, gene, study_order, celltype_order, cluster_key="clusters"):
    """
    Clean and standardize table for study-by-celltype dotplot.
    """
    # Validate gene
    assert np.all(np.char.lower(df.gene.unique().astype(str)) == gene.lower())
    
    # Remove unwanted cell types
    unwanted_clusters = [
        "Unknown", "Immune_Other", "Olfactory ensheathing cells", 
        "Fibromyocytes", "T cells", "NK/T cells", 
        "Choroid plexus epithelial cells", "Neuroepithelial cells", 
        "Hemoglobin-expressing vascular cells", "Olfactory ensheathing glia", 
        "Hypendymal cells", 
        "ECs_non_AV", "ECs_unclassified" # incl zonation
    ]
    df = df[~df[cluster_key].isin(unwanted_clusters)]
    
    # Standardize cluster names
    replacements = {
        "Oligos": "Oligodendrocytes",
        "Microglia": "Microglia/Mφ",
        "Macrophages": "Microglia/Mφ",
        "Microglia/Macrophages": "Microglia/Mφ",
        "SMCs/Pericytes": "SMCs",
        "Leptomeningeal cells": "Fibroblasts",
        "VLMCs": "Fibroblasts",
        "Neuroblasts": "Neuroblasts/NSCs",
        "Neuronal stem cells": "Neuroblasts/NSCs",
        "Neurogenesis": "Neuroblasts/NSCs",
        "Neural stem cells": "Neuroblasts/NSCs",
        "ECs": "Endothelial cells",
        "ECs_Arterial":"aECs",
        "ECs_Capillary":"capECs",
        "ECs_Venous":"vECs"
    }
    df = df.replace(replacements)
    
    # Create categorical columns with specified order
    df.source = pd.Categorical(df.source, categories=study_order)
    df[cluster_key] = pd.Categorical(df[cluster_key], categories=celltype_order)
    
    # Rename studies
    study_replacements = {
        "Saunders2018": "Saunders, 2018, Cell",
        "OwnData": "Own data", "Heindl2022": "Own data",
        "Zeisel2018": "Zeisel, 2018, Cell",
        "TabulaMuris2018": "Tabula Muris, 2018, Nature",
        "Winkler2022": "Winkler, 2022, Science",
        "Yang2022": "Yang, 2022, Nature",
        "Vanlandewijck2018": "Vanlandewijck, 2018, Nature",
        "Siletti2022": "Siletti, 2022, bioRxiv",
        "Garcia2022": "Garcia, 2022, Nature"
    }
    df = df.replace(study_replacements)

    df["fraction_of_cells"] = df["fraction_of_cells"] * 100
    
    return df.sort_values(['source', cluster_key])

def create_heatmap(df, gene, cluster_key="clusters", show=True):
    """
    Create and save study-by-celltype heatmap.
    """
    # Prepare data for heatmap
    heatmap_data = df.set_index([cluster_key,'source'])['mean_expression'].unstack().reset_index()
    heatmap_data.index = heatmap_data[cluster_key]
    heatmap_data = heatmap_data.reindex(list(df[cluster_key].unique()))
    heatmap_data = heatmap_data.drop(cluster_key, axis=1)

    # plot
    mycolormap = mpl.colors.LinearSegmentedColormap.from_list("", ['#ebebeb','tomato'])
    pl = sns.heatmap(heatmap_data, 
                     cmap=mycolormap, 
                     vmin=0, 
                     vmax=1, 
                     linewidths=0.01
                   ).set(ylabel=None,xlabel=None)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
    plt.xticks(rotation=45, ha='left')
    plt.legend(loc=(1.27,0.345), title=f'{gene} \nexpression \n(mean)', frameon=False)
    if show:
        plt.show()

def create_dotplot(df, gene, min_tile=0, figsize=(9,4), cluster_key="clusters", show=True, **kwargs):
    """
    Create study-by-celltype dotplot.
    """
    
    # format columns
    column_mapping = {
    cluster_key: cluster_key,
    'mean_expression': 'Mean expression\n       in group',
    'fraction_of_cells': 'Fraction of cells\n   in group (%)',
    'gene': 'gene',
    'source': 'source',
    'organism': 'organism',
    'cell_number': 'cell_number'
    }
    
    if any(col not in df.columns for col in column_mapping.items()):
        df.rename(columns=column_mapping, inplace=True)
    
    # plot
    mycolormap = mpl.colors.LinearSegmentedColormap.from_list("", ['#d1d1d1','tomato'])
    sns.set(style="white")
    pl = sns.relplot(data=df, 
                     x="source", 
                     y=cluster_key,
                     hue='Mean expression\n       in group', 
                     size='Fraction of cells\n   in group (%)',
                     palette=mycolormap, 
                     sizes=(min_tile, 550), 
                     linewidth=1,
                     **kwargs
                    )
    pl.fig.set_size_inches(*figsize)
    
    pl.set(ylabel=None, xlabel=None)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
    plt.xticks(rotation=45, ha='left')
    pl.set_xticklabels(fontsize=14.4, family="arial", color="black")
    pl.set_yticklabels(fontsize=14.1, family="arial", color="black")
    
    pl._legend.remove()
    legend = plt.legend(frameon=True, framealpha=0.2, borderpad=0.5, bbox_to_anchor=(1,1), 
                        title=gene, 
                        prop=mpl.font_manager.FontProperties(family='arial', size=10), 
                        labelcolor='black')
    plt.setp(legend.get_title(), color='black', family='arial', size=13)

    if show:
        plt.show()

def get_cell_numbers(df, cluster_key="clusters"):
    """
    Get study-by-celltype cell numbers
     """
    cell_numbers = df.set_index([cluster_key,'source'])['cell_number'].unstack().reset_index()
    cell_numbers.index = cell_numbers[cluster_key]
    cell_numbers = cell_numbers.reindex(list(df[cluster_key].unique()))
    cell_numbers = cell_numbers.drop(cluster_key, axis=1)
    cell_numbers = cell_numbers.fillna(0).astype(int)
    
    cell_numbers.loc["Total"] = cell_numbers.sum(skipna=True)
    cell_numbers['Total'] = cell_numbers[list(cell_numbers.columns)].sum(axis=1)
    
    return cell_numbers