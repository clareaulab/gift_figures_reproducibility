import itertools
import os
import shutil

import buencolors as bc
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
import spatialdata as sd
from shapely import Polygon
import scanpy as sc
from skimage.transform import resize
from spatialdata import read_zarr, polygon_query, transform, rasterize, sanitize_table, SpatialData
import giftwrap as gw
import spatialdata.io as sio
import hdf5plugin
from spatialdata.transformations import get_transformation, set_transformation, Identity


def numpy_to_shapely(poly):
    return Polygon([(x, y) for x, y in poly])


# Common data reading utils:

# # Gapfill data was saved to h5ad files to allow for compression which can be uploaded to Github using the following code:
# dp_adata = gw.read_h5_file("../data/DualPanel_CellLineMix/counts.1.h5")
# gf_adata = gw.read_h5_file("../data/GapfillPanel_CellLineMix/counts.1.h5")
#
# # Reduce the size before writing
# layers_to_remove = [l for l in dp_adata.layers.keys() if l.startswith("X_pcr_threshold") and int(l.split('_')[-1]) > 5]
# for l in layers_to_remove:
#     del dp_adata.layers[l]
# dp_adata.write_h5ad("../data/DualProbe_CellLineMix_Visium_GF.h5ad")
#
# layers_to_remove = [l for l in gf_adata.layers.keys() if l.startswith("X_pcr_threshold") and int(l.split('_')[-1]) > 5]
# for l in layers_to_remove:
#     del gf_adata.layers[l]
# gf_adata.write_h5ad("../data/GapFill_CellLineMix_Visium_GF.h5ad")

# WTAs generated with:
# dual_probe_wta = sio.visium_hd(
#     "../data/DualPanel_WTA/outs",
#      dataset_id='',
#      fullres_image_file='../data/DualPanel_WTA/outs/spatial/tissue_hires_image.png'
# )
# dual_probe_wta.write("../data/DualPanel_WTA.zarr")
# shutil.make_archive("../data/DualPanel_WTA.zarr", "xztar", ".", "../data/DualPanel_WTA.zarr")
#
# gapfill_wta = sio.visium_hd(
#     "../data/GapFill_WTA/outs",
#      dataset_id='',
#      fullres_image_file='../data/GapFill_WTA/outs/spatial/tissue_hires_image.png'
# )
# gapfill_wta.write("../data/GapFill_WTA.zarr")
# shutil.make_archive("../data/GapFill_WTA.zarr", "xztar", ".", "../data/GapFill_WTA.zarr")


def read_dual_probe_data() -> sd.SpatialData:
    dp_adata = sc.read_h5ad("../data/DualProbe_CellLineMix_Visium_GF.h5ad")
    if not os.path.exists("../data/DualPanel_WTA.zarr"):
        shutil.unpack_archive("../data/DualPanel_WTA.zarr.zip", "../data/DualPanel_WTA.zarr", "xztar")
    dual_probe_wta = read_zarr("../data/DualPanel_WTA.zarr")
    dp_adata = gw.pp.filter_by_min_pcr_duplicates(dp_adata, 5)

    # Harmonize the dual probe data to resemble gapfill data for simplicity
    # We don't expect any gapfills, so we should filter out UMIs with a gapfill
    probes_with_gapfill = dp_adata.var[dp_adata.var['gapfill'] != ''].index
    # Count number of probes and the number of umis this affects
    print(f"Number of probes with gapfill: {len(probes_with_gapfill)} / {dp_adata.var.shape[0]}")
    umis_with_gapfill = np.sum(dp_adata[:, probes_with_gapfill].X)
    print(f"Number of UMIs with gapfill: {umis_with_gapfill} / {dp_adata.X.sum()}")
    # Drop them
    dp_adata = dp_adata[:, dp_adata.var['gapfill'] == '']
    # Select all the 0bp control probes, these will not be used for genotyping
    zerobp_probes = dp_adata.var[dp_adata.var.probe.str.contains("0bp")].probe.values
    mut_wt_pairs = []
    # Now for each non 0bp probe, we will find the mutant probe and its associated WT probe
    for probe in dp_adata.var.probe.values:
        if "0bp" in probe:
            continue
        if ">" in probe:
            orig = probe.split(">")[0]
            alt = probe.split(">")[1]
            if orig[-1] == alt:  # This was a WT probe
                continue
            wt_probe = f"{orig}>{orig[-1]}"
            mut_wt_pairs.append((probe, wt_probe, alt, orig[-1]))
        else:
            print(f"Unexpected probe name: {probe}")

    # Manually add the probes that were not named correctly
    mut_wt_pairs.append(("BCR-ABL c.fusion", "BCR-ABL null", 'fusion', 'null'))
    mut_wt_pairs.append(("TP53 c.405insC", "TP53 c.405", "insC", "ref"))

    print(
        f"Dropping the following probes that could not be paired: {set(dp_adata.var.probe) - set(zerobp_probes) - set([a for a, b, c, d in mut_wt_pairs]) - set([b for a, b, c, d in mut_wt_pairs])}")
    # Create a new AnnData with just the probes we want
    new_var = dict(
        probe=[],
        gene=[],
        gapfill=[],
    )
    new_X = np.zeros((dp_adata.n_obs, 2 * len(mut_wt_pairs) + zerobp_probes.shape[0]), dtype=np.int32)
    new_layers = {k: np.zeros((dp_adata.n_obs, 2 * len(mut_wt_pairs) + zerobp_probes.shape[0]), dtype=np.int32) for k in
                  dp_adata.layers.keys() if 'X' in k and int(k.split("_")[-1]) < 25}
    for i, probe in enumerate(zerobp_probes):
        idx = dp_adata.var.index[dp_adata.var.probe == probe][0]
        new_var['probe'].append(probe)
        new_var['gene'].append(dp_adata.var.loc[idx, 'gene'])
        new_var['gapfill'].append(dp_adata.var.loc[idx, 'gapfill'])
        new_X[:, i] = dp_adata[:, idx].X.toarray().flatten()
        for k in list(new_layers.keys()):
            new_layers[k][:, i] = dp_adata[:, idx].layers[k].toarray().flatten()
    for j, (mut, wt, mut_genotype, wt_genotype) in enumerate(mut_wt_pairs):
        wt_idx = dp_adata.var.index[dp_adata.var.probe == wt][0]
        mut_idx = dp_adata.var.index[dp_adata.var.probe == mut][0]
        new_var['probe'].append(mut)
        new_var['gene'].append(dp_adata.var.loc[wt_idx, 'gene'])
        new_var['gapfill'].append(wt_genotype)
        new_X[:, len(zerobp_probes) + 2 * j] = dp_adata[:, wt_idx].X.toarray().flatten()
        new_var['probe'].append(mut)
        new_var['gene'].append(dp_adata.var.loc[wt_idx, 'gene'])
        new_var['gapfill'].append(mut_genotype)
        new_X[:, len(zerobp_probes) + 2 * j + 1] = dp_adata[:, mut_idx].X.toarray().flatten()
        for k in list(new_layers.keys()):
            new_layers[k][:, len(zerobp_probes) + 2 * j] = dp_adata[:, wt_idx].layers[k].toarray().flatten()
            new_layers[k][:, len(zerobp_probes) + 2 * j + 1] = dp_adata[:, mut_idx].layers[k].toarray().flatten()

    var_df = pd.DataFrame(new_var)
    var_df['probe_gapfill'] = var_df['probe'] + "_" + var_df['gapfill']
    var_df = var_df.set_index('probe_gapfill')
    dual_probe_gdata = ad.AnnData(
        X=new_X,
        obs=dp_adata.obs.copy(),
        var=var_df,
        uns=dp_adata.uns.copy(),
        layers=new_layers
    )
    ### Dual probe should now resemble gapfill data structure

    ## Annotate cell lines by known regions
    dual_probe_polygons = {
        "HEL": numpy_to_shapely(np.array([[9_000, 43_000], [40_000, 45_000], [40_000, 59_000], [9_000, 57_000]])),
        "K562": numpy_to_shapely(np.array([[9_000, 57_000], [22_000, 57_000], [22_000, 75_000], [9_000, 75_000]])),
        "SET2": numpy_to_shapely(np.array([[23_000, 58_000], [40_000, 58_000], [40_000, 75_000], [23_000, 75_000]])),
    }

    dual_probe_wta.tables['square_002um'].obs['cell_line'] = 'N/A'
    dual_probe_wta.tables['square_008um'].obs['cell_line'] = 'N/A'
    dual_probe_wta.tables['square_016um'].obs['cell_line'] = 'N/A'
    for cell_line, poly in dual_probe_polygons.items():
        filtered = polygon_query(
            dual_probe_wta,
            polygon=poly,
            target_coordinate_system="",
        )
        dual_probe_wta.tables['square_002um'].obs.loc[
            filtered.tables['square_002um'].obs_names, 'cell_line'] = cell_line
        dual_probe_wta.tables['square_008um'].obs.loc[
            filtered.tables['square_008um'].obs_names, 'cell_line'] = cell_line
        dual_probe_wta.tables['square_016um'].obs.loc[
            filtered.tables['square_016um'].obs_names, 'cell_line'] = cell_line

    # Crop to tissue area
    dual_probe_wta = dual_probe_wta.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=np.array([0, 30_000]),
        max_coordinate=np.array([40_000, 80_000]),
        target_coordinate_system="",
        filter_table=False
    )

    gw.pp.filter_gapfills(dp_adata, min_cells=10)
    # Genotype and intersect the WTA and GF libraries
    dual_probe_gdata = gw.tl.call_genotypes(
        dual_probe_gdata
    )
    dual_probe_sdata = gw.sp.join_with_wta(dual_probe_wta, dual_probe_gdata)

    return dual_probe_sdata


def read_gapfill_data(gf_adata_path = "../data/GapFill_CellLineMix_Visium_GF.h5ad", WTA_dir = "../data/GapFill_WTA.zarr", cores = 1) -> sd.SpatialData:
    gf_adata = sc.read_h5ad(gf_adata_path)
    # if not os.path.exists(WTA_dir):
    #     shutil.unpack_archive(WTA_dir + ".zip", WTA_dir, "zip")
    gapfill_wta = read_zarr(WTA_dir)

    gf_adata = gw.pp.filter_by_min_pcr_duplicates(gf_adata, 5)

    gapfill_polygons = {
        "HEL": numpy_to_shapely(np.array([[10_000, 12_000], [40_000, 12_000], [40_000, 31_000], [10_000, 31_000]])),
        "K562": numpy_to_shapely(np.array([
            [10_000, 27_000], [19_000, 27_000], [19_000, 37_000],
            [19_500, 38_000], [19_000, 40_000], [10_000, 50_000],
            [10_000, 50_000]
        ])),
        "SET2": numpy_to_shapely(np.array([
            [40_000, 31_000], [19_500, 31_000], [19_000, 40_000],
            [18_000, 43_000], [40_000, 44_500]
        ])),
    }

    gapfill_wta.tables['square_002um'].obs['cell_line'] = 'N/A'
    gapfill_wta.tables['square_008um'].obs['cell_line'] = 'N/A'
    gapfill_wta.tables['square_016um'].obs['cell_line'] = 'N/A'
    for cell_line, poly in gapfill_polygons.items():
        filtered = polygon_query(
            gapfill_wta,
            polygon=poly,
            target_coordinate_system="",
        )
        gapfill_wta.tables['square_002um'].obs.loc[filtered.tables['square_002um'].obs_names, 'cell_line'] = cell_line
        gapfill_wta.tables['square_008um'].obs.loc[filtered.tables['square_008um'].obs_names, 'cell_line'] = cell_line
        gapfill_wta.tables['square_016um'].obs.loc[filtered.tables['square_016um'].obs_names, 'cell_line'] = cell_line

    gapfill_wta = gapfill_wta.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=np.array([0, 0]),
        max_coordinate=np.array([40_000, 45_000]),
        target_coordinate_system="",
        filter_table=False
    )

    gw.pp.filter_gapfills(gf_adata, min_cells=10)
    gf_adata = gw.tl.call_genotypes(
        gf_adata, cores = cores
    )
    gapfill_sdata = gw.sp.join_with_wta(gapfill_wta, gf_adata)

    return gapfill_sdata



def read_genotype_annotations() -> tuple[list, dict, dict, dict]:
    # Read the genotype annotations
    celltype_genotypes_df = pd.read_csv("../data/3cl_predicted_genotypes.csv", index_col=0)

    # Map genotypes
    annotated_genotypes = celltype_genotypes_df.name.unique().tolist()
    wt_alleles = dict()
    alt_alleles = dict()
    celltype_genotypes = {
        "HEL": dict(),
        "K562": dict(),
        "SET2": dict(),
    }
    celltype_annotated = {
        "HEL": dict(),
        "K562": dict(),
        "SET2": dict()
    }
    for i, row in celltype_genotypes_df.iterrows():
        wt_alleles[row['name']] = row["gapfill_from_transcriptome"]
        alt_alleles[row['name']] = row["gap_probe_sequence"]
        if row['genotype_from_bulk'] == 'heterozygous':
            celltype_annotated[row['cell_type']][row["name"]] = "HET"
        elif row['genotype_from_bulk'] == 'homozygous_ref':
            celltype_annotated[row['cell_type']][row["name"]] = "REF"
        elif row['genotype_from_bulk'] == 'homozygous_alt':
            celltype_annotated[row['cell_type']][row["name"]] = "ALT"
        first_gf = row["0"]
        second_gf = row["1"]
        celltype_genotypes[row['cell_type']][row["name"]] = [first_gf]
        if not pd.isna(second_gf) and second_gf != "":
            celltype_genotypes[row['cell_type']][row["name"]].append(second_gf)

    return annotated_genotypes, celltype_genotypes, wt_alleles, alt_alleles


# Functions for plotting:
def get_0bp_probe(adata, probe_name: str):
    curr_gene = adata.var[adata.var.probe == probe_name].gene.values[0]
    zero_bp_probe = adata.var[(adata.var.gene.str == curr_gene) & (adata.var.probe.str.contains("0bp") | (adata.var.probe.str == adata.var.gene.str))].probe.values
    if len(zero_bp_probe) < 1 or zero_bp_probe[0] == probe_name:
        return None
    return zero_bp_probe[0]

def get_all_0bp_probes(adata):
    zero_bp_probes = []
    for probe in adata.var.probe.unique():
        zero_bp_probe = get_0bp_probe(adata, probe)
        if zero_bp_probe is not None and zero_bp_probe not in zero_bp_probes:
            zero_bp_probes.append(zero_bp_probe)
    return zero_bp_probes

def plot_celltype_specific_probes_spatial_multi_cellline(
    adata,
    annotated_genotypes,
    celltype_genotypes,
    wt_alleles,
    alt_alleles,
    resolution: int = 2,
    include_het: bool = False,
    color_by_celline: bool = False,
    log_scale_marginals: bool = True,
    smooth_lines: bool = False,
    figsize: tuple = (19, 19),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
):
    """
    Plot spatial distribution of UMI counts for cell type-specific probes with line plot marginals for all cell lines.

    This function loops through all cell lines, identifies cell-line-specific probes for each
    (using the same logic as plot_celltype_specific_probes_spatial), and displays line plot
    marginals showing the spatial distribution of each cell line's specific probes with different colors.

    Parameters:
    -----------
    adata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    annotated_genotypes : list or set
        List/set of probe names that have genotype annotations
    celltype_genotypes : dict
        Dict mapping cell line names to probe genotypes.
        Format: {cell_line: {probe: [alleles]}}
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    include_het : bool
        If True, allow probes where other (non-target) cell lines have HET genotypes
        to be considered cell-type specific. When counting UMIs for such probes,
        spatial bins belonging to HET cell lines are excluded from the visualization
        The target cell line is never allowed to be HET
        (default: False)
    color_by_celline : bool
        If True, color the spatial plot by cell_line annotation instead of UMI counts
        (default: False)
    log_scale_marginals : bool
        If True, apply log scaling to the marginal line plots
        (default: True)
    smooth_lines : bool
        If True, apply Gaussian smoothing to the marginal line plots to reduce noise
        (default: False)
    figsize : tuple
        Figure size for the plot (default: (19, 15))
    fig : matplotlib Figure
        Optional existing figure to plot on (default: None)
    ax : matplotlib Axes
        Optional existing axis to plot on. If provided, marginals will be created as inset axes (default: None)

    Returns:
    --------
    fig, axes, summary_data : matplotlib figure, axes objects (main, top, right), and summary dictionary
    """
    # Get the gapfill table
    if isinstance(adata, ad.AnnData):
        table = adata
    else:
        table = adata.tables[f'gf_square_{resolution:03d}um']

    # Add cell line annotations if not present
    if 'cell_line' not in table.obs.columns:
        if not isinstance(adata, ad.AnnData):
            wta = adata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta.obs.columns:
                table = table.copy()
                table.obs['cell_line'] = wta.obs.loc[table.obs_names, 'cell_line'].values
            else:
                raise ValueError("cell_line annotation not found in data.")
        else:
            raise ValueError("cell_line annotation not found in data.")

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    table.obs['x_coord'] = [c[1] for c in coords]
    table.obs['y_coord'] = [c[0] for c in coords]

    # Get non-0bp probes
    zero_bp_probes = get_all_0bp_probes(table)
    non_zero_probes = [p for p in table.var.probe.unique() if p not in zero_bp_probes]

    # Get all unique cell lines and assign colors
    unique_cell_lines = sorted(table.obs['cell_line'].unique())
    n_cell_lines = len(unique_cell_lines)

    if len(set(unique_cell_lines) - set(['HEL', 'K562', 'SET2'])) == 0:  # Only has our typical benchmark cell lines
        cell_line_colors = {
            'K562':'#F79520',
            'SET2':'#1C75BC',
            'HEL':'#39B54A'
        }
    else:
        # Use tab10 colormap for up to 10 cell lines, otherwise use a larger palette
        if n_cell_lines <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_cell_lines]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_cell_lines]

        cell_line_colors = {cl: colors[i] for i, cl in enumerate(unique_cell_lines)}

    # Initialize UMI count tracking per cell line
    umi_counts_by_cellline = {cl: np.zeros(table.shape[0]) for cl in unique_cell_lines}
    umi_counts_total = np.zeros(table.shape[0])

    # Track probe info across all cell lines
    all_probes_info = []

    # Loop through each cell line to find its specific probes
    for target_cell_line in unique_cell_lines:
        if target_cell_line not in celltype_genotypes:
            continue

        # Prepare data for identifying cell-type specific probes for this cell line
        probe_info = {
            'cell_line': [],
            'probe': [],
            'probe_norm': [],
            'target_genotype': [],
            'is_specific': [],
            'het_cell_lines': [],
            'same_genotype_cell_lines': []
        }

        for probe in non_zero_probes:
            probe_norm = probe.split("|")
            if len(probe_norm) > 1:
                probe_norm = " ".join(probe_norm[1:3])
            else:
                probe_norm = probe

            if probe_norm not in annotated_genotypes:
                continue
            if probe_norm not in celltype_genotypes[target_cell_line]:
                continue

            target_alleles = set(celltype_genotypes[target_cell_line][probe_norm])

            if len(target_alleles) > 1:
                target_genotype = "HET"
            elif wt_alleles[probe_norm] in target_alleles:
                target_genotype = "WT"
            elif alt_alleles[probe_norm] in target_alleles:
                target_genotype = "ALT"
            else:
                target_genotype = "Unknown"

            # ALWAYS skip HET target genotypes
            if target_genotype == 'HET':
                continue

            # Skip Unknown genotypes
            if target_genotype == 'Unknown':
                continue

            has_different_genotype = False
            het_cell_lines = []
            same_genotype_cell_lines = []

            for other_cell_line, genotypes in celltype_genotypes.items():
                if other_cell_line == target_cell_line:
                    continue
                if probe_norm in genotypes:
                    other_alleles = set(genotypes[probe_norm])

                    if len(other_alleles) > 1:
                        other_genotype = "HET"
                    elif wt_alleles[probe_norm] in other_alleles:
                        other_genotype = "WT"
                    elif alt_alleles[probe_norm] in other_alleles:
                        other_genotype = "ALT"
                    else:
                        other_genotype = "Unknown"

                    if other_genotype == 'HET':
                        het_cell_lines.append(other_cell_line)
                        if not include_het:
                            break
                        elif target_genotype in ('WT', 'ALT'):
                            has_different_genotype = True

                    elif target_genotype in ('WT', 'ALT') and other_genotype == target_genotype:
                        same_genotype_cell_lines.append(other_cell_line)

                    elif target_genotype in ('WT', 'ALT') and other_genotype in ('WT', 'ALT') and target_genotype != other_genotype:
                        has_different_genotype = True

            is_specific = has_different_genotype

            if not include_het and len(het_cell_lines) > 0:
                is_specific = False

            probe_info['cell_line'].append(target_cell_line)
            probe_info['probe'].append(probe)
            probe_info['probe_norm'].append(probe_norm)
            probe_info['target_genotype'].append(target_genotype)
            probe_info['is_specific'].append(is_specific)
            probe_info['het_cell_lines'].append(het_cell_lines)
            probe_info['same_genotype_cell_lines'].append(same_genotype_cell_lines)

        # Filter to only cell-type specific probes for this cell line
        df_probes = pd.DataFrame(probe_info)
        df_probes = df_probes[df_probes['is_specific']]

        # Calculate UMI counts for this cell line's specific probes
        for _, row in df_probes.iterrows():
            probe = row['probe']
            probe_norm = row['probe_norm']
            target_genotype = row['target_genotype']
            het_cell_lines = row['het_cell_lines']
            same_genotype_cell_lines = row['same_genotype_cell_lines']

            # Get probe-specific data
            probe_mask = table.var.probe == probe
            probe_table = table[:, probe_mask]

            # Detect if dual probe or gapfill probe
            available_gapfills = probe_table.var.gapfill.unique().tolist()
            is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)

            # Get WT and ALT alleles
            if is_dual_probe:
                if ">" in probe_norm:
                    variant_part = probe_norm.split()[-1]
                    if ">" in variant_part:
                        bases = variant_part.split(">")
                        wt_allele = bases[0][-1]
                        alt_allele = bases[1]
                    else:
                        continue
                else:
                    continue
            else:
                wt_allele = wt_alleles[probe_norm]
                alt_allele = alt_alleles[probe_norm]

            if target_genotype == 'WT':
                valid_alleles = [wt_allele]
            elif target_genotype == 'ALT':
                valid_alleles = [alt_allele]
            elif target_genotype == 'HET':
                valid_alleles = [wt_allele, alt_allele]
            else:
                continue

            # Filter to specific gapfill alleles
            gapfill_mask = table.var.gapfill.isin(valid_alleles)
            combined_mask = probe_mask & gapfill_mask

            if combined_mask.any():
                probe_counts = table[:, combined_mask].X.sum(axis=1)
                if hasattr(probe_counts, 'A1'):
                    probe_counts = probe_counts.A1
                probe_counts = probe_counts.flatten()

                # Build list of cell lines to exclude from UMI counts
                exclude_cell_lines = []

                # When include_het=True, exclude UMI counts from HET cell lines for this probe
                if include_het and len(het_cell_lines) > 0:
                    exclude_cell_lines.extend(het_cell_lines)

                # Always exclude cell lines with the same genotype as target
                if len(same_genotype_cell_lines) > 0:
                    exclude_cell_lines.extend(same_genotype_cell_lines)

                # Apply exclusion mask (keep all non-excluded cell lines, just like the original function)
                if len(exclude_cell_lines) > 0:
                    # Create a mask for bins NOT belonging to excluded cell lines
                    non_excluded_mask = ~table.obs['cell_line'].isin(exclude_cell_lines)
                    # Zero out counts from excluded cell lines
                    probe_counts_filtered = probe_counts * non_excluded_mask.values
                else:
                    probe_counts_filtered = probe_counts.copy()

                # Add to this cell line's total (this tracks which probes were specific to this cell line)
                umi_counts_by_cellline[target_cell_line] += probe_counts_filtered
                umi_counts_total += probe_counts_filtered

        # Store probe info for this cell line
        all_probes_info.append(df_probes)

    # Combine all probe info across cell lines
    if len(all_probes_info) > 0:
        summary_df = pd.concat(all_probes_info, ignore_index=True)
    else:
        summary_df = pd.DataFrame()

    # Get all spatial coordinates
    x_coords = table.obs['x_coord'].values
    y_coords = table.obs['y_coord'].values

    # Remove NaN coordinates
    valid_mask = ~(np.isnan(x_coords) & np.isnan(y_coords))
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    umi_counts_valid = umi_counts_total[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create 2D matrix for heatmap
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

    if color_by_celline:
        # Get cell_line values for each coordinate
        cell_line_values = table.obs['cell_line'].values[valid_mask]

        # Create a categorical mapping for cell lines
        cell_line_to_num = {cl: i for i, cl in enumerate(unique_cell_lines)}

        # Fill in the matrix with cell_line indices
        for x, y, cl in zip(x_coords, y_coords, cell_line_values):
            heatmap_matrix[y - y_min, x - x_min] = cell_line_to_num.get(cl, np.nan)
    else:
        # Fill in the matrix with UMI counts
        for x, y, count in zip(x_coords, y_coords, umi_counts_valid):
            heatmap_matrix[y - y_min, x - x_min] = count

    # Compute marginal line plots per cell line
    x_marginals_by_cellline = {cl: np.zeros(x_max - x_min + 1) for cl in unique_cell_lines}
    y_marginals_by_cellline = {cl: np.zeros(y_max - y_min + 1) for cl in unique_cell_lines}

    for cl in unique_cell_lines:
        cl_counts_valid = umi_counts_by_cellline[cl][valid_mask]
        for x, y, count in zip(x_coords, y_coords, cl_counts_valid):
            x_marginals_by_cellline[cl][x - x_min] += count
            y_marginals_by_cellline[cl][y - y_min] += count

    # Apply Gaussian smoothing if requested
    if smooth_lines:
        for cl in unique_cell_lines:
            x_marginals_by_cellline[cl] = gaussian_filter1d(x_marginals_by_cellline[cl], sigma=.01)
            y_marginals_by_cellline[cl] = gaussian_filter1d(y_marginals_by_cellline[cl], sigma=.01)

    # Create figure and axes
    if ax is not None:
        # Use provided axis and create inset axes for marginals
        fig = ax.figure
        ax_main = ax

        # Create inset axes for marginals
        # Position: [x0, y0, width, height] in axes coordinates
        ax_top = ax_main.inset_axes([0, 1.05, 1, 0.25])  # Above main plot
        ax_right = ax_main.inset_axes([1.05, 0, 0.25, 1])  # Right of main plot

        if not color_by_celline:
            ax_cbar = ax_main.inset_axes([1.35, 0, 0.05, 1])  # Colorbar position

        ax_top.sharex(ax_main)
        ax_right.sharey(ax_main)
    else:
        # Original GridSpec approach when no axis is provided
        fig = plt.figure(figsize=figsize) if fig is None else fig
        if color_by_celline:
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.05,
                          width_ratios=[1, 1, 1, 1])
        else:
            gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.05,
                          width_ratios=[1, 1, 1, 1, 0.15])

        # Create axes
        if color_by_celline:
            ax_main = fig.add_subplot(gs[1:, :-1])
            ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        else:
            ax_main = fig.add_subplot(gs[1:, :-2])
            ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)
            ax_cbar = fig.add_subplot(gs[1:, -1])

    # Plot main heatmap
    if color_by_celline:
        # Create discrete colormap using the same colors as the line plots
        cmap = ListedColormap([cell_line_colors[cl] for cl in unique_cell_lines])
        cmap_copy = cmap.copy()
        cmap_copy.set_bad(color='white')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='equal',
            origin='upper',
            cmap=cmap_copy,
            interpolation='nearest',
            extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
            vmin=0,
            vmax=n_cell_lines - 1
        )

        legend_elements = [Patch(facecolor=cell_line_colors[cl], label=cl)
                          for cl in unique_cell_lines]
        ax_main.legend(handles=legend_elements, loc='upper right',
                      title='Cell Line', framealpha=0.9)
    else:
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')

        im = ax_main.imshow(
            heatmap_matrix,
            aspect='equal',
            origin='upper',
            cmap=cmap,
            interpolation='nearest',
            extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution]
        )

        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('UMI Count', rotation=270, labelpad=20)

    ax_main.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (μm)', fontsize=12)
    ax_main.set_title(f'Cell Line-Specific Probes', fontsize=14, fontweight='bold')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    # ax_main.set_aspect('equal')
    # Disable ticks
    ax_main.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False
    )

    max_total_value = max(max(itertools.chain(*x_marginals_by_cellline.values())), max(itertools.chain(*y_marginals_by_cellline.values())))

    # Plot top marginal (x-axis aggregate)
    # Use x_min + 0.5 to align with pixel centers in imshow
    x_positions = (np.arange(x_min, x_max + 1) + 0.5) * resolution
    for cl in unique_cell_lines:
        ax_top.plot(x_positions, x_marginals_by_cellline[cl],
                    color=cell_line_colors[cl], label=cl, linewidth=2, alpha=0.8)

    ax_top.set_xlim(x_min * resolution, (x_max + 1) * resolution)  # Force alignment with imshow extent
    ax_top.set_ylabel('Total UMIs', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.set_ylim(bottom=0, top=max_total_value)

    # Plot right marginal (y-axis aggregate)
    y_positions = (np.arange(y_min, y_max + 1) + 0.5) * resolution
    for cl in unique_cell_lines:
        ax_right.plot(y_marginals_by_cellline[cl], y_positions,
                      color=cell_line_colors[cl], label=cl, linewidth=2, alpha=0.8)

    # CRITICAL: Invert y-axis to match origin='upper' of imshow
    ax_right.set_ylim((y_max + 1) * resolution, y_min * resolution)
    ax_right.set_xlabel('Total UMIs', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.set_xlim(left=0, right=max_total_value)

    # Apply log scaling to the marginals if requested
    if log_scale_marginals:
        ax_top.set_yscale('log')
        ax_right.set_xscale('log')

    # Print summary statistics: % of cell-line-specific UMIs matching genotype per cell line
    print("\n=== UMI Genotype Specificity Summary ===")
    print(f"{'Cell Line':<12} {'CL-Specific UMIs':>18} {'Total Specific UMIs':>20} {'% Matching Genotype':>20}")
    print("-" * 74)
    grand_total = umi_counts_total.sum()
    for cl in unique_cell_lines:
        # UMIs from this cell line's specific probes, only in bins annotated as this cell line
        cl_bin_mask = table.obs['cell_line'] == cl
        cl_umi_in_cl_bins = umi_counts_by_cellline[cl][cl_bin_mask].sum()
        total_specific_in_cl_bins = umi_counts_total[cl_bin_mask].sum()
        pct = (cl_umi_in_cl_bins / total_specific_in_cl_bins * 100) if total_specific_in_cl_bins > 0 else float('nan')
        print(f"{cl:<12} {cl_umi_in_cl_bins:>18.0f} {total_specific_in_cl_bins:>20.0f} {pct:>19.1f}%")
    print("-" * 74)
    print(f"{'Grand total':<12} {grand_total:>18.0f}")
    print()

    # Return figure, axes, and summary data
    axes = {
        'main': ax_main,
        'top': ax_top,
        'right': ax_right
    }

    return fig, axes, summary_df


def plot_wt_alt_alleles_spatial(
    sdata,
    probe_name: str,
    wt_alleles: dict,
    alt_alleles: dict,
    resolution: int = 2,
    include_he: bool = False,
    figsize: tuple = (15, 15),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    cell_line_colors: dict = None,
    title: str = None
):
    """
    Plot spatial distribution of WT and ALT allele genotypes for a specific probe.

    This function colors each spatial bin by its genotype for the given probe:
    - Blue for WT (wild-type)
    - Red for ALT (alternate)
    - Purple/magenta for HET (heterozygous, interpolated between WT and ALT)

    Parameters:
    -----------
    sdata : SpatialData or AnnData
        Spatial data object containing gapfill and spatial coordinate data
    probe_name : str
        Name of the probe to visualize. Can be either the full probe name from the data
        (e.g., "SET2_homozygous|SPN|c.879C>T") or a substring that matches
        (e.g., "SPN c.879C>T" or just "SPN")
    wt_alleles : dict
        Dict mapping probe names to WT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    alt_alleles : dict
        Dict mapping probe names to ALT alleles.
        Format: {probe: 'A'/'C'/'G'/'T'}
    resolution : int
        Resolution in microns (default: 2)
    include_he: bool
        Whether to use an empty background or use the H&E image as the background (default: False)
    figsize : tuple
        Figure size for the plot (default: (15, 15))
    fig : plt.Figure, optional
        Matplotlib figure object to plot on. If None, a new figure is created.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, a new axes is created.
    cell_line_colors : dict, optional
        Dict mapping cell line names to colors for drawing tissue outlines.
        Format: {cell_line: color} where color can be any matplotlib color specification.
        If None, no outlines are drawn.

    Returns:
    --------
    fig, ax, genotype_data
        - fig: matplotlib figure object
        - ax: matplotlib axes object
        - genotype_data: DataFrame with spatial coordinates and genotype calls
    """
    # Get the gapfill table
    if isinstance(sdata, ad.AnnData):
        table = sdata
    else:
        table = sdata.tables[f'gf_square_{resolution:03d}um']

    # Find the full probe name in the data
    matching_probes = [p for p in table.var.probe.unique() if probe_name in p]

    if len(matching_probes) == 0:
        raise ValueError(f"Probe '{probe_name}' not found in data")
    elif len(matching_probes) > 1:
        print(f"Warning: Multiple probes match '{probe_name}'. Using first match: {matching_probes[0]}")

    probe = matching_probes[0]

    # Normalize the probe name to match the format in wt_alleles/alt_alleles dictionaries
    probe_norm = probe.split("|")
    if len(probe_norm) > 1:
        probe_norm = " ".join(probe_norm[1:3])
    else:
        probe_norm = probe

    # Get available gapfills for this probe to determine if it's a dual probe
    probe_mask = table.var.probe == probe
    probe_table = table[:, probe_mask]
    available_gapfills = probe_table.var.gapfill.unique().tolist()
    is_dual_probe = all(len(gf) == 1 for gf in available_gapfills if gf)  # Single nucleotide gapfills

    # Get WT and ALT alleles for this probe
    if is_dual_probe:
        # For dual probes, extract from probe name (e.g., "AKAP9 c.1389G>T" -> WT='G', ALT='T')
        if ">" in probe_norm:
            variant_part = probe_norm.split()[-1]  # Get "c.1389G>T"
            if ">" in variant_part:
                bases = variant_part.split(">")
                wt_allele = bases[0][-1]  # Last character before '>'
                alt_allele = bases[1]     # Everything after '>'
            else:
                raise ValueError(f"Cannot parse variant notation from probe name: '{probe_norm}'")
        else:
            raise ValueError(f"Dual probe '{probe_norm}' does not contain '>' notation for WT/ALT extraction")
    else:
        # For gapfill probes, use the provided dictionaries
        if probe_norm not in wt_alleles or probe_norm not in alt_alleles:
            raise ValueError(f"WT/ALT alleles not defined for gapfill probe '{probe_norm}' (original: '{probe}')")
        wt_allele = wt_alleles[probe_norm]
        alt_allele = alt_alleles[probe_norm]

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    table.obs['x_coord'] = [c[1] for c in coords]
    table.obs['y_coord'] = [c[0] for c in coords]

    # Get UMI counts for WT and ALT alleles per bin
    probe_mask = table.var.probe == probe
    wt_mask = table.var.gapfill == wt_allele
    alt_mask = table.var.gapfill == alt_allele

    wt_combined_mask = probe_mask & wt_mask
    alt_combined_mask = probe_mask & alt_mask

    if wt_combined_mask.any():
        wt_counts = table[:, wt_combined_mask].X.sum(axis=1)
        if hasattr(wt_counts, 'A1'):
            wt_counts = wt_counts.A1
        wt_counts = wt_counts.flatten()
    else:
        wt_counts = np.zeros(table.shape[0])

    if alt_combined_mask.any():
        alt_counts = table[:, alt_combined_mask].X.sum(axis=1)
        if hasattr(alt_counts, 'A1'):
            alt_counts = alt_counts.A1
        alt_counts = alt_counts.flatten()
    else:
        alt_counts = np.zeros(table.shape[0])

    # Determine genotype for each bin
    # WT = 0, HET = 0.5, ALT = 1, N/A (no coverage) = NaN
    genotype_values = np.full(table.shape[0], np.nan)

    for i in range(table.shape[0]):
        wt_count = wt_counts[i]
        alt_count = alt_counts[i]
        total_count = wt_count + alt_count

        if total_count > 0:
            # Calculate ALT allele fraction
            alt_fraction = alt_count / total_count
            genotype_values[i] = alt_fraction

    # Get spatial coordinates
    x_coords = table.obs['x_coord'].values
    y_coords = table.obs['y_coord'].values

    # Remove NaN coordinates
    valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    genotype_values_valid = genotype_values[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Create 2D matrix for heatmap
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

    for x, y, genotype in zip(x_coords, y_coords, genotype_values_valid):
        heatmap_matrix[y - y_min, x - x_min] = genotype

    # Create figure and axes
    if ax is not None:
        fig = ax.figure if fig is None else fig
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # First, plot lightgray background for bins with WTA coverage
    if not isinstance(sdata, ad.AnnData):
        try:
            if include_he:
                safe_column = f"genotype_{probe.replace('>', 'to').replace('|', '_').replace(' ', '_')}"
                table.obs[safe_column] = genotype_values

                # 1. Use consistent shape key matching your table source
                shape_key = f'gf_square_{resolution:03d}um'
                if shape_key not in sdata.shapes:
                    # Fallback for naming inconsistencies seen in your warnings
                    shape_key = f'_square_{resolution:03d}um'

                table.uns['spatialdata_attrs']['region'] = shape_key
                sanitize_table(table)

                img_element = sdata.images['_hires_image']
                all_systems_dict = get_transformation(img_element, get_all=True)
                target_sys = '' if '' in all_systems_dict else list(all_systems_dict.keys())[0]

                transformed_img = transform(img_element, to_coordinate_system=target_sys)

                sdata_tmp = SpatialData(
                    images={'he_aligned': transformed_img},
                    shapes={shape_key: sdata.shapes[shape_key]},
                    tables={f'table_{shape_key}': table}
                )

                set_transformation(sdata_tmp.shapes[shape_key], Identity(), target_sys)
                set_transformation(sdata_tmp.images['he_aligned'], Identity(), target_sys)

                # 2. Setup Colormap
                # Option: Teal (WT) -> White (HET) -> Orange/Yellow (ALT)
                colors = ['#00FF00', '#FFFFFF', '#FFFF00']
                cmap = mcolors.LinearSegmentedColormap.from_list('he_contrast', colors, N=100)
                cmap.set_bad(alpha=0)

                if ax is None:
                    fig, ax = plt.subplots(figsize=figsize)

                # 3. Render and Show
                # Note the .pl. accessor before show()
                (
                    sdata_tmp.pl.render_images('he_aligned', alpha=1.0)
                    .pl.render_shapes(
                        shape_key,
                        color=safe_column,
                        cmap=cmap,
                        alpha=0.6,
                        vmin=0,
                        vmax=1,
                    )
                    .pl.show(ax=ax, coordinate_systems=target_sys, colorbar=False)
                )

                # 4. Aesthetic Overrides
                ax.set_facecolor('none')
                ax.set_title(title if title else f"Genotype: {probe}", fontsize=14, fontweight='bold')

                # Colorbar matching the non-HE style
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 0.5, 1])
                cbar.set_ticklabels([f'WT ({wt_allele})', 'HET', f'ALT ({alt_allele})'])

                # Prepare return data
                genotype_data = pd.DataFrame({
                    'x': table.obs['x_coord'], 'y': table.obs['y_coord'],
                    'genotype_value': genotype_values,
                    'wt_count': wt_counts, 'alt_count': alt_counts
                }).dropna(subset=['genotype_value'])

                return fig, ax, genotype_data
            else:
                wta_table = sdata.tables[f'square_{resolution:03d}um']
                # Get WTA coverage for each bin
                wta_coverage = wta_table.X.sum(axis=1)
                if hasattr(wta_coverage, 'A1'):
                    wta_coverage = wta_coverage.A1
                wta_coverage = wta_coverage.flatten()

                # Create background matrix
                background_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)

                # Match bins and fill background where WTA coverage exists
                wta_bin_map = {bin_name: idx for idx, bin_name in enumerate(wta_table.obs_names)}
                for i, bin_name in enumerate(table.obs_names):
                    if bin_name in wta_bin_map:
                        wta_idx = wta_bin_map[bin_name]
                        if wta_coverage[wta_idx] > 0:
                            # Get coordinates for this bin
                            x = table.obs['x_coord'].values[i]
                            y = table.obs['y_coord'].values[i]
                            if not np.isnan(x) and not np.isnan(y):
                                background_matrix[int(y) - y_min, int(x) - x_min] = 1

                # Plot background (lightgray for bins with WTA, white for no WTA)
                background_cmap = mcolors.ListedColormap(['#e6e6e6'])
                background_cmap.set_bad(color='white', alpha=0)
                ax.imshow(
                    background_matrix,
                    aspect='auto',
                    origin='upper',
                    cmap=background_cmap,
                    interpolation='nearest',
                    extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
                    vmin=0,
                    vmax=1
                )
        except (KeyError, AttributeError):
            # If WTA table not available, skip background
            pass

    # Create custom colormap: Blue (WT=0) -> Orange (HET=0.5) -> Red (ALT=1)
    colors = ['blue', 'orange', 'red']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('wt_alt', colors, N=n_bins)
    cmap.set_bad(alpha=0)  # Make NaN transparent to show background

    im = ax.imshow(
        heatmap_matrix,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        interpolation='nearest',
        extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
        vmin=0,
        vmax=1
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'WT ({wt_allele})', 'HET', f'ALT ({alt_allele})'])
    cbar.set_label('Genotype', rotation=270, labelpad=20, fontsize=12)

    ax.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax.set_ylabel('Y Coordinate (μm)', fontsize=12)

    # Draw cell line tissue outlines if colors are provided
    if cell_line_colors is not None and not isinstance(sdata, ad.AnnData):
        try:
            wta_table = sdata.tables[f'square_{resolution:03d}um']
            if 'cell_line' in wta_table.obs.columns:
                # Create a matrix for cell line labels
                cell_line_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)

                # Map cell line names to integers
                unique_cell_lines = list(cell_line_colors.keys())
                cell_line_to_idx = {cl: i for i, cl in enumerate(unique_cell_lines)}

                # Match bins and fill cell line matrix
                wta_bin_map = {bin_name: idx for idx, bin_name in enumerate(wta_table.obs_names)}
                for i, bin_name in enumerate(table.obs_names):
                    if bin_name in wta_bin_map:
                        wta_idx = wta_bin_map[bin_name]
                        cell_line = wta_table.obs['cell_line'].iloc[wta_idx]
                        if cell_line in cell_line_to_idx:
                            x = table.obs['x_coord'].values[i]
                            y = table.obs['y_coord'].values[i]
                            if not np.isnan(x) and not np.isnan(y):
                                cell_line_matrix[int(y) - y_min, int(x) - x_min] = cell_line_to_idx[cell_line]

                # Draw contours for each cell line (will automatically handle multiple blobs)
                for cell_line, color in cell_line_colors.items():
                    if cell_line in cell_line_to_idx:
                        idx = cell_line_to_idx[cell_line]
                        # Create binary mask for this cell line
                        mask = (cell_line_matrix == idx).astype(float)
                        # Draw contours around all blobs of this cell line
                        ax.contour(
                            mask,
                            levels=[0.5],
                            colors=[color],
                            linewidths=3,
                            extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
                            origin='upper'
                        )
        except (KeyError, AttributeError):
            # If WTA table or cell_line column not available, skip outlines
            pass

    # Remove ticks
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False
    )
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')

    # Count genotypes for title
    n_wt = np.sum(genotype_values_valid == 0)
    n_het = np.sum((genotype_values_valid > 0) & (genotype_values_valid < 1))
    n_alt = np.sum(genotype_values_valid == 1)
    n_bins_with_data = np.sum(~np.isnan(genotype_values_valid))

    ax.set_title(
        f'WT: {n_wt} bins | HET: {n_het} bins | ALT: {n_alt} bins | Total: {n_bins_with_data} bins' if not title else title,
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # Create summary dataframe
    genotype_data = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'genotype_value': genotype_values_valid,
        'wt_count': wt_counts[valid_mask],
        'alt_count': alt_counts[valid_mask]
    })

    print("Total Spots with WT only:", np.sum(genotype_data['genotype_value'] == 0))
    print("Total Spots with ALT only:", np.sum(genotype_data['genotype_value'] == 1))
    print("Total Spots with HET:", np.sum((genotype_data['genotype_value'] > 0) & (genotype_data['genotype_value'] < 1)))

    return fig, ax, genotype_data


def plot_marker_gene_spatial(
    sdata,
    marker: str,
    color: str = 'gray',
    resolution: int = 16,
    figsize: tuple = (8, 8),
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    title: str = None,
    vmax_quantile: float = 0.9,
):
    """
    Plot spatial distribution of WTA marker gene expression using imshow heatmap.

    Uses the same grid-based plotting approach as plot_wt_alt_alleles_spatial,
    coloring each spatial bin by its expression level for the given gene.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing WTA tables.
    marker : str
        Gene name to plot (must be in the WTA table var_names).
    color : str
        Color for the high end of the colormap (default: 'gray').
    resolution : int
        Resolution in microns (default: 16).
    figsize : tuple
        Figure size (default: (8, 8)).
    fig : plt.Figure, optional
        Existing figure to plot on.
    ax : plt.Axes, optional
        Existing axes to plot on.
    title : str, optional
        Plot title. If None, uses '{marker} Expression'.
    vmax_quantile : float
        Quantile of nonzero values to use as vmax (default: 0.9).

    Returns
    -------
    fig, ax
    """
    from scipy.sparse import issparse

    wta_table = sdata.tables[f'square_{resolution:03d}um']

    # Extract expression values
    marker_vals = wta_table[:, marker].X
    if issparse(marker_vals):
        marker_vals = marker_vals.toarray()
    marker_vals = marker_vals.ravel()

    # Extract spatial coordinates from bin names
    coords = []
    for bin_name in wta_table.obs_names:
        parts = bin_name.split('_')
        if len(parts) >= 4:
            y_coord = int(parts[2])
            x_coord = int(parts[3].split('-')[0])
            coords.append((x_coord, y_coord))
        else:
            coords.append((np.nan, np.nan))

    x_coords = np.array([c[1] for c in coords])
    y_coords = np.array([c[0] for c in coords])

    # Remove NaN coordinates
    valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    x_coords = x_coords[valid_mask].astype(int)
    y_coords = y_coords[valid_mask].astype(int)
    marker_valid = marker_vals[valid_mask]

    if len(x_coords) == 0:
        raise ValueError("No valid spatial coordinates found")

    # Create full grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Build heatmap matrix (NaN for empty bins and zero-count bins so background shows)
    heatmap_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)
    for x, y, val in zip(x_coords, y_coords, marker_valid):
        if val > 0:
            heatmap_matrix[y - y_min, x - x_min] = val

    # Compute vmax from nonzero values
    nonzero_vals = marker_valid[marker_valid > 0]
    if len(nonzero_vals) > 0:
        vmax = np.quantile(nonzero_vals, vmax_quantile)
    else:
        vmax = 1

    # Create figure/axes
    if ax is not None:
        fig = ax.figure if fig is None else fig
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Background: lightgray for bins with WTA coverage
    wta_coverage = wta_table.X.sum(axis=1)
    if hasattr(wta_coverage, 'A1'):
        wta_coverage = wta_coverage.A1
    wta_coverage = np.asarray(wta_coverage).flatten()

    background_matrix = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)
    for i in range(len(x_coords)):
        orig_idx = np.where(valid_mask)[0][i]
        if wta_coverage[orig_idx] > 0:
            background_matrix[y_coords[i] - y_min, x_coords[i] - x_min] = 1

    background_cmap = ListedColormap(['#e6e6e6'])
    background_cmap.set_bad(color='white', alpha=0)
    ax.imshow(
        background_matrix,
        aspect='auto',
        origin='upper',
        cmap=background_cmap,
        interpolation='nearest',
        extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
        vmin=0,
        vmax=1
    )

    # Expression heatmap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', color], N=256)
    cmap.set_bad(alpha=0)

    im = ax.imshow(
        heatmap_matrix,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        interpolation='nearest',
        extent=[x_min * resolution, (x_max + 1) * resolution, (y_max + 1) * resolution, y_min * resolution],
        vmin=0,
        vmax=vmax
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('UMI Counts', rotation=270, labelpad=20, fontsize=12)

    # Clean up axes
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')

    ax.set_title(title if title else f'{marker} Expression', fontsize=14, fontweight='bold')

    return fig, ax