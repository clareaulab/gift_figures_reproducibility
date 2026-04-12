import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import fnmatch
from kneed import KneeLocator, DataGenerator as dg
import scipy
import itertools
import pickle
from Bio.Seq import Seq
from statsmodels.stats.proportion import proportions_ztest
import utils_00 as gf_utils

large_data_dir = '' ### needs to be set

def plot_reads_per_umi(gf_dir = None, probe_reads = None, read_threshold = 0):
    if not isinstance(probe_reads, pd.DataFrame):
        if gf_dir is None:
            print('Please provide either a directory or a dataframe of probe reads')
        else:
            probe_reads = pd.read_csv(gf_dir + 'probe_reads.tsv.gz', sep='\t')
    fig,ax = plt.subplots(figsize=(3,3))
    ax.hist(np.log10(probe_reads['pcr_duplicate_count']),bins=100)
    ax.set_xlabel('log10(PCR duplicate count)')
    ax.set_ylabel('Number of UMIs')
    if read_threshold > 0:
        ax.axvline(np.log10(read_threshold), color='red', linestyle='--', label=f'Read threshold: {read_threshold}')
    plt.tight_layout()
    return fig,ax

def get_barcodes(gf_dir):
    barcodes = pd.read_csv(gf_dir + 'barcodes.tsv.gz', sep='\t',header=0)
    if 'barcode' not in barcodes.columns:
        barcodes = pd.read_csv(gf_dir + 'barcodes.tsv.gz', sep='\t',header=None)
        print('Barcodes opened with  no header.')
    return barcodes

def get_input_probe_reads(gf_dir, read_threshold, min_percent_supporting, adata_path='', min_wta_counts_per_cell=0, min_wta_genes_per_cell=0,collapse_across_probes=False,correct_gapfills=False,filter_unexpected_gapfills=False,filter_unexpected_gapfill_length=False,save_probe_reads=False,keep_umi=False,cell_barcode_suffix='',adata = None, remove_adata_probe_barcode=False, remove_probe_reads_probe_barcode=False):
    if (adata_path != '') and (adata is not None):
        filter_by_adata_cells = True
    else:
        filter_by_adata_cells = False
    if filter_unexpected_gapfill_length + filter_unexpected_gapfills + correct_gapfills > 1:
        print('Please select only one of filter_unexpected_gapfills, filter_unexpected_gapfill_length, or correct_gapfills')
        return
    gapfill_correction = ''
    if filter_unexpected_gapfills:
        gapfill_correction = '_gapfill_filtered'
    if filter_unexpected_gapfill_length:
        gapfill_correction = '_gapfill_length_filtered'
    if correct_gapfills:
        gapfill_correction = '_gapfill_corrected'
    filename = gf_dir + 'probe_reads_' + str(read_threshold) + '_' + str(min_percent_supporting) + '_' + str(filter_by_adata_cells)[0] + '_' + str(collapse_across_probes)[0] + gapfill_correction + '.tsv.gz'
    if os.path.exists(filename):
        probe_reads = pd.read_csv(filename,sep='\t',compression='gzip')
        print('opening existing file:',filename)
        barcodes = get_barcodes(gf_dir)
        probe_reads = probe_reads.merge(barcodes,left_on='cell_idx',right_index=True,how='left')
        probe_reads.rename(columns={0: 'barcode'}, inplace=True)
    else:
        barcodes = get_barcodes(gf_dir)
        probe_reads = pd.read_csv(gf_dir + 'probe_reads.tsv.gz', sep='\t')
        if 'barcode' not in probe_reads.columns:
            probe_reads = probe_reads.merge(barcodes,left_on='cell_idx',right_index=True,how='left')
        probe_reads['barcode'] = probe_reads['barcode'] + cell_barcode_suffix
        if 'pcr_duplicate_count' not in probe_reads.columns:
            if 'umi_count' in probe_reads.columns:
                probe_reads.rename(columns={'umi_count':'pcr_duplicate_count'},inplace=True)
            else:
                print('Probe reads missing PCR duplicate count')
                return
        probe_reads.rename(columns={0: 'barcode'}, inplace=True)
        print(f'{len(probe_reads)} UMIs found')
        initial_count = len(probe_reads)
        if collapse_across_probes:
            probe_reads = probe_reads.sort_values('pcr_duplicate_count',ascending=False).drop_duplicates(['cell_idx','umi']).sort_values(['cell_idx','probe_idx'])
            print(f'Collapsing UMIs across probes, {len(probe_reads)} UMIs remaining ({(len(probe_reads) / initial_count * 100):.2f}%)')
        probe_reads = probe_reads.loc[probe_reads['pcr_duplicate_count'] > read_threshold]
        if 'percent_supporting' in probe_reads.columns:
            probe_reads = probe_reads.loc[probe_reads['percent_supporting'] > min_percent_supporting]
        else:
            print('Percent supporting not found in probe reads; ignoring min_percent_supporting')
        print(f'Filtering probe reads based on read threshold ({read_threshold}) and min percent supporting ({min_percent_supporting}), {len(probe_reads)} UMIs remaining ({(len(probe_reads) / initial_count * 100):.2f}%)')
        probe_reads['gapfill'] = probe_reads['gapfill'].fillna('')
        if adata_path != '':
            adata = read_adata(adata_path)
        if isinstance(adata, sc.AnnData):
            if remove_adata_probe_barcode:
                adata.obs.index = adata.obs.index.str[0:16] + adata.obs.index.str[-2:]
            if remove_probe_reads_probe_barcode:
                probe_reads['barcode'] = probe_reads['barcode'].str[0:16] + probe_reads['barcode'].str[-2:]
            if ('n_genes_by_counts' not in adata.obs.columns) or ('total_counts' not in adata.obs.columns):
                sc.pp.calculate_qc_metrics(adata, inplace = True)
            print(f'Filtering cells based on min counts ({min_wta_counts_per_cell}) and genes ({min_wta_genes_per_cell}) in WTA')
            sc.pp.filter_cells(adata,min_counts=min_wta_counts_per_cell)
            sc.pp.filter_cells(adata,min_genes=min_wta_genes_per_cell)
            probe_reads = probe_reads.loc[probe_reads['barcode'].isin(adata.obs.index)]
            print(f'Filtering probe reads based on cell barcodes in adata, {len(probe_reads)} UMIs remaining ({(len(probe_reads) / initial_count * 100):.2f}%)')
            if 'cell_type' in adata.obs.columns:
                probe_reads = probe_reads.merge(adata.obs['cell_type'],left_on='barcode',right_index=True,how='left')
        if correct_gapfills:
            print('Correcting truncated and expanded gapfills')
            manifest = get_manifest(gf_dir)
            probe_reads,correction_counts_df = correct_all_truncated_expanded_gapfills(probe_reads,manifest,return_stats=True)
            print('Correction stats:')
            stats = correction_counts_df.sum()
            for stat_type in stats.index:
                if stat_type != 'n_total':
                    print('\t' + stat_type.replace('n_','') + ': ' + str(stats[stat_type]/stats['n_total']))
        if filter_unexpected_gapfills:
            print('Filtering unexpected gapfills')
            manifest = get_manifest(gf_dir).fillna('')
            probe_reads = filter_gapfills_matching_expection(probe_reads,manifest)
        if filter_unexpected_gapfill_length:
            print('Filtering unexpected gapfill lengths')
            manifest = get_manifest(gf_dir).fillna('')
            probe_reads = filter_gapfills_matching_expection_length(probe_reads,manifest)
        if save_probe_reads:
            print('saving file:',filename)
            if keep_umi:
                probe_reads.drop(['barcode'],axis=1).to_csv(filename, index=False, sep='\t', compression='gzip')
            else:
                probe_reads.drop(['barcode','umi'],axis=1).to_csv(filename, index=False, sep='\t', compression='gzip')
    return probe_reads

def filter_gapfills_matching_expection(probe_reads,manifest):
    probe_indices = manifest.loc[manifest['gapfill_from_transcriptome'].notna() | manifest['gap_probe_sequence'].notna()].index
    to_filter = probe_reads.loc[probe_reads['probe_idx'].isin(probe_indices)]
    to_filter = to_filter.merge(manifest[['gapfill_from_transcriptome','gap_probe_sequence']],left_on='probe_idx',right_index=True)
    to_filter = to_filter.loc[(to_filter['gapfill'] == to_filter['gapfill_from_transcriptome']) | (to_filter['gapfill'] == to_filter['gap_probe_sequence'])]
    filtered_probe_reads = pd.concat([to_filter.drop(['gapfill_from_transcriptome','gap_probe_sequence'],axis=1),probe_reads.loc[~probe_reads['probe_idx'].isin(probe_indices)]])
    filtered_probe_reads = filtered_probe_reads.sort_values(['cell_idx','probe_idx'])
    return filtered_probe_reads

def filter_gapfills_matching_expection_length(probe_reads,manifest):
    probe_indices = manifest.loc[manifest['gapfill_from_transcriptome'].notna() | manifest['gap_probe_sequence'].notna()].index
    to_filter = probe_reads.loc[probe_reads['probe_idx'].isin(probe_indices)]
    to_filter = to_filter.merge(manifest[['gapfill_from_transcriptome','gap_probe_sequence']],left_on='probe_idx',right_index=True)
    c1 = False
    c2 = False
    if to_filter['gapfill_from_transcriptome'].str.len().sum() != 0:
        c1 = to_filter['gapfill'].str.len() == to_filter['gapfill_from_transcriptome'].str.len()
    if to_filter['gap_probe_sequence'].str.len().sum() != 0:
        c2 = to_filter['gapfill'].str.len() == to_filter['gap_probe_sequence'].str.len()
    to_filter = to_filter.loc[c1 | c2]
    filtered_probe_reads = pd.concat([to_filter.drop(['gapfill_from_transcriptome','gap_probe_sequence'],axis=1),probe_reads.loc[~probe_reads['probe_idx'].isin(probe_indices)]])
    filtered_probe_reads = filtered_probe_reads.sort_values(['cell_idx','probe_idx'])
    return filtered_probe_reads

def read_adata(adata_path):
    if adata_path.endswith('.h5'):
        adata = sc.read_10x_h5(adata_path)
    elif adata_path.endswith('.h5ad'):
        adata = sc.read_h5ad(adata_path)
    else:
        print('Please provide a valid adata file')
    return adata

def get_manifest(gf_dir, add_flex_info=False):
    try:
        manifest = pd.read_csv(gf_dir + 'manifest_detailed.tsv', sep='\t')
    except FileNotFoundError:
        manifest = pd.read_csv(gf_dir + 'manifest.tsv', sep='\t')
    if add_flex_info:
        manifest = add_if_gene_in_flex(manifest)
    manifest.index = manifest['index']
    manifest.index.set_names('', inplace=True)
    return manifest

def get_genes_in_flex():
    flex_probes = pd.read_csv(os.path.dirname(__file__) + '/resources/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2020-A.csv', sep=',', comment='#')
    gene_info = pd.read_csv(os.path.dirname(__file__) + '/resources/geneInfo.tab', sep='\t').reset_index().rename(columns={'level_0':'gene_id', 'level_1':'gene_name'})
    flex_probes['gene'] = flex_probes['probe_id'].str.split('|',expand=True)[0].map(dict(zip(gene_info['gene_id'], gene_info['gene_name'])))
    return flex_probes.drop_duplicates(subset='gene')

def add_if_gene_in_flex(manifest,name_delimiter=' ',name_loc=0):
    if 'gene' not in manifest.columns:
        manifest['gene'] = manifest['name'].str.split(name_delimiter).str[name_loc]
    manifest = manifest.merge(get_genes_in_flex()['gene'], on='gene', how='left', indicator=True)
    manifest['gene_in_flex'] = manifest['_merge'] == 'both'
    manifest = manifest.drop(columns=['_merge'])
    return manifest

def correct_truncated_expanded_gapfills(input_probe_reads, expected_gapfills, truncation_size=2):
    new_gapfills = []
    for expected_gapfill in expected_gapfills:
        if 'N' in expected_gapfill:
            expected_gapfill = expected_gapfill.replace('N', '?') ## change N to ? for compatibility with fnmatch
        else:
            expected_gapfill = expected_gapfill
        new_gapfills.append(expected_gapfill)
    expected_gapfills = pd.Series(new_gapfills)
    probe_reads = input_probe_reads.copy()
    n_truncated = 0
    n_second_to_last_deletion = 0
    n_left_insertion = 0
    n_right_insertion = 0
    n_total = 0
    probe_reads = input_probe_reads.copy()
    min_len = min(expected_gapfills.str.len())
    max_len = max(expected_gapfills.str.len())

    for i in probe_reads.index:
        n_total += 1
        gapfill = probe_reads.loc[i, 'gapfill']
        gapfill_len = len(gapfill)

        if gapfill_len < max_len:
            top_keys = [expected_gapfill for expected_gapfill in expected_gapfills
                        if (fnmatch.fnmatch(gapfill,expected_gapfill[:len(gapfill)])) & (len(expected_gapfill) <= gapfill_len + truncation_size)]
            if len(top_keys) == 1:
                probe_reads.loc[i, 'gapfill'] = top_keys[0].replace('?','N')
                n_truncated += 1
            elif len(top_keys) == 0:
                top_keys = [expected_gapfill for expected_gapfill in expected_gapfills
                            if fnmatch.fnmatch(gapfill,expected_gapfill[:-2] + expected_gapfill[-1])]
                if len(top_keys) == 1:
                    probe_reads.loc[i, 'gapfill'] = top_keys[0].replace('?','N')
                    n_second_to_last_deletion += 1
                    
        elif gapfill_len > min_len:
            top_keys = [expected_gapfill for expected_gapfill in expected_gapfills
                        if (fnmatch.fnmatch(gapfill[:-1],expected_gapfill) | fnmatch.fnmatch(gapfill[1:],expected_gapfill))]
            if len(top_keys) == 1:
                if fnmatch.fnmatch(gapfill[:-1],top_keys[0]):
                    probe_reads.loc[i, 'gapfill'] = gapfill[:-1]
                    n_right_insertion += 1
                else:
                    probe_reads.loc[i, 'gapfill'] = gapfill[1:]
                    n_left_insertion += 1
    return probe_reads, n_truncated, n_second_to_last_deletion, n_left_insertion, n_right_insertion, n_total

def correct_all_truncated_expanded_gapfills(probe_reads,manifest,return_stats=True):
    corrected_probe_reads = probe_reads.copy()
    # Initialize dictionaries to store counts
    correction_counts = {
        'n_truncated': {},
        'n_second_to_last_deletion': {},
        'n_left_insertion': {},
        'n_right_insertion': {},
        'n_total': {}
    }
    # Process each probe index
    for probe_idx in sorted(probe_reads['probe_idx'].unique()):
        expected_gapfills = manifest.loc[probe_idx].reindex(['gap_probe_sequence', 'original_gap_probe_sequence', 'gapfill_from_transcriptome']).dropna()
        if len(expected_gapfills) > 0:
            corrected_sub, correction_counts['n_truncated'][probe_idx], correction_counts['n_second_to_last_deletion'][probe_idx], correction_counts['n_left_insertion'][probe_idx], correction_counts['n_right_insertion'][probe_idx], correction_counts['n_total'][probe_idx] = correct_truncated_expanded_gapfills(corrected_probe_reads.loc[corrected_probe_reads['probe_idx'] == probe_idx], expected_gapfills)
            corrected_probe_reads.loc[corrected_probe_reads['probe_idx'] == probe_idx] = corrected_sub
        else:
            correction_counts['n_truncated'][probe_idx] = 0
            correction_counts['n_second_to_last_deletion'][probe_idx] = 0
            correction_counts['n_left_insertion'][probe_idx] = 0
            correction_counts['n_right_insertion'][probe_idx] = 0
            correction_counts['n_total'][probe_idx] = 0
    # Convert correction counts to a DataFrame
    correction_counts_df = pd.DataFrame(correction_counts)
    if return_stats:
        return corrected_probe_reads, correction_counts_df
    else:
        return corrected_probe_reads


def get_p(allele1,allele2,fracs_1,sub_probe_reads):
    prob_columns = []
    for frac_1 in fracs_1:
        frac_2 = 1 - frac_1
        sub_probe_reads[allele1] = sub_probe_reads['p_' + allele1] * frac_1
        sub_probe_reads[allele2] = sub_probe_reads['p_' + allele2] * frac_2
        col_name = 'p_' + allele1 + '_' + allele2 + '_' + str(frac_1) + '_' + str(frac_2)
        if frac_1 == 0 and frac_2 == 1:
            col_name = 'p_' + allele2 + '_1'
        elif frac_1 == 1 and frac_2 == 0:
            col_name = 'p_' + allele1 + '_1'
        sub_probe_reads[col_name] = sub_probe_reads[['pcr_swap_likelihood', allele1, allele2]].max(axis=1)
        prob_columns.append(col_name)
        allele_call_col = col_name.replace('p_','') + '_n_allele_1'
        sub_probe_reads[allele_call_col] = sub_probe_reads[['pcr_swap_likelihood', allele1, allele2]].idxmax(axis=1)
        sub_probe_reads.loc[sub_probe_reads[allele_call_col] == 'pcr_swap_likelihood',allele_call_col] = None
        sub_probe_reads.loc[sub_probe_reads[allele_call_col] == allele2,allele_call_col] = 0
        sub_probe_reads.loc[sub_probe_reads[allele_call_col] == allele1,allele_call_col] = 1
        sub_probe_reads.drop(columns=[allele1, allele2], inplace=True)
    return sub_probe_reads, prob_columns

def update_het_frac(sub_probe_reads, prob_columns):
    cell_genotypes = sub_probe_reads.groupby(['cell_idx', 'barcode'])[prob_columns].prod()
    cell_genotypes = cell_genotypes.div(cell_genotypes.sum(axis=1), axis=0)
    het_cols = cell_genotypes.columns[~(cell_genotypes.columns.str.contains('0_1') | cell_genotypes.columns.str.contains('1_0'))]
    het_cells = cell_genotypes.loc[cell_genotypes[het_cols].sum(axis=1) > 0.8].index.get_level_values('barcode').unique()
    if len(het_cells) > 20:
        cols = sub_probe_reads.columns[(sub_probe_reads.columns.str.contains('n_allele_1')) & ~(sub_probe_reads.columns.str.contains('0_1')) & ~(sub_probe_reads.columns.str.contains('1_0'))]
        updated_frac1 = sub_probe_reads.loc[sub_probe_reads['barcode'].isin(het_cells)].groupby(['cell_idx','barcode'])[cols].mean().mean().mean()
        return updated_frac1
    else:
        return 0.5

def get_cell_genotypes(probe_idx,probe_reads,fracs_1 = [0,0.5,1], learn_het_frac = True):
    sub_probe_reads = probe_reads.loc[probe_reads['probe_idx'] == probe_idx]
    sub_probe_reads = sub_probe_reads.dropna(axis=1, how='all')
    possible_alleles = sub_probe_reads.columns[sub_probe_reads.columns.str.contains('p_gapfill_given_')]
    sub_probe_reads[possible_alleles.str.replace('_gapfill_given','')] = sub_probe_reads[possible_alleles].div(sub_probe_reads[possible_alleles].sum(axis=1), axis=0)
    sub_probe_reads = sub_probe_reads.drop(columns=possible_alleles)
    possible_alleles = possible_alleles.str.replace('p_gapfill_given_','')
    all_prob_columns = []

    if len(possible_alleles) == 1:
        allele1 = possible_alleles[0]
        sub_probe_reads, prob_columns = get_p(allele1, allele1, [0], sub_probe_reads)
        all_prob_columns.extend(prob_columns)

    else:
        for allele_combination in list(combinations(possible_alleles, 2)):
            allele1 = allele_combination[0]
            allele2 = allele_combination[1]
            sub_probe_reads, prob_columns = get_p(allele1, allele2, fracs_1, sub_probe_reads)
            all_prob_columns.extend(prob_columns)
            if learn_het_frac:
                updated_het_frac = np.round(update_het_frac(sub_probe_reads, prob_columns),2)
                sub_probe_reads, prob_columns = get_p(allele1, allele2, [0,updated_het_frac,1], sub_probe_reads)
                all_prob_columns.extend(prob_columns)
    all_prob_columns = list(set(all_prob_columns))
    cell_genotypes = sub_probe_reads.groupby(['cell_idx', 'barcode'])[all_prob_columns].prod()
    cell_genotypes = cell_genotypes.div(cell_genotypes.sum(axis=1), axis=0)
    cell_genotypes = cell_genotypes.reset_index()
    for probe_name in cell_genotypes.columns[cell_genotypes.columns.str.contains('_wt')].str.replace('_wt',''):
        cell_genotypes[probe_name + '_high_confidence_counts'] = cell_genotypes['barcode'].map(sub_probe_reads.loc[sub_probe_reads['pcr_swap_likelihood'] < 0.1]['barcode'].value_counts().to_dict()).fillna(0)
    return cell_genotypes

def get_genotyped_adata(probe_reads, adata, variants):
    if 'genotypes' in adata.obsm:
        del adata.obsm['genotypes']
    col_map = {'p_alt_1': 'mutated', 'p_ref_alt_0.5_0.5': 'heterozygous', 'p_ref_1': 'wt'}
    for probe_idx in probe_reads.loc[probe_reads['probe_idx'].isin(variants)]['probe_idx'].sort_values().unique():
        cell_genotypes = get_cell_genotypes(probe_idx,probe_reads,learn_het_frac = False)
        cell_genotypes.columns = cell_genotypes.columns.map(lambda x: col_map.get(x, x))
        cell_genotypes.drop('cell_idx', axis=1, inplace=True)
        cell_genotypes.set_index('barcode', inplace=True)
        cell_genotypes.dropna(subset=['mutated', 'heterozygous', 'wt'], inplace=True)
        cell_genotypes['genotype_call'] = cell_genotypes[['wt','mutated','heterozygous']].idxmax(axis=1)
        cell_genotypes = cell_genotypes[['genotype_call']].copy()
        cell_genotypes.columns = [probe_idx]
        if 'genotypes' not in adata.obsm:
            adata.obsm['genotypes'] = cell_genotypes.reindex(adata.obs.index).copy()
        else:
            adata.obsm['genotypes'] = adata.obsm['genotypes'].merge(cell_genotypes, left_index=True, right_index=True, how='left')
    return adata

def define_gene_score(adata, gene_set, cell_type_name, subset=None, rename=False):
    genes_present = []
    for gene_name in gene_set:
        if gene_name in adata.var_names:
            genes_present.append(gene_name)
    geneset_id = [adata.var_names.get_loc(j) for j in genes_present]

    adata.obs['gene_score_' + cell_type_name] = np.mean(adata.layers['zs_norm_log'][:, geneset_id], axis = 1)
    if rename:
        high_score_clusters = adata.obs.groupby('pheno_leiden', observed=False)['gene_score_' + cell_type_name].mean()
        high_score_clusters = high_score_clusters[high_score_clusters > 2].index.tolist()
        if subset is not None:
            high_score_clusters = [j for j in high_score_clusters if j in subset]
        to_rename = adata.obs.loc[(adata.obs['pheno_leiden'].isin(high_score_clusters)) & (adata.obs['cell_type'].isna()), 'cell_type'].copy()
        if len(to_rename) > 0:
            if isinstance(adata.obs['cell_type'].dtype, pd.CategoricalDtype):
                if cell_type_name not in adata.obs['cell_type'].cat.categories:
                    adata.obs['cell_type'] = adata.obs['cell_type'].cat.add_categories([cell_type_name])
            adata.obs.loc[(adata.obs['pheno_leiden'].isin(high_score_clusters)) & (adata.obs['cell_type'].isna()), 'cell_type'] = cell_type_name


### likelihood model functions

def add_expected_gapfills_for_pcr_swap_likelihood(expected_genotypes_file, preloaded_probe_reads):
    probe_reads = preloaded_probe_reads.copy()
    if isinstance(expected_genotypes_file, str):
        expected_genotypes = pd.read_csv(expected_genotypes_file, index_col=0)
    else:
        expected_genotypes = expected_genotypes_file.copy()
    probe_idx_with_multiple_gapfills = expected_genotypes.groupby('probe_idx')['expected_gapfill'].nunique()
    probe_idx_with_multiple_gapfills = probe_idx_with_multiple_gapfills[probe_idx_with_multiple_gapfills > 1].index.tolist()
    print('number of probes to use:',len(probe_idx_with_multiple_gapfills)) ## note that this excludes heterozygous cell types and INSR which was not covered by bulk
    probe_reads = probe_reads.loc[probe_reads['probe_idx'].isin(probe_idx_with_multiple_gapfills)]
    ## add expected genotype by probe idx / cell type
    probe_reads = probe_reads.merge(expected_genotypes, on = ['probe_idx','cell_type'], how = 'left')
    probe_reads = probe_reads.loc[probe_reads['expected_gapfill'].notna()]
    ### try excluding gapfills that are not wt or alt
    possible_gapfills = probe_reads[['probe_idx','expected_gapfill']].drop_duplicates().sort_values('probe_idx')
    possible_gapfills_pivoted = possible_gapfills.groupby('probe_idx')['expected_gapfill'].apply(list).reset_index()
    possible_gapfills_pivoted[['gapfill_1', 'gapfill_2']] = pd.DataFrame(possible_gapfills_pivoted['expected_gapfill'].tolist(), index=possible_gapfills_pivoted.index)
    possible_gapfills_pivoted = possible_gapfills_pivoted.drop(columns=['expected_gapfill'])
    probe_reads = probe_reads.merge(possible_gapfills_pivoted, on = 'probe_idx', how = 'left')
    probe_reads = probe_reads.loc[((probe_reads['gapfill'] == probe_reads['gapfill_1']) | (probe_reads['gapfill'] == probe_reads['gapfill_2']))]
    return probe_reads

def get_likelihood(probe_reads):
    try:
        likelihood = (probe_reads['gapfill'] == probe_reads['expected_gapfill']).sum() / len(probe_reads)
    except KeyError:
        likelihood = 0
    return likelihood

def get_likelihood_list(probe_reads,min_size):
    likelihoods = []
    x = []
    start_threshold = 1
    while start_threshold <= probe_reads['pcr_duplicate_count'].max():
        window = probe_reads.loc[probe_reads['pcr_duplicate_count'] == start_threshold]
        end_threshold = start_threshold
        while len(window) < min_size:
            window = probe_reads.loc[(probe_reads['pcr_duplicate_count'] >= start_threshold) & (probe_reads['pcr_duplicate_count'] <= end_threshold)]
            end_threshold += 1
            if end_threshold > probe_reads['pcr_duplicate_count'].max():
                break
        if len(probe_reads.loc[(probe_reads['pcr_duplicate_count'] >= (end_threshold+1))]) < (0.8*min_size):
            window = probe_reads.loc[(probe_reads['pcr_duplicate_count'] >= start_threshold)]
            start_threshold = (probe_reads['pcr_duplicate_count'].max() + 1)
            end_threshold = (probe_reads['pcr_duplicate_count'].max() + 1)
        else:
            start_threshold = end_threshold + 1
        likelihood = get_likelihood(window)
        likelihoods.append(likelihood)
        x.append(window['pcr_duplicate_count'].mean())
    return x,np.array(likelihoods)

def get_knee(probe_reads, min_threshold):
    sorted_counts = np.log10(probe_reads['pcr_duplicate_count']).sort_values(ascending=True)
    sorted_counts = sorted_counts[sorted_counts > min_threshold]
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    x, y = dg.concave_decreasing()
    kl = KneeLocator(cdf, sorted_counts, curve="concave", direction="increasing", interp_method="polynomial")
    return kl

def sample_reference_probe_reads(probe_reads, ratio, plot=False, comparator_probe_reads=None):
    n_sample = int(probe_reads['pcr_duplicate_count'].sum() * ratio)
    sampled = probe_reads.sample(n=n_sample, weights=probe_reads['pcr_duplicate_count'],replace=True).index.value_counts()
    probe_reads.loc[sampled.index,'pcr_duplicate_count'] = sampled.values
    if plot and (comparator_probe_reads is not None):
        fig,ax = plt.subplots(figsize=(4,4))
        hist = ax.hist(np.log10(comparator_probe_reads['pcr_duplicate_count']),bins=100,density=True, label='test')
        hist = ax.hist(np.log10(probe_reads['pcr_duplicate_count']),bins=100,alpha=0.5,density=True, label='reference (sampled)')
        ax.legend()
        ax.set_xlabel('log10(PCR Duplicate Count)', size=12)
        ax.set_ylabel('UMIs (density)', size=12)
        plt.tight_layout()
    return probe_reads
    
def permute_gapfills(probe_reads, ratio):
    probe_reads = probe_reads.copy()
    probe_reads['new_gapfill'] = probe_reads['expected_gapfill']
    to_permute = probe_reads.sample(frac=ratio, random_state=1).index
    permuted = probe_reads.loc[to_permute]
    probe_reads.loc[to_permute, 'new_gapfill'] = permuted.groupby('probe_idx')['expected_gapfill'].transform(np.random.permutation)
    frac_correct = (probe_reads['new_gapfill'] == probe_reads['expected_gapfill']).sum() / len(probe_reads)
    return frac_correct

def fit_likelihoods(i,duplicate_count,likelihoods,swap_slope=None,swap_intercept=None):
    if i == 0:
        return None
    if i in duplicate_count.values:
        to_return = (likelihoods[duplicate_count[duplicate_count == i].index[0]])
    else:
        below = (duplicate_count[duplicate_count < i])
        above = (duplicate_count[duplicate_count > i])
        if (not below.empty) & (not above.empty):
            if len(below) == 1:
                idx_below = below.index
            else:
                idx_below = below.nlargest(2).index
            if len(above) == 1:
                idx_above = above.index
            else:
                idx_above = above.nsmallest(2).index
            idx_above = above.nsmallest(2).index
            x_vals = pd.concat([duplicate_count[idx_below], duplicate_count[idx_above]])
            y_vals = pd.concat([pd.Series(likelihoods).loc[idx_below], pd.Series(likelihoods).loc[idx_above]])
            slope, intercept, _, _, _ = scipy.stats.linregress(x_vals, y_vals)
            to_return = ((i * slope) + intercept)
        else:
            if not below.empty:
                idx_below = below.idxmax()
                ## just take the last value
                to_return = likelihoods[idx_below]
            elif not above.empty:
                idx_above = above.idxmin()
                ## just take the first value
                to_return = likelihoods[idx_above]
            else:
                print(i,'both empty')
                to_return = None
    if to_return == 0:
        to_return = 0.001
    if swap_slope is not None:
        to_return = 1 - ((to_return * swap_slope) + swap_intercept)
    return to_return

def get_swap_probabilities(probe_reads_patient,probe_reads_cl, x,likelihoods, plot=True):

    ## first use permutation to get a baseline for fraction correct depending on fraction swapped. This can then get a probability of swapping 
    # given the proportion correct in ground truth cell line data

    frac_correct = {}
    for ratio in np.arange(0, 1.1, 0.1):
        frac_correct[ratio] = permute_gapfills(probe_reads_cl, ratio)

    slope, intercept = np.polyfit(list(frac_correct.values()), list(frac_correct.keys()), 1)

    ## now use above to get likelihoods for the patient data
    counts = probe_reads_patient.dropna()['pcr_duplicate_count'].drop_duplicates().dropna().sort_values()
    counts.index = counts
    prob_real = counts.apply(lambda count: fit_likelihoods(count, pd.Series(x), likelihoods, swap_slope=slope, swap_intercept=intercept)).rename('likelihood')
    
    if plot:
        fig,ax = plt.subplots(figsize=(4,4))
        ax.scatter(np.log10(prob_real.index),prob_real)
        ax.set_xlabel('log10(PCR Duplicate Count)', size=12)
        ax.set_ylabel('Likelihood', size=12)
        plt.tight_layout()
    return prob_real

def sample_and_get_swap_probabilities(probe_reads_patient, probe_reads_cl, patient_min_threshold, cl_min_threshold, expected_genotypes_file, return_likelihood_list=False):
    ## get the knee point for the patient
    kl_patient = get_knee(probe_reads_patient, min_threshold = np.log10(patient_min_threshold)) ## min_threshold is an inclusive threshold above obvious background
    kl_cl = get_knee(probe_reads_cl, min_threshold = np.log10(cl_min_threshold)) ## min_threshold is an inclusive threshold above obvious background

    probe_reads_cl_sampled = sample_reference_probe_reads(probe_reads_cl, (10**kl_patient.knee_y)/ (10**kl_cl.knee_y), plot=True, comparator_probe_reads=probe_reads_patient)

    probe_reads_cl_sampled = add_expected_gapfills_for_pcr_swap_likelihood(expected_genotypes_file, probe_reads_cl_sampled)
    x,likelihoods = get_likelihood_list(probe_reads_cl_sampled,min_size=2000)
    prob_real = get_swap_probabilities(probe_reads_patient,probe_reads_cl_sampled,x,likelihoods,plot=True)
    if return_likelihood_list:
        return prob_real, x, likelihoods
    return prob_real

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def get_likelihoods_of_true_allele(gapfill, error_rate_dicts):
    likelihoods = {}
    ##### first simulate an RHS truncation (gapfill is truncated from true allele) or insertion (gapfill has extra bp)
    ##### gives probability of the observed given each possible true allele
    for rhs_n_truncation,rhs_p_truncation in error_rate_dicts['rhs_truncation_probabilities'].items():
        for perm in itertools.product('ACGT', repeat=rhs_n_truncation):
            test_gapfill = {}
            likelihood = {}
            i = 0
            test_gapfill[i] = gapfill + ''.join(perm)
            likelihood[i] = rhs_p_truncation
            for rhs_n_insertion,rhs_p_insertion in error_rate_dicts['rhs_insertion_probabilities'].items():
                i = 1
                cont = False
                if (rhs_n_truncation == 0) or (rhs_n_insertion == 0): ## only allow rhs insertion if not truncated
                    if rhs_n_insertion <= len(test_gapfill[i-1]):
                        likelihood[i] = likelihood[i-1] * rhs_p_insertion
                        if rhs_n_insertion == 0:
                            test_gapfill[i] = test_gapfill[i-1]
                        else:
                            test_gapfill[i] = test_gapfill[i-1][:-rhs_n_insertion]
                        cont = True
                if cont:
                    #### now simulate an LHS truncation or insertion
                    for lhs_n_truncation,lhs_p_truncation in error_rate_dicts['lhs_truncation_probabilities'].items():
                        for perm in itertools.product('ACGT', repeat=lhs_n_truncation):
                            i = 2
                            cont = False
                            test_gapfill[i] = ''.join(perm) + test_gapfill[i-1]
                            likelihood[i] = lhs_p_truncation * (likelihood[i-1])
                            for lhs_n_insertion,lhs_p_insertion in error_rate_dicts['lhs_insertion_probabilities'].items():
                                i = 3
                                cont = False
                                if (lhs_n_truncation == 0) or (lhs_n_insertion == 0): ## only allow lhs insertion if not truncated
                                    if lhs_n_insertion <= len(test_gapfill[i-1]):
                                        likelihood[i] = likelihood[i-1] * lhs_p_insertion
                                        test_gapfill[i] = test_gapfill[i-1][lhs_n_insertion:]
                                        cont = True
                                if cont:
                                    #### finally, simulate SNVs
                                    i = 4
                                    max_snv = max(error_rate_dicts['snv_probabilities'].keys())
                                    max_snv = min(max_snv, len(test_gapfill[i-1]))
                                    added = set()
                                    for pos in itertools.combinations(range(len(test_gapfill[i-1])), max_snv):
                                        for replacements in itertools.product('ACGT', repeat=max_snv):
                                            variant = list(test_gapfill[i-1])
                                            for p, rep in zip(pos, replacements):
                                                variant[p] = rep
                                            candidate = ''.join(variant)
                                            if candidate not in added:
                                                added.add(candidate)
                                                test_gapfill[i] = candidate
                                                dist = hamming_distance(test_gapfill[i],test_gapfill[i-1])
                                                likelihood[i] = error_rate_dicts['snv_probabilities'][dist] * likelihood[i-1]
                                                if test_gapfill[i] in likelihoods.keys():
                                                    likelihoods[test_gapfill[i]] += likelihood[i]
                                                else:
                                                    likelihoods[test_gapfill[i]] = likelihood[i]
    return likelihoods

def get_error_rate_dicts(dir, n_lhs_truncation = 3, n_rhs_truncation = 3, n_lhs_insertion = 2, n_rhs_insertion = 2, n_snv = 2):
    error_rate_dicts = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(dir, filename), 'rb') as f:
                error_rate_dicts[filename.replace('.pkl','')] = pickle.load(f)
    required_keys = [
        'rhs_truncation_probabilities', 
        'lhs_truncation_probabilities', 
        'rhs_insertion_probabilities', 
        'lhs_insertion_probabilities', 
        'snv_probabilities'
    ]
    for key in required_keys:
        if key not in error_rate_dicts:
            raise KeyError(f"Missing required key for error rate dicts: {key}")
    
    # ## for now, shorten the dictionaries to speed up computation
    error_rate_dicts['rhs_truncation_probabilities'] = {k:v for k,v in error_rate_dicts['rhs_truncation_probabilities'].items() if k in range(n_rhs_truncation)}
    error_rate_dicts['lhs_truncation_probabilities'] = {k:v for k,v in error_rate_dicts['lhs_truncation_probabilities'].items() if k in range(n_lhs_truncation)}
    error_rate_dicts['rhs_insertion_probabilities'] = {k:v for k,v in error_rate_dicts['rhs_insertion_probabilities'].items() if k in range(n_rhs_insertion)}
    error_rate_dicts['lhs_insertion_probabilities'] = {k:v for k,v in error_rate_dicts['lhs_insertion_probabilities'].items() if k in range(n_lhs_insertion)}
    error_rate_dicts['snv_probabilities'] = {k:v for k,v in error_rate_dicts['snv_probabilities'].items() if k in range(n_snv)}
    
    return error_rate_dicts

def reversecomplement(seq):
    return str(Seq(seq).reverse_complement())

def get_hgvs_from_gapfill(ref: str, test: str, gapfill_start: int = 1, return_all: bool = False, revcomp: bool = False) -> list | str:
    if not isinstance(ref, str) or not isinstance(test, str) or not isinstance(gapfill_start, int):
        return None
    
    if revcomp:
        ref = reversecomplement(ref)
        test = reversecomplement(test)

    if ref == test:
        return ["c.="] if return_all else "c.="

    # SNV detection early
    if len(ref) == len(test):
        diffs = [i for i in range(len(ref)) if ref[i] != test[i]]
        if len(diffs) == 1:
            pos = gapfill_start + diffs[0]
            return [f"c.{pos}{ref[diffs[0]]}>{test[diffs[0]]}"] if return_all else f"c.{pos}{ref[diffs[0]]}>{test[diffs[0]]}"

    # Find leftmost and rightmost differences
    left = 0
    while left < len(ref) and left < len(test) and ref[left] == test[left]:
        left += 1

    right_ref = len(ref)
    right_test = len(test)
    while right_ref > left and right_test > left and ref[right_ref - 1] == test[right_test - 1]:
        right_ref -= 1
        right_test -= 1

    ref_segment = ref[left:right_ref]
    test_segment = test[left:right_test]

    pos_left = gapfill_start + left
    pos_right = gapfill_start + right_ref - 1

    hgvs_list = []

    # Deletion
    if ref_segment and not test_segment:
        if return_all:
            # Always include full-range deletion with and without sequence
            if pos_left == pos_right:
                hgvs_list += [
                    f"c.{pos_left}del{ref_segment}",
                    f"c.{pos_left}del",
                    f"c.{pos_left}_{pos_right}del{ref_segment}",
                    f"c.{pos_left}_{pos_right}del"
                ]
            else:
                hgvs_list += [
                    f"c.{pos_left}_{pos_right}del{ref_segment}",
                    f"c.{pos_left}_{pos_right}del"
                ]

            # ✅ Only emit per-base del forms if exactly 1 base deleted
            if len(ref_segment) == 1:
                base = ref_segment
                pos = gapfill_start + left
                hgvs_list += [f"c.{pos}del{base}", f"c.{pos}del"]
        else:
            if pos_left == pos_right:
                hgvs_list = [f"c.{pos_left}del"]
            else:
                hgvs_list = [f"c.{pos_left}_{pos_right}del"]

    # Insertion
    elif not ref_segment and test_segment:
        insertion_call = f"c.{pos_left}_{pos_left + 1}ins{test_segment}"
        dup_candidates = []

        # Search backward for matching sequence to test_segment
        for i in range(max(0, left - len(test_segment)), left + 1):
            candidate = ref[i:i + len(test_segment)]
            if candidate == test_segment:
                dup_start = gapfill_start + i
                dup_end = dup_start + len(test_segment) - 1
                if dup_start == dup_end:
                    dup = f"c.{dup_start}dup{test_segment}"
                    ins = f"c.{dup_start - 1}_{dup_start}ins{test_segment}"
                else:
                    dup = f"c.{dup_start}_{dup_end}dup{test_segment}"
                    ins = f"c.{dup_start - 1}_{dup_end}ins{test_segment}"
                dup_candidates.append((dup, ins))

        if return_all:
            hgvs_list.append(insertion_call)
            for dup, ins in dup_candidates:
                hgvs_list.extend([
                    dup,
                    dup.replace(f"dup{test_segment}", "dup"),
                    ins
                ])
        else:
            if dup_candidates:
                hgvs_list = [dup_candidates[-1][0]]  # most 3′ dup with base
            else:
                hgvs_list = [insertion_call]

    # Delins
    elif ref_segment and test_segment:
        if pos_left == pos_right:
            hgvs_list = [f"c.{pos_left}delins{test_segment}"]
        else:
            hgvs_list = [f"c.{pos_left}_{pos_right}delins{test_segment}"]

    # Deduplicate and sort
    hgvs_list = sorted(set(hgvs_list), key=lambda x: (int(''.join(filter(str.isdigit, x))), x))

    return hgvs_list if return_all else hgvs_list[-1]

def extract_matching_hgvs(row, expected_mutations):
    """Extracts the HGVS change from a row and matches it against expected mutations."""
    if not (isinstance(row['hgvs_change'], (list, np.ndarray))):
        return None
    hgvs = row['hgvs_change']
    gene = row['name'].split(' ')[0]
    comparison_list = expected_mutations.loc[expected_mutations['gene'] == gene, 'HGVSc'].tolist()
    if isinstance(hgvs, list):
        matches = [x for x in hgvs if x in comparison_list]
        if matches:
            return gene + ' ' + matches[0]
    return gene + ' ' + hgvs[0] + '_novel'

def get_likelihood_given_edit_distance(merged_long):
    df = merged_long.copy()
    ## aggregate counts across control samples
    df['expected_gapfill_count'] = df['likelihood_given_wt_edit_dist'] * df['count_of_this_probe']
    # Run proportions_ztest for each row
    df['likelihood_observed_proportion_given_edit_dist'] = df.apply(
        lambda row: 1.0 if (
            row['count_of_this_gapfill'] == row['count_of_this_probe'] and
            row['expected_gapfill_count'] == row['count_of_this_probe']  ## p is high because proportions are the same in sample and control
        ) else (
            proportions_ztest(
                [row['count_of_this_gapfill'], row['expected_gapfill_count']],
                [row['count_of_this_probe'], row['count_of_this_probe']]
            )[1] if row['count_of_this_probe'] > 0 and row['expected_gapfill_count'] > 0 else np.nan
        ),
        axis=1
    )
    df['proportion_greater_than_expected_from_edit_dist'] = np.where(df['expected_gapfill_count'] > 0, (df['count_of_this_gapfill'] / df['count_of_this_probe']) > (df['expected_gapfill_count'] / df['count_of_this_probe']), np.nan)
    df['proportion_greater_than_expected_from_edit_dist'] = df['proportion_greater_than_expected_from_edit_dist'].map({1:1, 0:-1, np.nan: np.nan})
    return df.drop(['expected_gapfill_count'], axis=1)

def get_likelihood_given_control_sample(merged_long):
    df = merged_long.copy()
    control_probe_count_cols = df.columns[df.columns.str.contains('count_of_this_probe_control_')]
    control_gapfill_count_cols = df.columns[df.columns.str.contains('count_of_this_gapfill_control_')]
    ## aggregate counts across control samples
    df['control_probe_count'] = df[control_probe_count_cols].fillna(0).sum(axis=1)
    df['control_gapfill_count'] = df[control_gapfill_count_cols].fillna(0).sum(axis=1)
    # Run proportions_ztest for each row
    df['likelihood_given_wt_control'] = df.apply(
        lambda row: 1.0 if (
            row['count_of_this_gapfill'] == row['count_of_this_probe'] and
            row['control_gapfill_count'] == row['control_probe_count']  # p is high because proportions are the same in sample and control
        ) else (
            1.0 if (
                row['count_of_this_gapfill'] == 0 and
                row['control_gapfill_count'] == 0
            ) else (
                proportions_ztest(
                    [row['count_of_this_gapfill'], row['control_gapfill_count']],
                    [row['count_of_this_probe'], row['control_probe_count']]
                )[1] if row['count_of_this_probe'] > 0 and row['control_probe_count'] > 0 else np.nan
            )
        ),
        axis=1
    )
    df['proportion_greater_than_expected_from_control'] = np.where(df['control_probe_count'] > 0, (df['count_of_this_gapfill'] / df['count_of_this_probe']) > (df['control_gapfill_count'] / df['control_probe_count']), np.nan)
    df['proportion_greater_than_expected_from_control'] = df['proportion_greater_than_expected_from_control'].map({1:1, 0:-1, np.nan: np.nan})
    return df

def get_likelihood_given_all_other_samples(merged_long):
    dfs = []
    for sample in merged_long['sample'].unique():
        ref = merged_long.copy()
        ref = ref.loc[ref['sample'] != sample].groupby(['gapfill','gapfill_start','lhs_probe','rhs_probe']).agg({
            'count_of_this_gapfill': 'sum'}).reset_index()
        probe_count = merged_long.loc[merged_long['sample'] != sample].drop_duplicates(subset=['sample','lhs_probe','rhs_probe']).groupby(['lhs_probe','rhs_probe']).agg({
            'count_of_this_probe': 'sum'}).reset_index()
        df = merged_long.loc[merged_long['sample'] == sample].merge(ref, on=['gapfill','lhs_probe','rhs_probe','gapfill_start'], how='left', suffixes=('', '_ref'))
        df = df.merge(probe_count, on=['lhs_probe','rhs_probe'], how='left', suffixes=('', '_ref'))
        df['count_of_this_gapfill_ref'] = df['count_of_this_gapfill_ref'].fillna(0)
        df['count_of_this_probe_ref'] = df['count_of_this_probe_ref'].fillna(0)
        # Run proportions_ztest for each row
        df['likelihood_given_other_samples'] = df.apply(
            lambda row: 1.0 if (
            row['count_of_this_gapfill'] == row['count_of_this_probe'] and
            row['count_of_this_gapfill_ref'] == row['count_of_this_probe_ref']  # p is high because proportions are the same in sample and control
            ) else (
            1.0 if (
                row['count_of_this_gapfill'] == 0 and
                row['count_of_this_gapfill_ref'] == 0
            ) else (
                proportions_ztest(
                [row['count_of_this_gapfill'], row['count_of_this_gapfill_ref']],
                [row['count_of_this_probe'], row['count_of_this_probe_ref']]
                )[1] if row['count_of_this_probe'] > 0 and row['count_of_this_probe_ref'] > 0 else np.nan
            )
            ),
            axis=1
        )
        df['proportion_greater_than_expected_from_others'] = np.where(df['count_of_this_probe_ref'] > 0, (df['count_of_this_gapfill'] / df['count_of_this_probe']) > (df['count_of_this_gapfill_ref'] / df['count_of_this_probe_ref']), np.nan)
        df['proportion_greater_than_expected_from_others'] = df['proportion_greater_than_expected_from_others'].map({1:1, 0:-1, np.nan: np.nan})
        dfs.append(df)
    merged_long = merged_long.merge(pd.concat(dfs, ignore_index=True)[['gapfill', 'lhs_probe', 'rhs_probe', 'gapfill_start', 'sample', 'count_of_this_probe_ref','count_of_this_gapfill_ref', 'likelihood_given_other_samples', 'proportion_greater_than_expected_from_others']], on=['gapfill', 'lhs_probe', 'rhs_probe','sample','gapfill_start'], how='left')
    return merged_long

def label_control_columns(merged_table, lib, control_idx):
    control_gapfill_count_column = 'count_of_this_gapfill_BC0' + str(control_idx)
    control_probe_count_column = 'count_of_this_probe_BC0' + str(control_idx)
    if control_gapfill_count_column not in merged_table.columns:
        print(f"Warning: {control_gapfill_count_column} not found in merged_table columns.")
    else:
        merged_table['count_of_this_gapfill_control'] = merged_table[control_gapfill_count_column]
        merged_table['count_of_this_probe_control'] = merged_table[control_probe_count_column]
    return merged_table

def make_merge_table(directory, label_control_column = False, lib = '', control_idx=''):
    BCs = []
    merge_columns = ['gapfill', 'gapfill_from_transcriptome','gapfill_start','gap_probe_sequence','likelihood_given_wt_edit_dist','lhs_probe','rhs_probe']
    # Read all tables in the directory
    i = 0
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if directory doesn't exist
    else:
        for file in os.listdir(directory):
            if file.endswith('.csv'):  # Assuming the tables are in CSV format
                table_name = os.path.splitext(file)[0].split('gapfills_')[1]
                BCs.append(table_name)
                current_table = pd.read_csv(os.path.join(directory, file))
                current_table.rename(columns={'likelihood': 'likelihood_given_wt_edit_dist'}, inplace=True)

                ### add empty gap_probe_sequence rows even for gapfills that were not found
                subset_table = current_table.drop_duplicates(subset=['lhs_probe','rhs_probe','gap_probe_sequence']).dropna(subset=['gap_probe_sequence'])
                subset_table.loc[:,'gapfill'] = subset_table['gap_probe_sequence'].copy()
                subset_table.loc[:,'likelihood_given_wt_edit_dist'] = 0
                subset_table.loc[:,'frequency'] = 0
                subset_table.loc[:,'count_of_this_gapfill'] = 0
                current_table = pd.concat([current_table, subset_table]).drop_duplicates(subset=['lhs_probe','rhs_probe','gapfill'])

                columns_to_rename = [col for col in current_table.columns if col not in merge_columns]
                current_table.rename(columns={col: col + '_' + table_name for col in columns_to_rename}, inplace=True)
                if i == 0:
                    merged_table = current_table
                else:
                    merged_table = merged_table.merge(
                    current_table,
                    on=merge_columns,
                    how='outer'
                    )
                i += 1

        merged_table.loc[:,merged_table.columns.str.contains('count_of_this_gapfill')] = merged_table.loc[:,merged_table.columns.str.contains('count_of_this_gapfill')].fillna(0)
        merged_table.loc[:,merged_table.columns.str.contains('frequency')] = merged_table.loc[:,merged_table.columns.str.contains('frequency')].fillna(0)
        
        if label_control_column:
            merged_table = label_control_columns(merged_table,lib, control_idx)

        ## rename columns with lib as suffix
        merged_table = merged_table.rename(
            columns={col: col + '_' + lib for col in merged_table.columns if col not in merge_columns}
        )
        merged_table['gapfill'] = merged_table['gapfill'].fillna('')
        merged_table['gapfill_from_transcriptome'] = merged_table['gapfill_from_transcriptome'].fillna('')
        return merged_table

def sum_probe_counts(merged_table):
    libs = merged_table.columns.str.split('count_of_this_gapfill_').str[1].dropna().unique()
    ### now add total probe counts for gapfills that were not present in that sample
    for BC in libs:
        gapfill_col = f'count_of_this_gapfill_{BC}'
        probe_col = f'count_of_this_probe_{BC}'
        if gapfill_col in merged_table.columns and probe_col in merged_table.columns:
            summed = merged_table.groupby(['lhs_probe','rhs_probe'])[gapfill_col].transform('sum')
            merged_table[probe_col] = summed
            # merged_table[probe_col] = merged_table[probe_col].fillna(summed).infer_objects(copy=False)
    return merged_table

def name_variants_by_gapfill(merged_long,expected_mutations,merge_columns):
    if 'HGVSc' in merged_long.columns:
        merged_long = merged_long.drop(columns=['HGVSc'])
    merged_long['hgvs_change'] = merged_long.apply(
        lambda row: gf_utils.get_hgvs_from_gapfill(
            str(row['gapfill_from_transcriptome']) if pd.notnull(row['gapfill_from_transcriptome']) else '',
            str(row['gapfill']) if pd.notnull(row['gapfill']) else '',
            int(row['gapfill_start']) if (pd.notnull(row['gapfill_start']) and ',' not in row['gapfill_start'] and ']' not in row['gapfill_start']) else '',
            return_all=True, revcomp = True
        ),
        axis=1
    )
    merged_long['HGVSc'] = merged_long.apply(lambda row: gf_utils.extract_matching_hgvs(row, expected_mutations), axis=1)
    merged_long = merged_long.drop(['hgvs_change'],axis=1)
    print('len merged long',len(merged_long))
    merged_long['HGVSc'] = merged_long['HGVSc'].str.replace('c.=_novel','wildtype')
    print('len merged long after merge',len(merged_long))
    return merged_long

def get_feature_set(mutated_df, sample, likelihood_column = 'signed_log_likelihood_given_wt_control', min_count=10, min_log_likelihood=-10, min_frequency = 0.01, min_ratio=0.3):
    """
    Get feature set for a specific sample based on mutation data.
    Parameters:
    - mutated_df: DataFrame containing mutation data.
    - sample: Sample identifier to filter the DataFrame.
    - min_count: Minimum count of gapfill occurrences to consider.
    - min_log_likelihood: Minimum signed log likelihood to consider.
    - min_ratio: Minimum ratio of each passing mutant gapfill to most abundant mutant gapfill. (e.g. ratio of AC to AAC for JAK2 is <0.2 so only AAC is considered a feature)
    Returns:
    - sub_mut_df: Filtered DataFrame containing relevant features.
    """
    sub_mut_df = mutated_df.loc[(mutated_df['count_of_this_gapfill'] > min_count) & (mutated_df[likelihood_column] < min_log_likelihood) & (mutated_df['sample'] == sample)][['name','HGVSc','gapfill','gapfill_from_transcriptome','frequency','expected_frequency_from_bulk','lhs_probe','rhs_probe','count_of_this_gapfill','count_of_this_probe','signed_log_likelihood_given_wt_control','signed_log_likelihood_given_other_samples','signed_log_likelihood_given_wt_edit_dist','sample','likelihood_given_wt_edit_dist','original_name']]
    sub_mut_df = sub_mut_df.merge(sub_mut_df.groupby(['lhs_probe','rhs_probe','sample']).agg({'count_of_this_gapfill':'max'}).reset_index(), on=['lhs_probe','rhs_probe','sample'], suffixes=('', '_max'))
    sub_mut_df['ratio'] = sub_mut_df['count_of_this_gapfill'] / sub_mut_df['count_of_this_gapfill_max']
    sub_mut_df = sub_mut_df.loc[sub_mut_df['ratio'] > min_ratio]
    sub_mut_df = sub_mut_df.loc[sub_mut_df['frequency'] > min_frequency]
    return sub_mut_df.drop(['count_of_this_gapfill_max','ratio'], axis=1).sort_values('frequency',ascending=False)

def assign_genotypes(adata, min_p=0.6, min_counts=1, max_p=1.0):
    variants = []
    cols = adata.obsm['genotypes'].columns
    for col in cols:
        if 'high_confidence_counts' in col:
            variant = col.replace('_high_confidence_counts', '')
            variants.append(variant)
    for variant in variants:
        variant_cols = [col for col in cols if col.startswith(variant)]
        variant_cols = [col for col in variant_cols if col.endswith('_mutated') or col.endswith('_wt') or col.endswith('_heterozygous')]
        max_probs = adata.obsm['genotypes'][variant_cols].dropna().max(axis=1)
        n_counts = adata.obsm['genotypes'][variant + '_high_confidence_counts']
        genotypes = adata.obsm['genotypes'][variant_cols].dropna().idxmax(axis=1)
        genotypes = genotypes[(max_probs > min_p) & (n_counts >= min_counts) & (max_probs <= max_p)]
        genotypes = genotypes.str.replace(variant + '_', '')
        if 'genotype_call' not in adata.obsm:
            adata.obsm['genotype_call'] = pd.DataFrame(index=adata.obs_names)
        adata.obsm['genotype_call'][variant] = genotypes
