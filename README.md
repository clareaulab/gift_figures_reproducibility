# Reproducing GIFT custom analyses and figures 

## Setup

### Set large data directory
Set large data directory to directory containing large data input files. All files can be recreated from GEO files, but intermediate file repository coming soon.
(large_data_dir is set in 1_figure_CL_proof_of_concept/code/utils_00.py)

### Create and install required packages (all analyses except Visium)

```bash
conda create -n gift_paper python=3.11.10
conda activate gift_paper
pip install pandas
pip install scanpy
conda install numpy=1.25.2
conda install -c conda-forge biopython
pip install gseapy
pip install picasso_phylo
pip install phenograph
pip install openpyxl
pip install scikit-misc
pip install scikit-image
```

### Install required packages for Visium

```bash
conda create -n gift_spatial python=3.12
conda activate gift_spatial
pip install giftwrap-sc[spatial]
pip install spatialdata_io
pip install bin2cell
pip install hdf5plugin
pip install buencolors
```

## Probe design

The repository for designing custom genotyping probes is [available here](https://github.com/clareaulab/probeset_design_pipeline). 

## Custom GIFT library parsing
Custom software for analyzing GIFT gapfill analyses is available at our [giftwrap repository](https://github.com/clareaulab/giftwrap), which can be installed via [pypi](https://pypi.org/project/giftwrap-sc/)

Full datasets are available as part of our [GEO submission GSE319999](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE319999). 

<br><br>
