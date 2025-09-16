# SST-RDEX Data Management

- Filepaths and other parameters are defined in `parameters.yaml`
- Many utilities rely on [`abcd-tools`](https://github.com/ajbarrows/abcd-tools.git). Follow installation instructions.

## Behavioral data (including prediction targets)

Data available from official ABCD Study tabular release. RDEX model parameter estimates
available from the authors.

```bash
cd pipelines
python assemble_behavioral.py
```

## Task fMRI

Vertexwise (tabular) task fMRI data come from `abcd-sync` and are converted from
Matlab-encoded proprietary files to open-source Parquet files using a Matlab script.
Good luck.

**Note** Filepaths for Matlab code are not contained in `parameters.yaml`.

```matlab
% Convert .mat files to parquet format
% Run from MATLAB directory
matlab/mat2parquet.m
```

Then, hemispheres are concatenated and Beta estimates are averaged across runs, weighted
by censored frames (i.e., degrees of freedom). Averaged Betas are limited to
the intersection of subjects with behavioral data. Imaging sessions which failed
DAIRC quality control (i.e., `imgincl_sst_include') are excluded.

Finally, data are saved as Parquet files.

```bash
cd pipelines
python assemble_mri.py
```

## Phenotype

Load mental health and substance use phenotypes from the ABCD Study tabular release.

```bash
cd pipelines
python assemble_phenotype.py
```
