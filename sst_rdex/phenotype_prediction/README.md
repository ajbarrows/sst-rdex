## Predict Behavioral Phenotypes using RDEX-ABCD Model

## Motivating question

Do RDEX-ABCD model parameter estimates correlate with phenotypes of interest?


## Overview

Predict ABCD Study tabular phenotypes (i.e., NIH Toolbox, CBCL, UPPS, and BISBAS)
in a regularized regression using
-   RDEX-ABCD parameters alone
-   Empirical measures (i.e., correct go response time, empirically-derrived SSRT) alone
-   RDEX-ABCD parameters + Empirical measures

## Models

Fit models using a command-line interface:

```bash
$ cd pipelines
$ python predict_phenotype.py
```

### SLURM

Models were fit using HPC and a SLURM scheduler:

```bash
$ cd slurm
$ sbatch predict_phenotypes.sh
```

## Gathering results

```bash
$ cd pipelines
$ python produce_model_results.py
```
