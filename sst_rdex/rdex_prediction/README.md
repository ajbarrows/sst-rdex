## RDEX-ABCD Model Prediction Using Task fMRI

## Motivating question

Does a formal model of behavior on the ABCD Study stop-signal task (SST) relate to individual differences in brain function during the task?

## Overview

This module uses the Python package [`BPt`](https://doi.org/10.1093/bioinformatics/btaa974) to fit a series of predictive models
exploring the association between individual differences in brain function during the SST and a formal model of behavior on the task.

Parameters, including file paths, are defined in `parameters.yaml`.

## Data

Data are prepared using `../data_management/pipelines/assemble_mri.py` and `../data_management/pipelines/assemble_behavioral.py`.

Then, a `BPt` dataset is created using `./pipelines/make_model_dataset.py`:

```bash
$ cd pipelines
$ python make_model_dataset.py
```


## Models

### Analysis 1

Models can be fit using a command-line interface from `./pipelines/run_model.py`:

```bash
$ cd pipelines
$ python run_model.py -h
```

```bash
usage: run_model.py [-h] [--dataset DATASET] [--scopes SCOPES] [--fpath FPATH] [--n_cores N_CORES] [--random_state RANDOM_STATE] model condition

Run RDEX prediction pipeline

positional arguments:
  model                 Options: ridge, lasso, elastic
  condition             Options: all, correct_go, correct_stop, incorrect_go, incorrect_stop

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to dataset
  --scopes SCOPES       Path to scopes
  --fpath FPATH         Path to save results
  --n_cores N_CORES     Number of cores to use. Default: -1 (all)
  --random_state RANDOM_STATE
                        Random state.

```

## SLURM

Additional functionality is available to run models on a SLURM cluster using `./slurm/00_generate_slurm_scripts.py`:

```bash
$ cd slurm
$ python 00_generate_slurm_scripts.py -h
```

```bash
usage: 00_generate_slurm_scripts.py [-h] [--models MODELS] [--conditions CONDITIONS] [--launch | --no-launch] [--directory DIRECTORY]

Generate slurm scripts for RDEX prediction pipeline

options:
  -h, --help            show this help message and exit
  --models MODELS       Options: ridge, lasso, elastic
  --conditions CONDITIONS
                        Options: all, correct_go, correct_stop, incorrect_go, incorrect_stop
  --launch, --no-launch
                        Launch slurm scripts (default: False)
  --directory DIRECTORY
                        Directory to save slurm scripts. Default: .
```


## Analysis 2

To determine the composition of empirically-derrived SSRT as it relates to RDEX-ABCD model parameters,
we fit a model using RDEX-ABCD model parameters as predictors of SSRT.

See `./experiments/crosspredict.ipynb` for details.


## Gathering Results

Running

```bash
$ cd pipelines
$ python produce_model_results.py
```
