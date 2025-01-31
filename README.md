# RDEX-ABCD Brain-Behavior Relationships

Manuscript in preparation.

## Installation

To reproduce the results from this project, first create a `conda` environment (I prefer using `mamba`):

```bash
mamba env update -n sst-rdex --file environment.yml
conda activate sst-rdex
```

You'll also need `abcd-tools`, a collection of utilities for working with ABCD Study data
(make sure your `sst-rdex` environment is active):

```bash
git clone git@github.com:ajbarrows/abcd-tools
cd abcd-tools
python -m pip install -e .
```

## Analysis

Follow the instructions for

- [`data_management`](./data_management/)
- [`rdex_prediction`](./rdex_prediction/)
- [`phenotype_prediction`](./phenotype_prediction/)


Please feel free to get in touch if you have any questions.
