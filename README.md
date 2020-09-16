# Estimating Uncertainty in Machine Learning Models for Drug Discovery

A dissertation submitted in partial fulfilment of the degree of Master of Science in Statistical Science. Department of Statistics, 24--29 St Giles', Oxford, OX1 3LB.
----

This repository contains all code, results, and plots I produced while completing my MSc dissertation. The pdf file with the full dissertation will be uploaded after it gets marked and I officially complete my degree.

## Abstract

"*My model says that I had just found an ultimate drug. Can I trust it?*"

In this work, I explore ways of quantifying the confidence of machine learning models used in drug discovery. In order to do this, I start with exploring methods to predict physicochemical properties of drugs and drug-like molecules crucial to drug discovery. I first attempt to reproduce and improve upon a subset of results to do with a drug's solubility in water, taken from a popular benchmark set called "MoleculeNet". Using XGBoost, which in the era of Deep Neural Networks, is already classified as a "conventional" machine learning method, I show that I am able to achieve state-of-the-art results. After that, I explore Gaussian Processes and Infinitesimal Jackknife for Random Forests and their associated uncertainty estimates. Finally, I attempt to understand whether the confidence of a model's prediction can be used to answer a similar but more general question: "*How do we know when to trust our models?*" The answer depends on the model. We can trust Gaussian Processes when they are confident, but the confidence estimates from Random Forests do not give us any assurance.

## Data

I used the [MoleculeNet dataset](http://moleculenet.ai/datasets-1) which accompanies the [MoleculeNet benchmarking paper](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract), and in particular, I focused on the Physical Chemistry datasets: [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x), [FreeSolv](https://link.springer.com/article/10.1007/s10822-014-9747-x), and [Lipophilicity](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2718). The MoleculeNet datasets are widely used to validate machine learning models used to estimate a particular property directly from small molecules including drug-like compounds.

The Physical Chemistry datasets can be downloaded from [MoleculeNet benchmark dataset collection](http://moleculenet.ai/datasets-1).

## Models



## Obtaining Confidence Intervals



# Set-up

This section outlines the set-up steps needed to start reproducing my results. It covers the following stages:

1. Directory set-up;
2. Creating an environment with [conda](https://docs.conda.io/en/latest/);
3. Data preparation; and
4. Creation of features.

## Directory

This section tells how to set up your directory via `git clone` or manually.

### Git clone

To make the reproduction process as simple as possible, clone this repository to your local machine. To do this, run the following command in your terminal/command prompt:

```
>>> git clone https://github.com/GeorgeBatch/moleculenet.git
```

### Manual directory set-up

If you decided not to clone the repository from GitHub but still want to reproduce the results, choose a directory, where you will store the data and code. Organise your directory as shown below.

```
- moleculenet
  |
  ---- data
  |
  ---- figures
  |
  ---- notebooks
  |
  ---- scripts
  |
  ---- results
  |
  ---- environment.yml
```

**Download** the Physical Chemistry datasets from the [MoleculeNet datasets page](http://moleculenet.ai/datasets-1).

Populate the `~/data/` directory with these six files:

- from the FreeSolv folder: `SAMPL.csv`, `FreeSolv_README`
- from the ESOL folder: `delaney-processed.csv`, `ESOL_README`
- from the lipophilicity folder: `Lipophilicity.csv`, `Lipo_README`

**Rename** the CSV files as follows:

- `SAMPL.csv` to `freesolv_original.csv`
- `delaney-processed.csv` to `esol_original.csv`
- `Lipophilicity.csv` to `lipophilicity_original.csv`

## Environment

In the root (`moleculenet`) directory create a project environment from the `environment.yml` file using:

```
>>> conda env create -f environment.yml
```

Environment's name is `batch-msc`, and we activate it using:
```
>>> conda activate batch-msc
```

Conda environments make managing Python library dependences and reproducing research much easier. Another reason why we use conda us that some package, *e.g.* RDKit: Open-Source Cheminformatics Software, are not available via `pip install`.

## Data preparation

### Standardise Names

To automate the process of working with three different datasets we standardise their file names and column names.

We need to get hold of IDs/Names, SMILES, and measured label values for all three datasets. We produce three CSV files with the following columns.

Run the following commands to get them in the `~/scripts/` directory:

```
>>> python get_original_id_smiles_labels_lipophilicity.py
>>> python get_original_id_smiles_labels_esol.py
>>> python get_original_id_smiles_labels_freesolv.py
```

The output files are in the `~/data/` directory:
- `esol_original_IdSmilesLabels.csv`, `esol_original_extra_features.csv`
- `freesolv_original_IdSmilesLabels.csv`
- `lipophilicity_original_IdSmilesLabels.csv`

**Note:** data for ESOL dataset also contained extra features which we also saved here.

### Compute and Store Features

Here we show how to produce the features and store them in .csv files with four different versions of extended-connectivity fingerprints ({ECFP_4, ECFP_6} * {1024 bits, 2048 bits}) and RDKit molecular descriptors for all datasets.

We compute the extended-connectivity fingerprints to use them as features from the SMILES string representations of the molecules from all three datasets at the very beginning and never worry about it in the future.

To compute and record the features run the corresponding commands in the `scripts` directory:

#### ECFP features
```
>>> python get_all_fingerprints_for_all_datasets.py
```

#### RDKit features

```
>>> python get_rdkit_descriptors_for_all_datasets.py
```
