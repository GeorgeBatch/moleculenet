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

In this section I outline the set-up steps required to start reproducing my results. It covers the following stages:

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

If you decided not to clone the repository from GitHub but still want to reproduce the results, choose a directory, where you will store the data and code. Organise your directory as follows:

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

[Conda](https://docs.conda.io/en/latest/) environments make managing Python library dependences and reproducing research much easier. Another reason why we use conda us that some packages, *e.g.* RDKit: Open-Source Cheminformatics Software, are not available via `pip install`.

## Data preparation

This section covers two data preparation stages: standardising input files and producing the features.

### Standardise Names

To automate the process of working with three different datasets (ESOL, FreeSolv, and Lipiphilicity) we standardise the column names from the original CSV files and store the results in the new CSV files.

We need to get hold of ID/Name, SMILES string representation, and measured label value for each of the compounds in all of the three datasets. To do this, run the following commands in the `~/scripts/` directory:

```
>>> python get_original_id_smiles_labels_lipophilicity.py
>>> python get_original_id_smiles_labels_esol.py
>>> python get_original_id_smiles_labels_freesolv.py
```

The resulting files are saved in the `~/data/` directory:
- `esol_original_IdSmilesLabels.csv`, `esol_original_extra_features.csv`
- `freesolv_original_IdSmilesLabels.csv`
- `lipophilicity_original_IdSmilesLabels.csv`

**Note:** the original file for the ESOL dataset also contained extra features which we also saved here.

### Compute and Store Features

We show how to produce the features and store them in CSV files.

From the SMILES string representations of the molecules for all three datasets compute Extended-Connectivity Fingerprints and RDKit Molecular Descriptors to use them as features. We do it at the very beginning and never worry about it in the future.

**Note**, we produce four different versions of extended-connectivity fingerprints:
- ECFP_4 hashed with 1024 bits
- ECFP_6 hashed with 1024 bits
- ECFP_4 hashed with 2048 bits
- ECFP_6 hashed with 2048 bits

To compute and record the features run the corresponding commands in the `scripts` directory:

#### ECFP features
```
>>> python get_all_fingerprints_for_all_datasets.py
```

#### RDKit features

```
>>> python get_rdkit_descriptors_for_all_datasets.py
```
