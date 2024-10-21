# Estimating Uncertainty in Machine Learning Models for Drug Discovery

## Project Details

- **Title:** Estimating Uncertainty in Machine Learning Models for Drug Discovery
- **Type:** MSc dissertation
- **Author:** George Batchkala, https://www.linkedin.com/in/george-batchkala/
- **Supervisor:** Professor Garrett M. Morris, garrett.morris@dtc.ox.ac.uk
- **Institution:** University of Oxford
- **Department:** Department of Statistics, 24-29 St Giles', Oxford, OX1 3LB
- **Project's dates:** June 1st, 2020 - September 14th, 2020
- **Data:** MoleculeNet, Physical Chemistry Datasets (http://moleculenet.ai/datasets-1)
- **GitHub repository:** https://github.com/GeorgeBatch/moleculenet

----

This repository contains all code, results, and plots I produced while completing my MSc dissertation. The pdf file with the full dissertation will be uploaded after it gets marked and I officially complete my degree.

## Abstract

"*My model says that I had just found an ultimate drug. Can I trust it?*"

In this work, I explore ways of quantifying the confidence of machine learning models used in drug discovery. In order to do this, I start with exploring methods to predict physicochemical properties of drugs and drug-like molecules crucial to drug discovery. I first attempt to reproduce and improve upon a subset of results to do with a drug's solubility in water, taken from a popular benchmark set called "MoleculeNet". Using XGBoost, which in the era of Deep Neural Networks, is already classified as a "conventional" machine learning method, I show that I am able to achieve state-of-the-art results. After that, I explore Gaussian Processes and Infinitesimal Jackknife for Random Forests and their associated uncertainty estimates. Finally, I attempt to understand whether the confidence of a model's prediction can be used to answer a similar but more general question: "*How do we know when to trust our models?*" The answer depends on the model. We can trust Gaussian Processes when they are confident, but the confidence estimates from Random Forests do not give us any assurance.

## Related work

This work is mostly based of four papers:
- "MoleculeNet: A Benchmark for Molecular Machine Learning" by [Wu *et al.*](https://pubs.rsc.org/en/content/articlelanding/2018/SC/C7SC02664A#!divAbstract);
- "Learning From the Ligand: Using Ligand-Based Features to Improve Binding Affinity Prediction" by [Boyles *et al.*](https://academic.oup.com/bioinformatics/article-abstract/36/3/758/5554651?redirectedFrom=fulltext);
- "The Photoswitch Dataset: A Molecular Machine Learning Benchmark for the Advancement of Synthetic Chemistry" by [Thawani *et al.*](https://chemrxiv.org/articles/preprint/The_Photoswitch_Dataset_A_Molecular_Machine_Learning_Benchmark_for_the_Advancement_of_Synthetic_Chemistry/12609899); and
- "Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife" by [Wager *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4286302/).

## Aims

In this dissertation I aim to achieve three primary goals:

1. **Reproduce** a subset of solubility-related prediction results from the MoleculeNet benchmarking paper;
2. **Improve** upon the reproduced results; and
3. Use **uncertainty estimation** methods with the best-performing models to get single prediction uncertainty estimates to evaluate and compare these methods.

## Data

I used the [MoleculeNet dataset](http://moleculenet.ai/datasets-1) which accompanies the [MoleculeNet benchmarking paper](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a#!divAbstract), and in particular, I focused on the Physical Chemistry datasets: [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x), [FreeSolv](https://link.springer.com/article/10.1007/s10822-014-9747-x), and [Lipophilicity](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2718). The MoleculeNet datasets are widely used to validate machine learning models used to estimate a particular property directly from small molecules including drug-like compounds.

The Physical Chemistry datasets can be downloaded from [MoleculeNet benchmark dataset collection](http://moleculenet.ai/datasets-1).

## Models

I use the following four models for the regression task of physicochemical property prediction:

- [Kernel Ridge Regression](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf)
- [eXtreme Gradient Boosting (XGBoost)](https://dl.acm.org/doi/10.1145/2939672.2939785)
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
- [Gaussian Processes](http://www.gaussianprocess.org/gpml/)

## Obtaining Confidence Intervals

I obtained per-prediction confidence intervals with:

- Gaussian Processes ([notes, chapter 7, section 7.2](https://github.com/ywteh/advml2020/blob/master/notes.pdf))
- Bias-corrected Infinitesimal Jackknife estimate for Random Forests ([paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4286302/))

## Implementation

All the data preparation, experiments, and visualisations were done in Python.

To convert molecules from their [SMILES](https://pubs.acs.org/doi/abs/10.1021/ci00057a005) string representations to either Molecular Descriptors or [Extended-Connectivity Fingerprints](https://pubs.acs.org/doi/10.1021/ci100050t), I used the open-source cheminformatics software, [RDKit](https://www.rdkit.org/) ([GitHub](https://github.com/rdkit/rdkit)).

[Wu *et al.*](https://pubs.rsc.org/en/content/articlelanding/2018/SC/C7SC02664A#!divAbstract) suggest to use their Python library, [DeepChem](https://www.deepchem.io/) ([GitHub](https://github.com/deepchem/deepchem)), to reproduce the results. We decided not to use it, since the user API only gives high-level access to the user, while I wanted to have more control of the implementation. To have comparable results, I decided to use the tools which the DeepChem library is built on.

For most of the machine learning pipeline, I used [Scikit-Learn](https://www.jmlr.org/papers/v12/pedregosa11a.html) ([GitHub](https://github.com/scikit-learn/scikit-learn)) for preprocessing, splitting, modelling, prediction, and validation. To obtain the confidence intervals for Random Forests, I used the [forestci](https://joss.theoj.org/papers/10.21105/joss.00124) ([GitHub](https://github.com/scikit-learn-contrib/forest-confidence-interval)) extension for Scikit-Learn. The implementation of a custom Tanimoto (Jaccard) kernel for Gaussian Process Regression and all the following GP experiments were performed with [GPflow](http://jmlr.org/papers/v18/16-537.html) ([GitHub](https://github.com/GPflow/GPflow)).

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

**Note:** the original file for the ESOL dataset also contained extra features which we also save here.

### Compute and Store Features

We show how to produce the features and store them in CSV files.

From the SMILES string representations of the molecules for all three datasets compute Extended-Connectivity Fingerprints and RDKit Molecular Descriptors to use them as features. We do it at the very beginning and never worry about it in the future.

**Note**, we produce four different versions of extended-connectivity fingerprints:
- ECFP_4 hashed with 1024 bits
- ECFP_6 hashed with 1024 bits
- ECFP_4 hashed with 2048 bits
- ECFP_6 hashed with 2048 bits

To compute and record the features run the corresponding commands in the `~/scripts/` directory:

#### ECFP features
```
>>> python get_all_fingerprints_for_all_datasets.py
```

#### RDKit features

```
>>> python get_rdkit_descriptors_for_all_datasets.py
```
