<a href="https://colab.research.google.com/github/GeorgeBatch/learning-ligand/blob/master/readme.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Physical Chemistry datasets

This repository aims to show the process of exploring how varying **feature sets**, **train-validation-test splits**, and **models** changes the prediction performance on the regression tasks for [Physical Chemistry datasets](http://moleculenet.ai/datasets-1).

# Set-up

To make the reproduction process as simple as possible, clone this repository (`moleculenet`) to your local machine.

## Directory

Choose a directory, where you will store the data and the code to reproduce the results. Organize the directory in it as shown below.

All the files in the `data` folder are either downloaded from the [Moleculenet web page](http://moleculenet.ai/). You can download the data from the [datasets page](http://moleculenet.ai/datasets-1).

Populate the data directory with the following files:

- from the FreeSolv folder: `SAMPL.csv`, `FreeSolv_README`
- from the ESOL folder: `delaney-processed.csv`, `ESOL_README`
- from the lipophilicity folder: `Lipophilicity.csv`, `Lipo_README`

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
  ---- environment.yml
  |
  ...
```

Rename the csv files as follows:

- `SAMPL.csv` $\to$ `freesolv_original.csv`
- `delaney-processed.csv` $\to$ `esol_original.csv`
- `Lipophilicity.csv` $\to$ `lipophilicity_original.csv`

## Environment

In the `moleculenet` directory create project envoronment from the environment.yml file using:
```
>>> conda env create -f environment.yml
```

Environment's name is `batch-msc`, and we activate it using:
```
>>> conda activate batch-msc
```

# Data preparation

## Standardise the file names and column names

We will need to get hold of IDs/Names, Smiles, and measured target values for all 3 datasets. We will produce 3 csv files with the following coloumns to standardise the future work:

Run the following commands to get them in the `scripts` directory:

```
>>> python get_original_id_smile_target_lipophilicity.py 
>>> python get_original_id_smile_target_esol.py 
>>> python get_original_id_smile_target_freesolv.py 
```

The output files are in the `../data/` directory:
- `esol_original_IdSmileTarget.csv`, `esol_original_extra_features.csv`
- `freesolv_original_IdSmileTarget.csv`
- `lipophilicity_original_IdSmileTarget.csv`

**Note:** data for ESOL dataset also contained extra features which we also saved here.

## Create .csv files with ECFP-4, ECFP-6 fingerprints for all datasets*smile-string combinations

We produce the files with ECFP-4, ECFP-6 fingerprints for all datasets*smile-string combinations. We do it once and never worry anout it in the future.

Run the following command `scripts` directory:
```
>>> python get_all_fingerprints_for_all_datasets.py
```

## Create .csv files with RDKit molecular descriptors for all datasets*smile-string combinations

We produce the files with ECFP-4, ECFP-6 fingerprints for all datasets*smile-string combinations. We do it once and never worry anout it in the future.

Run the following command `scripts` directory:
```
>>> python get_rdkit_descriptors_for_all_datasets.py
```


