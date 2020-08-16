# Standard imports
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global variables to be used in scripts and notebooks

# titles (everywhere)
datasets_to_titles = {
    'freesolv': 'solvation energy'.title(),
    'esol': 'log solubility'.title(),
    'lipophilicity': 'octanol/water distribution coefficient'.title()
}

# units (CIs)
datasets_to_units = {
    'freesolv': '(kcal/mol)',
    'esol': '(mol/litre)',
    'lipophilicity': '(logD)'
}

# labels (heatmaps)
metrics_to_labels = {
    'RMSE': 'root-mean-square error',
    'MAE': 'mean absolute error',
    'R^2': 'R^2 (coefficient of determination)',
    'pearson_r': 'Pearson correlation'
}

# -----------------------------------------------------------------------------
# Global plotting options

FIGSIZE = (6, 6)

PLOTS_DIR = '../figures'

# -----------------------------------------------------------------------------
# Plotting functions
