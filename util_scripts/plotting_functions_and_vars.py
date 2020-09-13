# Standard imports
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Global variables to be used in scripts and notebooks

# titles (everywhere)
datasets_to_titles = {
    'freesolv': 'solvation free energy'.title(),
    'esol': 'log solubility'.title(),
    'lipophilicity': 'octanol/water distribution coefficient'.title()
}

# units (CIs)
datasets_to_units = {
    'freesolv': '(kcal/mol)',
    'esol': '', # no units since on log-scale'(mol/litre)',
    'lipophilicity': '(logD)'
}

# labels (heatmaps)
metrics_to_labels = {
    'RMSE': 'root-mean-square error',
    'MAE': 'mean absolute error',
    'R^2': 'R^2 (coefficient of determination)',
    'pearson_r': 'Pearson correlation'
}

datasets_to_rounding_precision = {
    'freesolv': 2,
    'esol': 2,
    'lipophilicity': 3
}

# -----------------------------------------------------------------------------
# Global plotting options

PLOTS_DIR = '../figures'

DPI = 300
FIGSIZE_CI = (4, 4)
FIGSIZE_HEATMAP = (8, 4)

# -----------------------------------------------------------------------------
# Plotting functions

def plot_algorithm_dataset_comparison_heatmap(df, dataset, metric, figsize=FIGSIZE_HEATMAP):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    # heatmap
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.3g',
                cbar_kws={'label': f"{metrics_to_labels[metric]} {datasets_to_units[dataset]}"},
                ax=ax)

    # horizontal lines to separate the heatmap
    ax.hlines([1, 2, 3], *ax.get_xlim(), linestyle=':', linewidth=2)

    # title and labels
    ax.set_ylabel('Algorithm')
    ax.set_xlabel('Features Used')
    ax.set_title(datasets_to_titles[dataset])

    return fig, ax
