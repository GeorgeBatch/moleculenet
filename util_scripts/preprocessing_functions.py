# standard imports
import numpy as np
import pandas as pd

# pearson correlation
from scipy.stats import pearsonr


def list_highly_correlated(df_features, targets, threshold=0.8):
    """
    List column names of the dataframe of features which are highly correlated
    to the target (absolute value of the correlation is greater than the threshold).

    Parameters
    ----------
    df_features : (n, p) pandas.core.frame.DataFrame of p features
                  Input array.
    targets     : (n,) pandas.core.series.Series of targets
                  Input array.
    threshold   : float in [0, 1] above which we consider a feature highly correlated

    Returns
    -------
    cols_to_remove : list of column names from df_features, which are highly correlated
                     to the target

    """
    # check bounds for abs(correlation) threshold
    assert 0 <= threshold <= 1

    # df_features and targets should have the same length
    assert df_features.shape[0] == targets.shape[0]
    #print('Original shapes:                 ', df_features.shape, targets.shape)

    # remove na rows
    X = df_features.dropna(axis=0)
    y = targets[X.index]
    #print('Removed NA rows, shapes:         ', X.shape, y.shape)

    # remove zero-variace columns
    zero_std = X.std() < 1e-5
    zero_std_cols = X.columns[zero_std]
    X = X.drop(zero_std_cols, axis=1)
    #print('Removed zero-var columns, shapes:', X.shape, y.shape)

    # record highly correlated features
    cols_to_remove = []
    for name in X.columns:
        # print(name, np.abs(pearsonr(X[name], y)[0]))
        if np.abs(pearsonr(X[name], y)[0]) > threshold:
            cols_to_remove.append(name)

    print(f'\nFound {len(cols_to_remove)} highly-correlated feature(s):')
    print(cols_to_remove)
    return cols_to_remove
