## Import modules

### Standard imports

import json
import pickle

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# metrics
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import jaccard_score # Tanimoto

### Custom imports

sys.path.insert(0, '..')

# plotting
from util_scripts.plotting_functions_and_vars import FIGSIZE_CI, FIGSIZE_HEATMAP, DPI, PLOTS_DIR
from util_scripts.plotting_functions_and_vars import datasets_to_titles, datasets_to_units, metrics_to_labels
from util_scripts.plotting_functions_and_vars import plot_algorithm_dataset_comparison_heatmap


from util_scripts.plotting_functions_and_vars import datasets_to_rounding_precision

sys.path.insert(0, './scripts')

## Set plotting style

plt.style.use('fivethirtyeight')

plt.rcParams['axes.facecolor']='w'
#plt.rcParams['axes.linewidth']=1
plt.rcParams['axes.edgecolor']='w'
plt.rcParams['figure.facecolor']='w'
plt.rcParams['savefig.facecolor']='w'
#plt.rcParams['grid.color']='white'


# ----------------------------------------------------------------------------
# Set constants

smile_type = 'original'
assert smile_type in ['original', 'protonated']

grid_search_type = 'extended'
assert grid_search_type in ['reproducing', 'extended']




# metric
metric = 'RMSE'
assert metric in metrics_to_labels
if metric in ['RMSE', 'MAE']:
    pass
else:
    # no units
    datasets_to_units = {'freesolv': '', 'esol': '', 'lipophilicity': ''}


# to print on top of the plots
models_to_title_additions = {
    'rf': 'Random Forests',
    'gp': 'Gaussian Processes'
}

dataset_to_num_cis = {
    'freesolv': 50,
    'esol': 30,
    'lipophilicity': 10
}


# ----------------------------------------------------------------------------
# main loop

for dataset, cf in [('freesolv', 'full'), ('esol', 'full'), ('esol', 'reduced'), ('lipophilicity', 'full')]:
    assert dataset in ['freesolv', 'esol', 'lipophilicity']
    assert cf in ['full', 'reduced']

    # report precision
    rp = datasets_to_rounding_precision[dataset]

    for model in ['rf', 'gp']:
        assert model in ['rf', 'gp']


        print(dataset, cf, model)


        # read the results for best combinations
        df_true = pd.read_csv(f'../results/{dataset}_{smile_type}_{grid_search_type}_{cf}_multiple_ci_runs_true_{model}.csv')
        df_pred = pd.read_csv(f'../results/{dataset}_{smile_type}_{grid_search_type}_{cf}_multiple_ci_runs_pred_{model}.csv')
        df_std = pd.read_csv(f'../results/{dataset}_{smile_type}_{grid_search_type}_{cf}_multiple_ci_runs_std_{model}.csv')


        # --------------------------------------------------------------------
        # one run

        # get values
        y_test = df_true.iloc[:, 0]
        y_test_pred = df_pred.iloc[:, 0]
        y_test_std = df_std.iloc[:, 0]

        # calculate everything
        upper = y_test_pred + 1.96 * y_test_std
        lower = y_test_pred - 1.96 * y_test_std
        CIs_df = pd.DataFrame(
            {'y_test': y_test,
             'y_test_pred': y_test_pred,
             'y_test_std': y_test_std,
             'lower': lower,
             'upper': upper,
             'sq_error': (y_test - y_test_pred) ** 2
             }
        )
        CIs_df = CIs_df.sort_values(by='y_test_std', ascending=True)
        CIs_df['cumul_sq_error'] = CIs_df['sq_error'].cumsum()
        CIs_df['cumul_mse'] = CIs_df['cumul_sq_error'].values / np.arange(1, CIs_df.shape[0]+1)
        CIs_df['cumul_rmse'] = np.sqrt(CIs_df['cumul_mse'])

        # --------------------------------------------------------------------
        # SCATTERPLOT: test observations with sdt values as colours
        #
        # sort values used for x axis
        CIs_df = CIs_df.sort_values(by='y_test')

        # Plot error bars for predicted quantity using unbiased variance
        plt.figure(figsize=FIGSIZE_CI)

        plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', label='perfect prediction')
        plt.scatter(x=CIs_df.y_test, y=CIs_df.y_test_pred, c=CIs_df.y_test_std, s=100, label='test data points')

        plt.xlabel(f'Measured {datasets_to_units[dataset]}')
        plt.ylabel(f'Predicted {datasets_to_units[dataset]}')
        plt.title(f'{models_to_title_additions[model]}')

        # add colourbar and legend
        plt.colorbar(label=f'estimated st.d. {datasets_to_units[dataset]}')
        plt.legend()

        plt.savefig(f'{PLOTS_DIR}/ci_plots/predicted_vs_measured_scatter_colour_conf_{dataset}_{cf}_{model}.png', dpi=DPI, bbox_inches='tight')
        plt.close()

        # --------------------------------------------------------------------
        # ## Confidence plot (RMSE vs Prcentile)
        #
        # sort values used for x axis
        CIs_df = CIs_df.sort_values(by='y_test_std', ascending=True)

        # set size
        plt.figure(figsize=FIGSIZE_CI)

        confidence_percentiles = np.arange(1e-14, 100, 100/len(y_test))
        flipped_cumul_rmse = CIs_df['cumul_rmse'].values[::-1]

        plt.plot(confidence_percentiles, flipped_cumul_rmse)
        plt.title(f'{models_to_title_additions[model]}')
        plt.xlabel('Confidence percentile')
        plt.ylabel(f'RMSE {datasets_to_units[dataset]}')

        plt.savefig(f'{PLOTS_DIR}/ci_plots/cumulrmse_vs_confidence_one_run_{dataset}_{cf}_{model}.png', dpi=DPI, bbox_inches='tight')
        plt.close()


        # --------------------------------------------------------------------
        # multiple runs

        print(f'Starting multiple runs for {dataset}, {cf}, {model}')
        rmse_mult_runs = []
        within_95_cis_mult_runs = []
        cumulrmse_vs_percentile_corr_mult_runs = []

        flipped_cumulrmse_mult_runs = []

        for i in range(dataset_to_num_cis[dataset]):

            # get data
            y_test = df_true.iloc[:, i]
            y_test_pred = df_pred.iloc[:, i]
            y_test_std = df_std.iloc[:, i]

            # calculate and record rmse
            rmse_mult_runs.append(mean_squared_error(y_true=y_test, y_pred=y_test_pred, squared=False))

            # calculate upper and lower 95% conf bounds
            upper = y_test_pred + 1.96 * y_test_std
            lower = y_test_pred - 1.96 * y_test_std

            # calculate and record proportion of true values within 95% CI from prediction
            within_cis = (lower <= y_test) & (y_test <= upper)
            within_cis_proportion = within_cis.sum() / len(within_cis)
            within_95_cis_mult_runs.append(within_cis_proportion)

            # create a dataframe to be able to sort things easily
            CIs_df = pd.DataFrame(
                {'y_test': y_test,
                 'y_test_pred': y_test_pred,
                 'y_test_std': y_test_std,
                 'lower': lower,
                 'upper': upper,
                 'sq_error': (y_test - y_test_pred) ** 2
                }
            )

            # create cumulative rmse column
            CIs_df = CIs_df.sort_values(by='y_test_std', ascending=True)
            CIs_df['cumul_sq_error'] = CIs_df['sq_error'].cumsum()
            CIs_df['cumul_mse'] = CIs_df['cumul_sq_error'].values / np.arange(1, CIs_df.shape[0]+1)
            CIs_df['cumul_rmse'] = np.sqrt(CIs_df['cumul_mse'])

            # record confidence percentiles and flip cumulative rmses
            confidence_percentiles = np.arange(1e-14, 100, 100/len(y_test))
            flipped_cumul_rmse = CIs_df['cumul_rmse'].values[::-1]

            # record flipped cumulative rmse
            flipped_cumulrmse_mult_runs.append(flipped_cumul_rmse)

            # record correlation between cumulative rmse and confidence percentile
            cumulrmse_vs_percentile_corr_mult_runs.append(pearsonr(confidence_percentiles, flipped_cumul_rmse)[0])

        print(f'Done with multiple runs for {dataset}, {cf}, {model}')


        # --------------------------------------------------------------------
        # calculations for big plots

        flipped_cumulrmse_mean = np.array(flipped_cumulrmse_mult_runs).mean(axis=0)
        flipped_cumulrmse_sdt = np.array(flipped_cumulrmse_mult_runs).mean(axis=0)

        flipped_cumulrmse_lower = flipped_cumulrmse_mean - 1.96*flipped_cumulrmse_sdt
        flipped_cumulrmse_upper = flipped_cumulrmse_mean + 1.96*flipped_cumulrmse_sdt


        # --------------------------------------------------------------------
        # big plots together

        # Create two subplots and unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8)) #, sharey=True

        for flipped_cumul_rmse in flipped_cumulrmse_mult_runs:
            ax1.plot(confidence_percentiles, flipped_cumul_rmse, linewidth=1)

        ax1.set_title(f'{datasets_to_titles[dataset]}. {models_to_title_additions[model]}.')
        #ax1.set_xlabel('Confidence percentile')
        ax1.set_ylabel(f'RMSE {datasets_to_units[dataset]}')


        ax2.plot(confidence_percentiles, flipped_cumulrmse_mean, label=f'mean {metric}')
        ax2.plot(confidence_percentiles, flipped_cumulrmse_lower, label=f'lower 95% CI bound')
        ax2.plot(confidence_percentiles, flipped_cumulrmse_upper, label=f'upper 95% CI bound')
        ax2.fill_between(confidence_percentiles, flipped_cumulrmse_upper, flipped_cumulrmse_lower, facecolor='blue', alpha=0.2)

        ax2.legend(loc='upper center')

        #ax2.set_title(datasets_to_titles[dataset])
        ax2.set_xlabel('Confidence percentile')
        ax2.set_ylabel(f'RMSE {datasets_to_units[dataset]}')

        plt.savefig(f'{PLOTS_DIR}/ci_plots/cumulrmse_vs_confidence_multiple_runs_both_{dataset}_{cf}_{model}.png', dpi=DPI, bbox_inches='tight')
        plt.close()

        print()
