import sys, os
import pandas as pd

# set the dataset name to be working with
dataset = 'esol'
# load data
df = pd.read_csv(f'../data/{dataset}_original.csv')


# Create a dataframe only with id, smile, labels columns
subset_df = df[['Compound ID', 'smiles', 'measured log solubility in mols per litre']]
columns_mapper = {'Compound ID': 'id',
                  'measured log solubility in mols per litre': 'labels'
                 }
ready_df = subset_df.rename(columns=columns_mapper)


# Create a dataframe with esol extra features
extra_features = df[['Compound ID', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors',
       'Number of Rings', 'Number of Rotatable Bonds', 'Polar Surface Area']]
ready_extra_features = extra_features.rename(columns={'Compound ID':'id'})
ready_extra_features.head()

# save files
ready_df.to_csv(f'../data/{dataset}_original_IdSmilesLabels.csv', index=False)
ready_extra_features.to_csv(f'../data/{dataset}_original_extra_features.csv', index=False)

## Check that we have all the files we should have after running this script

# list files present, and input the names of the original files
present = set(os.listdir('../data/'))
original_files = set(['esol_original.csv', 'freesolv_original.csv', 'lipophilicity_original.csv'])

# check that we have not deleted original files
assert original_files.issubset(present)
# check that we produced the needed file(s)
assert f'{dataset}_original_IdSmilesLabels.csv' in present
assert f'{dataset}_original_extra_features.csv' in present
