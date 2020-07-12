import sys, os
import pandas as pd

# set the dataset name to be working with
dataset = 'freesolv'
# load data
df = pd.read_csv(f'../data/{dataset}_original.csv')

# Create a dataframe only with id, smile, target columns
subset_df = df[['iupac', 'smiles', 'expt']]
columns_mapper = {'iupac': 'id',
                  'expt': 'target'
                 }
ready_df = subset_df.rename(columns=columns_mapper)


# save file
ready_df.to_csv(f'../data/{dataset}_original_IdSmilesTarget.csv', index=False)



## Check that we have all the files we should have after running this script

# list files present, and input the names of the original files
present = set(os.listdir('../data/'))
original_files = set(['esol_original.csv', 'freesolv_original.csv', 'lipophilicity_original.csv'])

# check that we have not deleted original files
assert original_files.issubset(present)
# check that we produced the needed file
assert f'{dataset}_original_IdSmilesTarget.csv' in present
