# Import modules
import time
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

# record time to show after execution of the script
since = time.time()

# For each dataset*smile_type combination get the ECFP-4 and ECFP-6 features from smile strings
for dataset in ['esol', 'freesolv', 'lipophilicity']:
    for smile_type in ['original']:
        # tell the user what is happenning
        print(f'Working on {dataset} dataset, {smile_type} smile_type...')

        # load data
        data = pd.read_csv(f'../data/{dataset}_{smile_type}_IdSmilesLabels.csv', index_col=0)

        # get smile-strings and constract molecules from them
        smiles = data['smiles']
        ms = [Chem.MolFromSmiles(smile) for smile in smiles]

        # get ecfp-4, ecfp-6 (with radii respectively 2 and 3)
        for radius in [2, 3]:
            # get them in 1024 and 2048 bit versions
            for nBits in [1024, 2048]:
                # get ECFP features
                ecfp = [AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in ms]
                ecfp_np = np.array(ecfp)
                ecfp_pd = pd.DataFrame(ecfp_np)
                ecfp_pd.index = list(data.index)
                ecfp_pd.columns = [f'{nBits}ecfp{radius*2}-{i}' for i in range(len(ecfp_pd.columns))]

                # save ECFP features to .csv files
                file_path = f'../data/{dataset}_{smile_type}_{nBits}ecfp{radius*2}_features.csv'
                ecfp_pd.to_csv(file_path, index=True)
                print(f'Saved file to: {file_path}')

        # skip line when starting to work on a new dataset*smile_type combination
        print()

time_elapsed = time.time() - since
print(f'Task completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s \n')
