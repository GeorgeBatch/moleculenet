# Import modules
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


# For each dataset*smile_type combination get the ECFP-4 and ECFP-6 features from smile strings
for dataset in ['esol', 'freesolv', 'lipophilicity']:
    for smile_type in ['original']:
        # tell the user what is happenning
        print(f'Working on {dataset} dataset, {smile_type} smile_type...')

        # load data
        data = pd.read_csv(f'../data/{dataset}_{smile_type}_IdSmilesTarget.csv', index_col=0)

        # get smile-strings and constract molecules from them
        smiles = data['smiles']
        ms = [Chem.MolFromSmiles(smile) for smile in smiles]

        # get ecfp-4, ecfp-6 (with radii respectively 2 and 3)
        for radius in [2, 3]:
            # get ECFP features
            ecfp = [AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=2048) for m in ms]
            ecfp_np = np.array(ecfp)
            ecfp_pd = pd.DataFrame(ecfp_np)
            ecfp_pd.index = list(data.index)
            ecfp_pd.columns = [f'ecfp{radius*2}-{i}' for i in range(len(ecfp_pd.columns))]

            # save ECFP features to .csv files
            file_path = f'../data/{dataset}_{smile_type}_ecfp{radius*2}_features.csv'
            ecfp_pd.to_csv(file_path, index=True)
            print(f'Saved file to: {file_path}')

        # skip line when starting to work on a new dataset*smile_type combination
        print()
