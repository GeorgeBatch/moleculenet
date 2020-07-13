# Import modules
import numpy as np
import pandas as pd
import time

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger
from rdkit.Chem import Descriptors


# record time to show after execution of the script
since = time.time()

# For each dataset*smile_typecombination get the rdkit features from smile strings
for dataset in ['esol', 'freesolv', 'lipophilicity']:
    for smile_type in ['original']:
        # tell the user what is happenning
        print(f'Working on {dataset} dataset, {smile_type} smile_type...')

        ## Load Data
        data = pd.read_csv(f'../data/{dataset}_{smile_type}_IdSmilesLabels.csv', index_col=0)
        smiles = data['smiles']

        ## Get RDKit Molecular descriptors

        # load ligands and compute features
        features = {}
        descriptors = {d[0]: d[1] for d in Descriptors.descList}

        for index in smiles.index:

            mol = Chem.MolFromSmiles(smiles.loc[index])

            # how exactly do we add hydrogens here???
            mol = Chem.AddHs(mol)

            try:
                features[index] = {d: descriptors[d](mol) for d in descriptors}
            except ValueError as e:
                print(e)
                continue

        features = pd.DataFrame.from_dict(features).T

        # save file
        file_path = f'../data/{dataset}_{smile_type}_rdkit_features.csv'
        features.to_csv(file_path, index=True)
        print(f'Saved file to: {file_path}\n')

time_elapsed = time.time() - since
print(f'Task completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s \n')
