from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tools import makedir
import os
import shutil
import random
import pandas as pd

if __name__ == "__main__":
    random.seed(1)
    folder = Path("yourpath/datasets/crossdocked/structure-files-train/")
    sdf_files = folder.glob("*.sdf")
    cnt = 0
    for _ in sdf_files:
        cnt += 1
    random_select_id = random.sample(range(cnt), 100)
    smiles_list = []
    for idx in random_select_id:
        file_path = folder.joinpath(f"{idx}-ligand.sdf")
        if os.path.isfile(file_path):
            supplier = Chem.SDMolSupplier(file_path)
            for mol in supplier:
                if mol is not None:
                    AllChem.Compute2DCoords(mol)
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
    selected_df = pd.DataFrame(smiles_list)
    selected_df.to_csv('../results/crossdocked/init.csv', index=False)
    for i in range(100):
        makedir(f'../results/crossdocked/{i}')
        shutil.copy('../results/crossdocked/init.csv', f'../results/crossdocked/{i}')
