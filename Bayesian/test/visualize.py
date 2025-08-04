import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
if __name__ == "__main__":
    exp = "6GCT_filter"
    mol = Chem.MolFromSmiles("O=C(Nc1cccc(C(=O)Nc2ccc(S(=O)(=O)O)c3cccc(S(=O)(=O)O)c23)c1)Nc1cccc(C(=O)Nc2ccc(S(=O)(=O)O)c3cccc(S(=O)(=O)O)c23)c1")
    mol = Chem.MolFromSmiles("CC(=O)Nc1cccc2c(c1)C(=O)Nc3ccc(cc3)S(=O)(=O)Oc4ccc(cc4)c5ccccc5c2CC(=O)Nc6cccc7c(c6)C(=O)Nc8ccc(cc8)S(=O)(=O)Oc9ccc(cc9)c10ccccc10c7")

    img = Draw.MolToImage(mol)
    img.save(f"../agent2_output_test.png")
    for seed in range(1, 6):
        bo_name = f"../{exp}/{seed}_result.csv"
        # ori_name = f"../6GCT/{seed}_result_ori.csv"
        df = pd.read_csv(bo_name)
        bo_docking = df["Docking Score"].tolist()
        max_docking = max(bo_docking)
        idx = np.argmax(bo_docking)
        mol = Chem.MolFromSmiles(df["Molecule"][idx])
        img = Draw.MolToImage(mol)
        img.save(f"../{exp}/img/{seed}_score_{max_docking}.png")

        # df = pd.read_csv(ori_name)
        # ori_docking = df["Docking Score"].tolist()
        # max_docking = max(ori_docking)
        # idx = np.argmax(ori_docking)
        # mol = Chem.MolFromSmiles(df["Molecule"][idx])
        # img = Draw.MolToImage(mol)
        # img.save(f"../6GCT/img/{seed}_ori_score_{max_docking}.png")