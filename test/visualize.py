import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
if __name__ == "__main__":
    exp = "6GCT_filter"

    mol = Chem.MolFromSmiles("CC1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1C1CCC1")
    img = Draw.MolToImage(mol)
    img.save(f"../error_result.png")
    # for seed in range(1, 6):
    #     bo_name = f"../{exp}/{seed}_result.csv"
    #     # ori_name = f"../6GCT/{seed}_result_ori.csv"
    #     df = pd.read_csv(bo_name)
    #     bo_docking = df["Docking Score"].tolist()
    #     max_docking = max(bo_docking)
    #     idx = np.argmax(bo_docking)
    #     mol = Chem.MolFromSmiles(df["Molecule"][idx])
    #     img = Draw.MolToImage(mol)
    #     img.save(f"../{exp}/img/{seed}_score_{max_docking}.png")

        # df = pd.read_csv(ori_name)
        # ori_docking = df["Docking Score"].tolist()
        # max_docking = max(ori_docking)
        # idx = np.argmax(ori_docking)
        # mol = Chem.MolFromSmiles(df["Molecule"][idx])
        # img = Draw.MolToImage(mol)
        # img.save(f"../6GCT/img/{seed}_ori_score_{max_docking}.png")