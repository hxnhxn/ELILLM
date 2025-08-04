import pandas as pd
import numpy as np
import os
from tools.tools import internal_diversity,get_fingerprint,tanimoto_similarity
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
import pickle

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

def calculate_mean_similarity(fps1, fps2):
    sims = []
    for fp1 in fps1:
        for fp2 in fps2:
            sims.append(tanimoto_similarity(fp1,fp2))
    return np.mean(sims)
def calculate_max_similarity(fps1, fps2):
    sims = []
    for fp2 in fps2:
        now_sims = []
        for fp1 in fps1:
            now_sims.append(tanimoto_similarity(fp1,fp2))
        sims.append(max(now_sims))
    return np.mean(sims)
if __name__ == "__main__":
    # name_protein = "6GCT_filter"

    dir_path = "results/rand"
    # dir_path = "results/ablation_random_select_emb"
    # dir_path = "results/ablation_no_knowledge"
    mean_sim_list = [[] for i in range(10)]
    max_sim_list = [[] for i in range(10)]

    for i in range(100):
        name_protein = os.path.join(dir_path, str(i))
        result_path = f"../{name_protein}/init_score.csv"
        result_df = pd.read_csv(result_path)
        init_docking_scores = result_df["docking_scores"].tolist()
        init_smiles = result_df["smile"].tolist()
        init_mols = [MolFromSmiles(smiles) for smiles in init_smiles]
        init_fps = get_fingerprint(init_mols)

        # init_docking_scores = np.sort(init_docking_scores)
        bo_dockings = []
        bo_smiles = []
        for seed in range(1, 2):
            bo_name = f"../{name_protein}/{seed}_result.csv"
            df = pd.read_csv(bo_name)
            bo_dockings.extend(df["Docking Score"].tolist())
            bo_smiles.extend(df["Molecule"].tolist())

        if len(bo_dockings) != 100:
            print(i)
            continue
        # bo_dockings = np.sort(bo_dockings).tolist()


        current_all_fp = []
        for k in range(10):
            current_smiles = bo_smiles[k*10:(k+1)*10]
            current_mols = [MolFromSmiles(smiles) for smiles in current_smiles]
            current_fp = get_fingerprint(current_mols)
            current_all_fp.extend(current_fp)
            mean_sim_list[k].append(calculate_mean_similarity(init_fps, current_fp))
            # order is necessary
            max_sim_list[k].append(calculate_max_similarity(init_fps, current_fp))

        pass

        # df = pd.DataFrame(topk)
        # df.to_csv(f"../{name_protein}/top{k}.csv", index=False)
    max_sim = []
    mean_sim = []
    print("mean_sim:")
    for k in range(10):
        mean_sim.append(np.mean(mean_sim_list[k]))
        print(f"{k}:", mean_sim[k])
    for k in range(10):
        max_sim.append(np.mean(max_sim_list[k]))
        print(f"{k}:", max_sim[k])
    sim_curve = {"mean_sim": mean_sim, "max_sim": max_sim}
    df = pd.DataFrame(sim_curve)
    df.to_csv(f"../{dir_path}/sim_curve.csv", index=False)
