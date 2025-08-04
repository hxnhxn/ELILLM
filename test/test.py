import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    # name_protein = "6GCT_filter"

    mean_init_list = []
    mean_bo_list = []
    mean_ori_list = []
    dir_path = "results/pdb_bind_10"
    name_proteins = ['1o3h','2vnf','3lw0','1ynd','4c16','5cyv','5zty','6b4h','7gpb','8abp']

    for name in name_proteins:
        name_protein = os.path.join(dir_path, name)
        # result_path = f"../{name_protein}/result.csv"
        # result_df = pd.read_csv(result_path)
        # docking_scores = result_df["Docking Score"].tolist()
        # mean = np.mean(docking_scores)
        # std = np.std(docking_scores)

        # result_path = f"../{name_protein}/result_ori.csv"
        # result_df = pd.read_csv(result_path)
        # docking_scores = result_df["Docking Score"].tolist()
        # mean_ori = np.mean(docking_scores)
        # std_ori = np.std(docking_scores)

        result_path = f"../{name_protein}/init_score.csv"
        result_df = pd.read_csv(result_path)
        docking_scores = result_df["docking_scores"].tolist()
        mean_init = np.mean(docking_scores)
        mean_init_list.append(mean_init)

        std_init = np.std(docking_scores)
        min_init = np.min(docking_scores)
        maax_init = np.max(docking_scores)
        pass
        bo_dockings = []
        ori_dockings = []
        for seed in range(1,6):
            bo_name = f"../{name_protein}/{seed}_result.csv"

            ori_name = f"../{name_protein}/{seed}_result_ori.csv"
            df = pd.read_csv(bo_name)
            bo_dockings.extend(df["Docking Score"].tolist())
            df = pd.read_csv(ori_name)
            ori_dockings.extend(df["Docking Score"].tolist())

        df = pd.DataFrame(bo_dockings)
        df.to_csv(f"../{name_protein}/bo_docking.csv", index=False)
        df = pd.DataFrame(ori_dockings)
        df.to_csv(f"../{name_protein}/ori_docking.csv", index=False)
        mean_bo = np.mean(bo_dockings)
        mean_bo_list.append(mean_bo)

        min_bo = np.min(bo_dockings)
        std_bo = np.std(bo_dockings)

        mean_ori = np.mean(ori_dockings)
        mean_ori_list.append(mean_ori)
        # min_ori = np.min(ori_dockings)
        # std_ori = np.std(ori_dockings)
        max_bo = np.max(bo_dockings)
        max_ori = np.max(ori_dockings)

    data = {
        'mean_init': mean_init_list,
        'mean_ori': mean_ori_list,
        'mean_bo': mean_bo_list
    }
    df = pd.DataFrame(data, index=name_proteins)
    df.to_csv(f"../{dir_path}/mean_result_score.csv", index_label='protein')

    print(mean_init_list)
    print(mean_ori_list)
    print(mean_bo_list)
    pass