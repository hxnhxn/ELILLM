import pickle
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

def read_baseline(dir):
    with open(dir, "rb") as f:
        result = pickle.load(f)
    return result
def read_ours_and_alidiff():
    dir = f"../results/alidiff_init_rdkit_seed"
    with open(f"{dir}/mean_result.pickle", "rb") as f:
        result = pickle.load(f)
    alidiff_result =  {}
    ours_result = {}
    for k in [1,5,10,20]:
        alidiff_result[f"mean_list_{k}"] = result[f"init_mean_list_{k}"]
        ours_result[f"mean_list_{k}"] = result[f"bo_mean_list_{k}"]
    return ours_result, alidiff_result

if __name__ == "__main__":
    ours_result, alidiff_result = read_ours_and_alidiff()
    pocket2mol_result = read_baseline("../baselines/pocket2mol_result/mean_result.pickle")
    targetdiff_result = read_baseline("../baselines/targetdiff_result/mean_result.pickle")
    cvae_result = read_baseline("../baselines/cvae_result/mean_result.pickle")
    # tamgen_result = read_baseline("../baselines/tamgen_result/mean_result.pickle")
    # stat, p_value = mannwhitneyu(ours_result["mean_list_5"], targetdiff_result["mean_list_5"], alternative='less')
    for k in [1,5,10,20]:
        stat, cvae_p_value = wilcoxon(ours_result[f"mean_list_{k}"], cvae_result[f"mean_list_{k}"], alternative='less')
        stat, pocket2mol_p_value = wilcoxon(ours_result[f"mean_list_{k}"], pocket2mol_result[f"mean_list_{k}"], alternative='less')
        stat, targetdeff_p_value = wilcoxon(ours_result[f"mean_list_{k}"], targetdiff_result[f"mean_list_{k}"], alternative='less')
        stat, alidiff_p_value = wilcoxon(ours_result[f"mean_list_{k}"], alidiff_result[f"mean_list_{k}"], alternative='less')
        print(f"k={k}: ")
        print("cvae_p_value: ", f"{cvae_p_value:.2e}", "pocket2mol_p_value: ", f"{pocket2mol_p_value:.2e}", "targetdeff_p_value: ", f"{targetdeff_p_value:.2e}"
              , "alidiff_p_value: ", f"{alidiff_p_value:.2e}")
    pass