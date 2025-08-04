import pickle
import numpy as np
import matplotlib.pyplot as plt
def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Draw Figure 2
    elillm_result = open_pickle('elillm.pickle')
    targetdiff_result = open_pickle('targetdiff.pickle')
    alidiff_result = open_pickle('alidiff.pickle')

    lsbo4llm_top1_list = elillm_result['bo_mean_list_1']
    # lsbo4llm_top1_list = alidiff_result['bo_mean_list_1']
    targetdiff_top1_list = targetdiff_result['mean_list_1']
    alidiff_top1_list = alidiff_result['init_mean_list_1']

    lsbo4llm_top1_cnt = 0
    targetdiff_top1_cnt = 0
    alidiff_top1_cnt = 0
    highest_cnt = 0
    for i in range(len(lsbo4llm_top1_list)):
        min_score = min([lsbo4llm_top1_list[i], targetdiff_top1_list[i], alidiff_top1_list[i]])
        if min_score == lsbo4llm_top1_list[i]:
            lsbo4llm_top1_cnt += 1
            continue
        if lsbo4llm_top1_list[i] >= targetdiff_top1_list[i] and lsbo4llm_top1_list[i] >= alidiff_top1_list[i]:
            highest_cnt += 1
        if min_score == alidiff_top1_list[i]:
            alidiff_top1_cnt += 1
            continue
        if min_score == targetdiff_top1_list[i]:
            targetdiff_top1_cnt += 1
    print(lsbo4llm_top1_cnt, alidiff_top1_cnt, targetdiff_top1_cnt, highest_cnt)

    # indices = np.argsort(lsbo4llm_top1_list)
    indices = np.argsort(targetdiff_top1_list)
    sorted_lsbo4llm_top1_list = np.array(lsbo4llm_top1_list)[indices]
    sorted_targetdiff_top1_list = np.array(targetdiff_top1_list)[indices]
    sorted_alidiff_top1_list = np.array(alidiff_top1_list)[indices]
    # sorted_lsbo4llm_diff_top1_list = np.array(lsbo4llm_diff_top1_list)[indices]

    num_targets = 100
    targets = np.arange(num_targets)

    fit_targetdiff = np.polyfit(targets, sorted_targetdiff_top1_list, 1)
    fit_alidiff = np.polyfit(targets, sorted_alidiff_top1_list, 1)
    fit_lsbo4llm = np.polyfit(targets, sorted_lsbo4llm_top1_list, 1)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(targets, sorted_targetdiff_top1_list, color='steelblue', label=f'TargetDiff(lowest in {targetdiff_top1_cnt}%)\n'
                                                                              f'y={fit_targetdiff[0]:.2f}x-{-fit_targetdiff[1]:.2f}', s=50)
    ax.scatter(targets, sorted_alidiff_top1_list, color='green', label=f'ALIDIFF (lowest in {alidiff_top1_cnt}%)\n'
                                                                       f'y={fit_alidiff[0]:.2f}x-{-fit_alidiff[1]:.2f}')
    ax.scatter(targets, sorted_lsbo4llm_top1_list, color='red', label=f'ELILLM-rand (lowest in {lsbo4llm_top1_cnt}%)\n'
                                                                      f'y={fit_lsbo4llm[0]:.2f}x-{-fit_lsbo4llm[1]:.2f}')

    line_targetdiff = np.poly1d(fit_targetdiff)(targets)
    line_alidiff = np.poly1d(fit_alidiff)(targets)
    line_lsbo4llm = np.poly1d(fit_lsbo4llm)(targets)

    # 画拟合线（可调颜色、线宽、透明度）
    ax.plot(targets, line_targetdiff, color='steelblue', linewidth=1.4, alpha=0.6)
    ax.plot(targets, line_alidiff, color='green', linewidth=1.4, alpha=0.6)
    ax.plot(targets, line_lsbo4llm, color='red', linewidth=1.4, alpha=0.6)
    # ax.scatter(targets, sorted_lsbo4llm_diff_top1_list, color='yellow', label=f'LSBO4LLM-diff')
    for x in targets:
        ax.axvline(x=x, color='black', alpha=0.6, linewidth=0.5)


    ax.set_xlim(-1, num_targets)
    ax.set_xticks(np.arange(0, num_targets, 10))
    ax.set_xticklabels([f'target {i}' for i in range(0, num_targets, 10)])
    ax.set_ylabel("Top1 Vina Score")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    legend = ax.legend(loc='best', fontsize=10, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    ax.tick_params(axis='x', which='both', direction='out', length=3)
    ax.tick_params(axis='y', which='both', length=0)


    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_alpha(0.6)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig("scatter_per_target.svg")
    plt.show()
    pass