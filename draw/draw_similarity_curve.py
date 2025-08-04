import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Draw Figure 3
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    x = np.arange(10, 101, 10)
    our_sim_df = pd.read_csv('our_sim_curve.csv')
    lmlf_sim_df = pd.read_csv('lmlf_sim_curve.csv')

    our_ali_sim_df = pd.read_csv('our_ali_sim_curve.csv')
    lmlf_ali_sim_df = pd.read_csv('lmlf_ali_sim_curve.csv')

    ax1, ax2, ax3, ax4 = axes

    ax1.plot(x, our_sim_df["mean_sim"], marker='o', color='red', label='ELILLM')
    ax1.plot(x, lmlf_sim_df["mean_sim"], marker='s', color='blue', label='LMLF')

    ax2.plot(x, our_sim_df["max_sim"], marker='o', color='red', label='ELILLM')
    ax2.plot(x, lmlf_sim_df["max_sim"], marker='s', color='blue', label='LMLF')

    ax3.plot(x, our_ali_sim_df["mean_sim"], marker='o', color='red', label='ELILLM')
    ax3.plot(x, lmlf_ali_sim_df["mean_sim"], marker='s', color='blue', label='LMLF')

    ax4.plot(x, our_ali_sim_df["max_sim"], marker='o', color='red', label='ELILLM')
    ax4.plot(x, lmlf_ali_sim_df["max_sim"], marker='s', color='blue', label='LMLF')

    labels = ['a', 'b', 'c', 'd']
    for i, ax in enumerate(axes):
        ax.text(-0.1, 1.08, labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
    ax1.set_ylabel('Mean similarity')
    ax3.set_ylabel('Mean similarity')
    ax2.set_ylabel('Max similarity')
    ax4.set_ylabel('Max similarity')
    ax1.set_title('rand')
    ax2.set_title('rand')
    ax3.set_title('diff')
    ax4.set_title('diff')
    # handles, legend_labels = ax1.get_legend_handles_labels()
    #
    # # 图例放在整张图顶部中央
    # fig.legend(handles, legend_labels, loc='upper center', ncol=2,
    #            bbox_to_anchor=(0.5, 1.02), fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax2.legend(loc='best', fontsize=10)
    ax3.legend(loc='best', fontsize=10)
    ax4.legend(loc='best', fontsize=10)

    # x轴标签放底部中央
    fig.text(0.5, 0.05, 'Number of new molecules', ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # 顶部和底部留空间
    plt.savefig("similarity_curve.svg", bbox_inches='tight')
    plt.show()


