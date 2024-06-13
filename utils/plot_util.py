
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter


def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.set(style="white", palette="muted")
    # palette = ['#A8BAE3', '#55AB83']
    # palette = ['#A8BAE3', '#FF9999']
    palette = ['#8E8BFE', '#FEA3A2']
    sns_plt = sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", 
                          palette=palette, fill = True, alpha = 0.8, linewidth=3, legend=False)
    # sns_plt._legend.set_bbox_to_anchor((0.85, 0.85))
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()  # 获取当前的ax对象
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # 设置线条宽度为2

    plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

# def plot_distribution(args, id_scores, ood_scores, out_dataset):
#     sns.set(style="white", palette="muted")
#     palette = ['#A8BAE3', '#55AB83']
    
#     sns_plt = sns.displot({"ID": -1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
#     sns_plt._legend.set_bbox_to_anchor((0.85, 0.85))
    
#     # Set x-axis to scientific notation
#     ax = plt.gca()  # 获取当前的轴
#     formatter = ScalarFormatter(useMathText=True)
#     ax.xaxis.set_major_formatter(formatter)
#     ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='x')
    
#     # Ensure that the font of scientific notation matches other x-axis labels
#     ax.xaxis.get_offset_text().set_fontsize(plt.rcParams['xtick.labelsize'])
    
#     plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.png"), bbox_inches='tight')


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


