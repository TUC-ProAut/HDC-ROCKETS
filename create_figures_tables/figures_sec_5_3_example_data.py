# script to plot examples from the UCR dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data.dataset_utils import *

from plot_config import mpl

save_path = "."

def main():
    IDs = [58, 19, 110, 98, 100, 58, 99]
    cs =  [22, 1, 1, 26, 41, 16, 0]
    max_samples = 5
    fs = 20
    config = {
        'dataset_path': "~/Datasets",
        'dataset_idx': 0,
        'dataset_name': 'UCR_NEW',
        'hard_case': False,
        'seed': 0,
        'normalize_input': True,
        'seed_for_splits': True,
    }
    for ID, c in zip(IDs, cs):
        config['dataset_idx'] = ID
        data = load_dataset('UCR_NEW', config)
        X_train = data[0]
        y_train = data[2]

        # Plot example data from positive and negative class
        fig, axs = plt.subplots(2, 1, figsize=(5, 7))
        # Plot 5 examples from the positive class
        for i in range(min(max_samples, np.sum(y_train == c))):
            axs[0].plot(X_train[y_train == c][i, 0, :], alpha=0.8)
        axs[0].set_title(f"Class {c} of UCR dataset {ID}", fontsize=fs)
        axs[0].set_xlabel("Time", fontsize=fs)
        axs[0].grid()
        # Plot 5 examples from the negative class in different colors per class
        for i in range(min(max_samples, np.sum(y_train != c))):
            random_idx = np.random.choice(np.where(y_train != c)[0])
            axs[1].plot(X_train[random_idx, 0, :], alpha=0.8)
        axs[1].set_title(f"All other classes", fontsize=fs)
        axs[1].set_xlabel("Time", fontsize=fs)
        axs[1].grid()
        plt.tight_layout()
        fig.savefig(f"{save_path}/images/ucr_data_examples_{str(ID)}_{str(c)}.pdf", format='pdf')
        plt.show()

if __name__ == '__main__':
    main()