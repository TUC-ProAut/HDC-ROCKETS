from argparse import ArgumentParser
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--result_path',
                        help='Path to results folder',
                        default="results")
    parser.add_argument('--dataset_idx',
                        help='index for UCR/UEA dataset ensemble - if all ensembles are used, set to -1',
                        type=int,
                        default=-1)
    parser.add_argument('--scale',
                        help='scaling of the scalar encoding with fractional binding ',
                        default=0)
    parser.add_argument('--kernel',
                        help='kernel for fractional binding',
                        default='sinc')
    args = parser.parse_args()

    global_save_path = args.result_path


    print(f"Result path: {global_save_path}")

    # read config json file
    with open(os.path.join(global_save_path, "config_log.json"), 'r') as f:
        config = json.load(f)

    # get model name
    model = config['model']
    configuration = config['variant']
    dataset = config['dataset']

    # filter all idx from log file
    if args.dataset_idx == -1:
        # find all acc_idx(*) in config
        idx_range = []
        keys = config.keys()
        for key in keys:
            match = re.search(rf"acc_idx(\d+)_", key)
            if match:
                idx_range.append(int(match.group(1)))
        idx_range = np.asarray(np.unique(idx_range))
    else:
        idx_range = [args.dataset_idx]

    # initial file creation
    save_path = os.path.join(global_save_path,"eval_results/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # find all scales and kernels in file names
    scores_files = [f for f in os.listdir(global_save_path) if re.search(r"scores_\d+.h5", f) and "lock" not in f]
    scores_data_per_seed = {s : pd.HDFStore(os.path.join(global_save_path, s), mode='r') for s in scores_files}
    keys = []
    for file in scores_data_per_seed:
        keys += scores_data_per_seed[file].keys()

    scales = np.unique([int(re.search(r"scale_(\d+)_", key).group(1)) for key in keys if re.search(r"scale_(\d+)_", key)])
    kernels = np.unique([re.search(r"kernel_([a-zA-Z0-9]+)", key).group(1) for key in keys if re.search(r"kernel_([a-zA-Z0-9]+)", key)])
    seeds = np.unique([int(re.search(r"scores(\d+)_", key).group(1)) for key in keys if re.search(r"scores(\d+)_", key)])

    # create empty dataframe with headers idx and all combinations of scale and kernel
    results = pd.DataFrame(columns=[f"{dataset}_idx"] +
                                   [f"auc_mean_{scale}_{kernel}" for scale in scales for kernel in kernels] +
                                     [f"prc_mean_{scale}_{kernel}" for scale in scales for kernel in kernels])

    # dataframe for individual aucroc scores
    results_indiv = pd.DataFrame(columns=[f"{dataset}_idx"] +
                                         [f"auc_mean_{scale}_{kernel}" for scale in scales for kernel in kernels] +
                                         [f"prc_mean_{scale}_{kernel}" for scale in scales for kernel in kernels])
    for i, idx in enumerate(idx_range):
        print(f"Ensemble idx: {idx}")
        description = {'model_name': model,
                       'dataset': dataset,
                       'dataset_idx': idx}
        # read all scales and kernel
        for kernel in kernels:
            for scale in scales:
                # load data (first column is class_id) with all seeds
                data = [scores_data_per_seed[f"scores_{seed}.h5"][f"scores{seed}_{dataset}_idx_{idx}_scale_{scale}_kernel_{kernel}"] for seed in seeds]
                # set first column as ground truth
                data = [d.rename(columns={d.columns[0]: 'class_id'}) for d in data]
                # set all other columns as score values
                data = [d.rename(columns={d.columns[i]: f'score_val_{i-1}' for i in range(1, len(d.columns))}) for d in data]

                roc_auc = []
                prc_auc = []
                roc_auc_indiv = []
                prc_auc_indiv = []
                for data_seed in data:
                    # replace inf and nan with 0
                    data_seed = data_seed.replace([np.inf, -np.inf], np.nan).fillna(0)
                    n_classes = len(data_seed['class_id'].unique())
                    # if multi-class
                    if n_classes > 2:
                        # compute the roc auc score (use all columns that start with score_val)
                        scores = data_seed[[col for col in data_seed.columns if col.startswith('score_val')]].values.astype(np.float32)
                        # convert to probabilities
                        scores = (scores - np.min(scores, axis=1)[:, None]) / (np.max(scores, axis=1)[:, None] - np.min(scores, axis=1)[:, None])
                        scores = scores/np.sum(scores, axis=1)[:, None]
                        # scores = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, None]
                        roc_auc.append(roc_auc_score(data_seed['class_id'], scores, multi_class='ovr', average='weighted'))
                        roc_auc_indiv.append(roc_auc_score(data_seed['class_id'], scores, multi_class='ovr', average=None))
                    else:
                        # compute the roc auc score (use all columns that start with score_val)
                        roc_auc.append(roc_auc_score(data_seed['class_id'], data_seed[
                                [col for col in data_seed.columns if col.startswith('score_val')]].values))
                        roc_auc_indiv.append(roc_auc_score(data_seed['class_id'], data_seed[[col for col in data_seed.columns if col.startswith('score_val')]].values, average=None))

                    # calc prc
                    auc_prc_scores = []
                    for c in range(data_seed.shape[1] - 1):
                        y_true = (data_seed['class_id'] == c).astype(int).values
                        # Calculate precision-recall pairs
                        precision, recall, _ = precision_recall_curve(y_true, data_seed.iloc[:, c + 1])
                        # Calculate the area under the precision-recall curve
                        auc_prc = auc(recall, precision)
                        auc_prc_scores.append(auc_prc)
                    prc_auc.append(np.mean(auc_prc_scores))
                    prc_auc_indiv.append(auc_prc_scores)
                roc_auc = np.mean(roc_auc)
                prc_auc = np.mean(prc_auc)
                prc_auc_indiv = np.mean(np.asarray(prc_auc_indiv),0)
                roc_auc_indiv = np.mean(np.asarray(roc_auc_indiv),0)
                # add individual aucroc scores to dataframe and corresponding column
                results_indiv.loc[i, f"{dataset}_idx"] = idx
                results_indiv.loc[i, f"auc_mean_{scale}_{kernel}"] = [roc_auc_indiv]
                results_indiv.loc[i, f"prc_mean_{scale}_{kernel}"] = [prc_auc_indiv]

                # write to dataframe
                results.loc[i, f"{dataset}_idx"] = idx
                results.loc[i, f"auc_mean_{scale}_{kernel}"] = roc_auc
                results.loc[i, f"prc_mean_{scale}_{kernel}"] = prc_auc

    # save dataframe
    results.to_pickle(f"{save_path}auc.pkl")
    results_indiv.to_pickle(f"{save_path}auc_indiv.pkl")

    # close all open HDF5 stores
    for store in scores_data_per_seed.values():
        store.close()

    print(f"    Results saved to {save_path}")
