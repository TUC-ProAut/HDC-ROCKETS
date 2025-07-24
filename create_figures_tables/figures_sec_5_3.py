import os
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob

from sympy.abc import alpha

from cd_diagram import draw_cd_diagram
from scipy.stats import friedmanchisquare
from multi_comp_matrix import MCM
import json
import subprocess
import sys

from plot_config import mpl
from plot_config import modern_palette
from data.constants import UCR_NEW_PREFIX

pd.DataFrame.iteritems = pd.DataFrame.items # for compatibility with older pandas versions
force_recompute = False

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# define the folder paths for models
kernels = ['sinc', 'gaussian', 'triangular', 'all_kernels']
only_best = True
fs = 15

for kernel in kernels:
    print("--"*20)
    print(f"\nKernel: {kernel}\n")
    # path to the results folder of the original models
    orig_model_path = "../results/UCR/_orig_models"
    # path to the results folder of the HDC models
    hdc_path = f"../results/UCR/_{kernel}/"
    # path to the results folder of the HDC models
    save_path = os.path.abspath(f"images/{kernel}/")
    save_path_tables = os.path.abspath(f"tables/{kernel}/")

    # create the save path if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables)

    scale = 0
    n_scales = 7

    # read all folders from orig and hdc path
    model_paths = []
    for folder in os.listdir(orig_model_path):
        dir = os.path.join(orig_model_path, folder)
        # check if dir is a directory and does not start with "_" (symbol for not included in paper)
        if os.path.isdir(dir) and not folder.startswith('_'):
            model_paths.append(dir)
    for folder in os.listdir(hdc_path):
        dir = os.path.join(hdc_path, folder)
        # check if dir is a directory
        if os.path.isdir(dir) and not folder.startswith('_'):
            model_paths.append(dir)

    if kernel == 'all_kernels':
        kernels = ['sinc', 'gaussian', 'triangular']
    else:
        kernels = [kernel]

    abbreviations = {'MINIROCKET': 'MiniROCKET',
                     'HDC-MINIROCKET auto': 'HDC-MiniROCKET auto',
                     'HDC-MINIROCKET oracle': 'HDC-MiniROCKET oracle',
                     'MULTIROCKET': 'MultiROCKET',
                     'HDC-MULTIROCKET auto': 'HDC-MultiROCKET auto',
                     'HDC-MULTIROCKET oracle': 'HDC-MultiROCKET oracle',
                     'HYDRA': 'HYDRA',
                     'HDC-HYDRA auto': 'HDC-HYDRA auto',
                     'HDC-HYDRA oracle': 'HDC-HYDRA oracle',
                     'MULTIROCKET-HYDRA': 'MultiROCKET-HYDRA',
                     'HDC-MULTIROCKET-HYDRA auto': 'HDC-MultiROCKET-HYDRA auto',
                     'HDC-MULTIROCKET-HYDRA oracle': 'HDC-MultiROCKET-HYDRA oracle',
                     }

    model_columns = {'MINIROCKET': [f"_mean_0_sinc"],
                    'HDC-MINIROCKET auto': ['_mean_0_auto'],
                    # 'HDC-MINIROCKET auto cv': ['_mean_0_auto'],
                    'HDC-MINIROCKET oracle': [f"_mean_{i}_{k}" for i in range(n_scales) for k in kernels],
                    'MULTIROCKET': ['_mean_0_sinc'],
                    'HDC-MULTIROCKET oracle': [f"_mean_{i}_{k}" for i in range(n_scales) for k in kernels],
                    'HDC-MULTIROCKET auto': ['_mean_0_auto'],
                    'HYDRA': ['_mean_0_sinc'],
                    'HDC-HYDRA auto': ['_mean_0_auto'],
                    'HDC-HYDRA oracle': [f"_mean_{i}_{k}" for i in range(n_scales) for k in kernels],
                    'MULTIROCKET-HYDRA': ['_mean_0_sinc'],
                    'HDC-MULTIROCKET-HYDRA auto': ['_mean_0_auto'],
                    'HDC-MULTIROCKET-HYDRA oracle': [f"_mean_{i}_{k}" for i in range(n_scales) for k in kernels],
                     }
    model_columns = {abbreviations[key]: value for key, value in model_columns.items()}

    # mapping of model names for comparison
    model_mapping = {'MINIROCKET': 'MINIROCKET',
                    'HDC-MINIROCKET auto': 'MINIROCKET',
                    # 'HDC-MINIROCKET auto cv': 'MINIROCKET',
                    'HDC-MINIROCKET oracle': 'MINIROCKET',
                    'MULTIROCKET': 'MULTIROCKET',
                    'HDC-MULTIROCKET auto': 'MULTIROCKET',
                    'HDC-MULTIROCKET oracle': 'MULTIROCKET',
                    'HYDRA': 'HYDRA',
                    'HDC-HYDRA auto': 'HYDRA',
                    'HDC-HYDRA oracle': 'HYDRA',
                    'MULTIROCKET-HYDRA': 'MULTIROCKET-HYDRA',
                    'HDC-MULTIROCKET-HYDRA auto': 'MULTIROCKET-HYDRA',
                    'HDC-MULTIROCKET-HYDRA oracle': 'MULTIROCKET-HYDRA',
                     }
    model_mapping = {abbreviations[key]: abbreviations[value] for key, value in model_mapping.items()}

    results = {}

    # read the data
    for folder in model_paths:
        print(f"Model path: {folder}")
        # read the json config file
        with open(os.path.join(folder, "config_log.json"), 'r') as f:
            config = json.load(f)

        model_variant = config.get('model', None).replace('_', '-')
        variant =  config.get('variant', None)
        if 'hdc' in variant:
            model_variant = f"{model_variant}{variant.split('hdc')[1].replace('_', ' ')}"
        model_variant = abbreviations[model_variant]
        if model_variant not in model_columns:
            continue

        dataset = config.get('dataset', None)

        acc_files = sorted(glob.glob(f"{folder}/results_{dataset}_seed_*_acc.xlsx"),
                           key=lambda x: int(re.findall(f"results_{dataset}_seed_(.*)_acc.xlsx", x)[0]))
        print("Read excel files...")
        acc_data, best_scale, best_kernel = [], [], []
        for file in acc_files:
            df = pd.read_excel(file)
            acc_data.append(df.filter(regex='scale_idx|acc_at_best_scale|'+dataset+'_idx'))
            best_scale.append(df.filter(like='best_scale'))
            best_kernel.append(df.filter(like='best_kernel'))

        data_acc = pd.concat(acc_data).groupby(level=0).mean()
        best_scale = pd.concat(best_scale).groupby(level=0).mean()
        best_kernel = pd.concat(best_kernel).groupby(level=0).value_counts().groupby(level=0).idxmax()

        rename_map = {col: col.replace('scale_idx', 'acc_mean_') if 'scale_idx' in col else 'acc_mean_0_auto'
                      for col in data_acc.columns if 'scale_idx' in col or 'acc_at_best_scale' in col}
        data_acc.rename(columns=rename_map, inplace=True)
        data_acc[f"{dataset}_idx"] = data_acc[f"{dataset}_idx"].astype(int)

        acc_cols = [f"acc{col}" for col in model_columns[model_variant]]
        data_acc['acc'] = data_acc[acc_cols].max(axis=1) * 100
        data_acc['acc_seed_0'] = acc_data[0].drop(columns=[f"{dataset}_idx"]).max(axis=1) * 100
        data_acc['mean_best_beta'] = best_scale['best_scale'] if 'auto' in model_variant else 0
        if 'auto' in model_variant:
            data_acc['top_kernel'] = best_kernel

        time_files = glob.glob(f"{folder}/*time.xlsx")
        time_data = [pd.read_excel(f) for f in time_files]
        if time_data:
            data_time = sum(time_data) / len(time_data)
        else:
            data_time = pd.DataFrame({f"{dataset}_idx": [], 'prep_time_test': [], 'inf_time': []})

        print("Compute AUC and PRC...")
        eval_path = os.path.join(folder, "eval_results")
        if (not os.path.exists(f"{eval_path}/auc.pkl") or
                not os.path.exists(f"{eval_path}/auc_indiv.pkl") or force_recompute):
            print("The precomputed files do not exist. Compute them now...")

            # Use full path to run.py
            args = [f"--result_path", folder,
                    "--dataset_idx", "-1"]
            command = [sys.executable, f"{os.path.dirname(os.path.abspath(__file__))}/create_eval_data.py"] + args
            subprocess.run(command)

        data_auc = pd.read_pickle(f"{eval_path}/auc.pkl")
        auc_cols = [f"auc{col}" for col in model_columns[model_variant]]
        prc_cols = [f"prc{col}" for col in model_columns[model_variant]]
        data_cols = data_auc[auc_cols].apply(pd.to_numeric)
        data_auc['auc'] = data_cols.max(axis=1)

        data_auc_indiv = pd.read_pickle(f"{eval_path}/auc_indiv.pkl")
        data_auc['auc_indiv'] = data_auc_indiv[auc_cols].iloc[:, 0:1]
        data_auc['prc'] = data_auc[prc_cols].apply(pd.to_numeric).max(axis=1)
        data_auc['prc_indiv'] = data_auc_indiv[prc_cols].iloc[:, 0:1]

        # combine all data
        data = pd.concat([data_acc, data_auc[['auc', 'prc', 'prc_indiv', 'auc_indiv']]], axis=1)
        data = pd.merge(data, data_auc_indiv, on=f"{dataset}_idx")
        data = pd.merge(data, data_time, on=f"{dataset}_idx")
        data['best_kernel'] = data_cols.idxmax(axis=1).str.split("_").str[-1]
        data['best_scale'] = data_cols.idxmax(axis=1).str.split("_").str[-2]

        results[model_variant] = data
        results[model_variant]['model_path'] = folder

    # all acc from each model
    df_acc = pd.DataFrame(columns=list(results.keys()))
    df_errors = pd.DataFrame(columns=list(results.keys()))
    # fill the dataframe with the acc values
    for model in results:
        df_acc[model] = results[model]['acc'].values
        df_errors[model] = -100 - results[model]['acc'].values

    # compute the acc and auc improvements (relative changes in acc) regarding model_mapping
    print("Compute improvements...")
    for folder in model_mapping:
        # check if folder is part of results
        if folder not in results:
            continue

        results[folder]['improvement_acc'] = (results[folder]['acc'] /
                                              results[model_mapping[folder]]['acc'] - 1)
        mask_acc = np.abs(results[folder]['acc'] - results[model_mapping[folder]]['acc']) < 1
        results[folder].loc[mask_acc, 'improvement_acc'] = 0
        results[folder]['improvement_auc'] = (results[folder]['auc'] /
                                                results[model_mapping[folder]]['auc'] - 1)
        results[folder]['improvement_prc'] = (results[folder]['prc'] /
                                                results[model_mapping[folder]]['prc'] - 1)
        results[folder]['improvement_error'] = (100-results[folder]['acc']) / (100-results[model_mapping[folder]]['acc']) - 1
        # set error improvement to 0 if acc is 100 for both models
        mask_error_100 = (results[folder]['acc'] == 100) & (results[model_mapping[folder]]['acc'] == 100)
        results[folder].loc[mask_error_100, 'improvement_error'] = 0
        # set error improvement to error if orig model is at 100 and the other model is not
        mask_error_partial = (results[folder]['acc'] == 100) & (results[model_mapping[folder]]['acc'] < 100)
        results[folder].loc[mask_error_partial, 'improvement_error'] = (
            (100 - results[folder].loc[mask_error_partial, 'acc']) /
            (100 - results[model_mapping[folder]].loc[mask_error_partial, 'acc']) - 1
        )
        mask = np.abs(results[folder]['acc'] - results[model_mapping[folder]]['acc']) < 1
        results[folder].loc[mask, 'improvement_error'] = 0

        # maximum aucpr improvements
        results[folder]['improvements_pr_max'] = np.zeros(len(results[folder]['prc_indiv']))
        for i in range(len(results[folder]['prc_indiv'])):
            results[folder].loc[results[folder].index[i], 'improvements_pr_max'] = np.max(results[folder]['prc_indiv'].iloc[i][0] / results[model_mapping[folder]]['prc_indiv'].iloc[i][0] - 1)
        # minimum aucpr improvements
        results[folder]['improvements_pr_min'] = np.zeros(len(results[folder]['prc_indiv']))
        for i in range(len(results[folder]['prc_indiv'])):
            results[folder].loc[results[folder].index[i], 'improvements_pr_min'] = np.min(results[folder]['prc_indiv'].iloc[i][0] / results[model_mapping[folder]]['prc_indiv'].iloc[i][0] - 1)

        # normalize improvements (inpt range [0,1])
        results[folder]['improvement_acc_norm'] = results[folder]['improvement_acc'] / np.max(results[folder]['improvement_acc'])
        results[folder]['improvement_auc_norm'] = results[folder]['improvement_auc'] / np.max(results[folder]['improvement_auc'])
        results[folder]['improvement_error_norm'] = results[folder]['improvement_error'] / np.max(results[folder]['improvement_error'])

        # improvements of auc per class
        # read array from df cell
        auc_model_a = results[folder]['auc_indiv'].values
        auc_model_b = results[model_mapping[folder]]['auc_indiv'].values
        # unfold the nested arrays
        individual_auc_improvements = []
        individual_auc_improvements_max = []
        individual_auc_improvements_min = []
        class_with_max_auc_improvement = []
        for i in range(len(auc_model_a)):
            auc_a = np.asarray(auc_model_a[i])
            auc_b = np.asarray(auc_model_b[i])
            individual_auc_improvements.append(auc_a/auc_b - 1)
            individual_auc_improvements_max.append(np.max(auc_a/auc_b - 1))
            individual_auc_improvements_min.append(np.min(auc_a/auc_b - 1))
            class_with_max_auc_improvement.append(np.argmax(auc_a/auc_b - 1))
        # add to dataframe
        results[folder]['improvement_auc_indiv'] = individual_auc_improvements
        results[folder]['improvement_auc_indiv_max'] = individual_auc_improvements_max
        results[folder]['improvement_auc_indiv_min'] = individual_auc_improvements_min
        results[folder]['class_with_max_auc_improvement'] = class_with_max_auc_improvement

    #########################################
    ##### create multi comparison matrix
    #########################################
    print("\nCreate multi comparison matrix...")
    # exclude model that have oracle in name
    exclude_models = [model for model in df_acc.columns if 'oracle' in model]
    # replace column names with abbreviations
    df_acc = df_acc.rename(columns=abbreviations)
    # replace "auto" with "" in column names
    df_acc.columns = [col.replace(' auto', '') for col in df_acc.columns]
    MCM.compare(output_dir=f"{save_path}/",
                df_results=df_acc,
                png_savename=f"mcm_{dataset}_{kernel}",
                excluded_col_comparates=exclude_models,
                excluded_row_comparates=exclude_models,
                load_analysis=False,
                font_size=18)


    #########################################
    #### plot stripplot of improvements
    #########################################
    print("\nCreate stripplot of improvements...")
    stripplot_models = []
    stripplot_models_auc_ovr = []
    improvement_acc = []
    improvement_auc = []
    improvement_error = []
    improvement_auc_ovr = []
    improvement_idx = []

    for model in model_mapping:
        # exclude oracle
        if 'oracle' in model or model not in results or model==model_mapping[model]:
            continue
        stripplot_models.extend([model] * len(results[model]['improvement_acc'].values))
        improvement_acc.extend(results[model]['improvement_acc'].values)
        improvement_auc.extend(results[model]['improvement_auc'].values)
        improvement_error.extend(results[model]['improvement_error'].values)
        improvement_idx.extend(results[model][f"{dataset}_idx"].values)
        auc_ovr = [np.asarray(results[model]['improvement_auc_indiv'][i]).flatten() for i in range(len(results[model]['auc_indiv']))]
        # create dataframe and fill it iteratively
        df = pd.DataFrame(columns=['Idx','class','auc_ovr'])
        for i in range(len(auc_ovr)):
            auc_per_dataset = pd.DataFrame({'Idx': [i] * len(auc_ovr[i]),
                                            'class': np.arange(len(auc_ovr[i])),
                                            'auc_ovr': auc_ovr[i]})
            df = pd.concat([df, auc_per_dataset], axis=0)
        improvement_auc_ovr.extend(df['auc_ovr'].values)
        stripplot_models_auc_ovr.extend([model] * len(df['auc_ovr'].values))


    # plot the stripplot
    stripplot_data = pd.DataFrame({"Model": stripplot_models, "Improvement Accuracy": improvement_acc})
    # Create a stripplot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.stripplot(x="Model", y="Improvement Accuracy", data=stripplot_data, ax=ax,
                  size=7, jitter=True, alpha=0.8)
    # convert y-axis to percentage
    vals = ax.get_yticks()
    ax.set_yticks(vals)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Set plot title and labels
    # plt.title('stripplot of Accuracy Improvement for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Relative ACC Change')
    plt.xticks(rotation=25)  # Rotate x-axis labels for better visibility
    plt.savefig(f"{save_path}/stripplot_improvement_acc_{dataset}_{kernel}.pdf")

    # stripplot of error improvement
    stripplot_data = pd.DataFrame({"Model": stripplot_models, "Improvement Error": improvement_error})
    # Create a stripplot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.stripplot(x="Model", y="Improvement Error", data=stripplot_data, ax=ax,
                  size=7, jitter=True, alpha=0.8)
    # convert y-axis to percentage
    vals = ax.get_yticks()
    ax.set_yticks(vals)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Set plot title and labels
    # plt.title('stripplot of Error Improvement for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Relative Error Change')
    plt.xticks(rotation=25)  # Rotate x-axis labels for better visibility
    plt.savefig(f"{save_path}/stripplot_improvement_error_{dataset}_{kernel}.pdf")

    # count the number of error improvements above 50% for each model individually
    for model in model_mapping:
        if 'oracle' in model or model not in results or model==model_mapping[model]:
            continue
        print(f"Number of error improvements above 50% for {model}: {np.sum(results[model]['improvement_error'] < -0.05)}")

    # count the number of acc improvements above 5% for each model individually
    for model in model_mapping:
        if 'oracle' in model or model not in results or model==model_mapping[model]:
            continue
        print(f"Number of acc improvements above 5% for {model}: {np.sum(results[model]['improvement_acc'] > 0.05)}")

    #########################################
    ##### plot figures of top candidates
    #########################################
    print("\nCreate figures of top candidates...")
    if 1:
        for model in model_mapping:
            if model not in results:
                continue

            # visualize the top k auc improvements
            k = 3
            # get the top k indices
            top_k_idx_acc = results[model].sort_values(by='improvement_acc', ascending=False).head(k)[
                f"{dataset}_idx"].values
            # Get top k indices based on error reduction (ascending order means smallest errors first)
            top_k_idx_err = results[model].sort_values(by='improvement_error', ascending=True).head(k)[
                f"{dataset}_idx"].values
            # Combine the top indices from both lists
            combined_top_k = np.concatenate([top_k_idx_acc, top_k_idx_err])
            # Get unique indices and ensure there are at least 6 unique ones
            top_k_idx = np.unique(combined_top_k)
            # If we have less than 6 unique indices, add more from the error list
            if len(top_k_idx) < 2 * k:
                additional_indices_needed = 2 * k - len(top_k_idx)
                additional_top_k_idx_err = results[model].sort_values(by='improvement_error', ascending=True).iloc[k:][
                    f"{dataset}_idx"].values
                # Append additional unique indices until we have enough
                for idx in additional_top_k_idx_err:
                    if idx not in top_k_idx:
                        top_k_idx = np.append(top_k_idx, idx)
                    if len(top_k_idx) == 2 * k:
                        break
            thresh = 0.01

            ########################
            # scatter plot the AUPRC data class wise in one figure for model a and b
            ########################
            # figure for plotting aucprc values class wise
            prc_model_a = {}
            prc_model_b = {}
            fig_prc, axs_prc = plt.subplots(2, k, figsize=(k * 4, 8))
            for i, k_idx in enumerate(top_k_idx):
                prc_model_a[k_idx] = np.asarray(
                    results[model].loc[results[model][f'{dataset}_idx'] == k_idx, 'prc_indiv'].values[0])
                prc_model_b[k_idx] = np.asarray(results[model_mapping[model]].loc[results[model_mapping[model]][
                                                                                      f'{dataset}_idx'] == k_idx, 'prc_indiv'].values[
                                                    0])

                dataset_name = eval(f"{dataset}_PREFIX[{k_idx}]")
                ax_prc = axs_prc[i // k, i % k]
                # Scatterplot where model a is on x-axis and model b on y-axis
                prc_b = prc_model_a[k_idx]
                prc_a = prc_model_b[k_idx]
                ax_prc.scatter(prc_a, prc_b, label='AUCPRC', color=modern_palette[0], edgecolors=modern_palette[0], alpha=0.9, s=100)
                min_val = np.min([prc_a, prc_b])
                max_val = np.max([prc_a, prc_b])
                # Ensure at least 0.2 difference between min and max value
                if max_val - min_val < 0.2:
                    max_val += 0.1
                    min_val -= 0.1
                ax_prc.set_ylim(min_val - 0.01, max_val + 0.01)
                ax_prc.set_xlim(min_val - 0.01, max_val + 0.01)
                ax_prc.set_ylabel(f"AUPRC {model.replace(' auto','')}")
                ax_prc.set_xlabel(f"AUPRC {model_mapping[model]}")
                ax_prc.set_title(f"ID {k_idx} {dataset_name}")
                ax_prc.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax_prc.legend()
                # two digits after comma for ticks
                ax_prc.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # Equal aspect ratio
                ax_prc.set_aspect('equal', adjustable='box')
                # Plot the diagonal
                ax_prc.plot([0, 1], [0, 1], transform=ax_prc.transAxes, ls="--", c=".3")

            # Increase font size of tick labels and adjust layout
            for ax in axs_prc.flat:
                ax.tick_params(axis='both')
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fs)

            plt.tight_layout()
            plt.savefig(f"{save_path}/aucprc_{model.replace(' ', '_')}_{kernel}.pdf")


            #########################################
            # generate table with the top k indices and the corresponding best scale (beta)
            #########################################
            if 'auto' in model:
                table_scales = pd.DataFrame()
                # get indices of datasets that improved more than 0.05
                # top_k_idx = results[model].loc[results[model]['improvement_acc'] > 0.05].sort_values(by='improvement_acc', ascending=False)[f"{dataset}_idx"].values
                for idx in top_k_idx:
                    mean_scale = results[model].loc[results[model][f"{dataset}_idx"] == idx, 'mean_best_beta'].values[0]
                    acc_improvement = results[model].loc[results[model][f"{dataset}_idx"] == idx, 'improvement_acc'].values[0]*100
                    err_decrease = results[model].loc[results[model][f"{dataset}_idx"] == idx, 'improvement_error'].values[0]*100
                    mean_acc = results[model].loc[results[model][f"{dataset}_idx"] == idx, 'acc'].values[0]
                    orig_acc = results[model_mapping[model]].loc[results[model_mapping[model]][f"{dataset}_idx"] == idx, 'acc'].values[0]
                    max_pr_improvement = results[model].loc[results[model][f"{dataset}_idx"] == idx, 'improvements_pr_max'].values[0]*100
                    table_scales = pd.concat([table_scales, pd.DataFrame({'Idx': [idx], 'mean $\\beta$': [mean_scale],
                                                                          'mean Acc with HDC': [mean_acc],
                                                                          'mean Acc without HDC': [orig_acc],
                                                                          'Acc improvement [\\%]': [acc_improvement],
                                                                          'Error decrease [\\%]': [err_decrease],
                                                                          'Max indiv. AUCPR improvement [\\%]': [max_pr_improvement],
                                                                          })], axis=0)
                # save table as latex
                table_scales = table_scales.round(2)
                table_scales = table_scales.applymap(lambda x: str(x))
                print(table_scales.to_latex(f"{save_path_tables}/scales_table_{model.replace(' ','_')}_{kernel}.tex", index=False))
                # save table as excel file
                table_scales.to_excel(f"{save_path_tables}/scales_table_{model.replace(' ','_')}_{kernel}.xlsx", index=False)

    ##############################
    # create latex table
    ##############################
    print("\nCreate latex table...")
    # get mean and std of the improvements
    kernel_name = {}
    mean_improvement = {}
    meadian_improvement = {}
    mean_auc_improvement = {}
    max_indiv_prc_improv = {}
    min_indiv_prc_improv = {}
    min_improvement = {}
    max_improvement = {}
    improvement_10_percentile = {}
    improvement_90_percentile = {}
    mean_error_improvement = {}
    min_error_improvement = {}
    max_error_improvement = {}
    mean_acc = {}
    mean_auc = {}
    mean_prc = {}
    worst_acc = {}
    num_ds = {}


    for folder in model_mapping:
        if folder not in results or 'oracle' in folder:
            continue
        kernel_name[folder] = kernel
        if only_best:
            thresh = 0.05
            # compare thresh with absolute values of improvement for error and accuracy (or connection)
            indices = (pd.Series(results[folder]['improvement_error']).abs() > thresh) | (pd.Series(results[folder]['improvement_acc']).abs() > thresh)
            if np.sum(indices) < 1:
                # find the value in model_mapping where they key is equal to the folder
                key = [key for key, value in model_mapping.items() if value == folder and value!=key][0]
                try:
                    indices = (pd.Series(results[key]['improvement_error']).abs() > thresh) | (pd.Series(results[key]['improvement_acc']).abs() > thresh)
                except:
                    continue
        else:
            indices = np.ones(len(results[folder]['improvement_acc']), dtype=bool)
        mean_improvement[folder] = results[folder]['improvement_acc'][indices].mean().round(2)*100
        mean_auc_improvement[folder] = results[folder]['improvement_auc'][indices].mean()
        min_improvement[folder] = (results[folder]['improvement_acc'][indices].min() * 100).round(2)
        max_improvement[folder] = (results[folder]['improvement_acc'][indices].max() * 100).round(2)
        max_indiv_prc_improv[folder] = (results[folder]['improvements_pr_max'][indices].max() * 100).round(2)
        min_indiv_prc_improv[folder] = (results[folder]['improvements_pr_min'][indices].min() * 100).round(2)
        improvement_10_percentile[folder] = np.percentile(results[folder]['improvement_auc'][indices], 10)
        improvement_90_percentile[folder] = np.percentile(results[folder]['improvement_auc'][indices], 90)
        mean_error_improvement[folder] = results[folder]['improvement_error'][indices].mean().round(2)*100
        min_error_improvement[folder] = results[folder]['improvement_error'][indices].min().round(2)*100
        max_error_improvement[folder] = results[folder]['improvement_error'][indices].max().round(2)*100
        mean_acc[folder] = (results[folder]['acc'][indices]).mean().round(2)
        mean_auc[folder] = (results[folder]['auc'][indices]).mean().round(4)
        mean_prc[folder] = (results[folder]['prc'][indices]).mean().round(4)
        worst_acc[folder] = (results[folder]['acc'][indices]).min().round(2)
        num_ds[folder] = np.sum(indices)
    model_names = list(mean_improvement.keys())
    # replace auto with ''
    model_names = [model.replace(' auto', '') for model in model_names]
    # pandas table
    table = pd.DataFrame({'Model': model_names,
                            'Num DS': list(num_ds.values()),
                            'Mean Acc': list(mean_acc.values()),
                            'Mean AUPRC': list(mean_prc.values()),
                            'Worst Acc': list(worst_acc.values()),
                            'Mean Acc improvement': list(mean_improvement.values()),
                            'Max Acc \n improvement': list(max_improvement.values()),
                            'Mean Error decrease': list(mean_error_improvement.values()),
                            'Min Error \n decrease': list(min_error_improvement.values()),
                          })
    # table = table.round(2)
    table = table.applymap(lambda x: str(x))
    print(table.to_latex(f"{save_path_tables}/result_table_{dataset}_{kernel}.tex", index=False))
    # save table as excel file
    table.to_excel(f"{save_path_tables}/result_table_{dataset}_{kernel}.xlsx", index=False)

    ### plot histogram of the used kernel if all kernels are used
    if kernel == 'all_kernels':
        print("\nCreate histogram of used kernels...")
        for model in model_mapping:
            if model=='HDC-MiniROCKET auto' or model=='HDC-MultiROCKET-HYDRA auto':
                kernels = results[model]['top_kernel'].values
                # use only kernels where scale is not 0
                data = [(kernels[i], results[model]['improvement_acc'].values[i]) for i in range(len(kernels))]
                used_kernels = [k[0][1] for k in data]
                acc_improvement = [k[1]*100 for k in data]
                scale = [k[1] for k in data]
                # plot histogram from list of strings
                fig = plt.figure(figsize=(10, 6))
                sns.histplot(used_kernels, bins=3, discrete=True)
                plt.title(f"Used Kernels for {model}")
                plt.xlabel('Kernel')
                plt.ylabel('Count')
                plt.savefig(f"{save_path}/histogram_kernels_{model.replace(' ','_')}_{kernel}.pdf")

                # make multiple histograms of kernel choices with respect to their rel. improvements
                fig = plt.figure(figsize=(6, 4))
                # Create a DataFrame for easy plotting with Seaborn
                df = pd.DataFrame({
                    'Kernel': used_kernels,
                    'Accuracy Improvement': acc_improvement,
                    'Scale': scale
                })

                # Set font sizes for the plot
                plt.rcParams.update({'font.size': 18})  # Increases font size globally
                plt.rcParams.update({'axes.labelsize': 16})  # Axis labels
                plt.rcParams.update({'xtick.labelsize': 14})  # X-axis tick labels
                plt.rcParams.update({'ytick.labelsize': 14})  # Y-axis tick labels
                plt.rcParams.update({'legend.fontsize': 14})  # Legend text
                plt.rcParams.update({'axes.titlesize': 18})  # Plot title

                # Remove all zero improvements
                df = df[df['Scale'] != 0]
                # replace gaussian with Gaussian
                df['Kernel'] = df['Kernel'].replace('gaussian', 'Gaussian')

                # define bar colors for sinc, gaussian and triangular
                colors = {'sinc': '#842e22', 'Gaussian': '#339999', 'triangular': '#e6b800'}

                # Plot multiple histograms, grouped by 'Kernel' with side-by-side bars
                sns.histplot(data=df, x='Accuracy Improvement', hue='Kernel', multiple='dodge', bins=20, kde=False, palette=colors)

                # Add labels and title
                plt.xlabel('Relative ACC change', fontsize=16)
                plt.ylabel('Frequency', fontsize=16)
                # plt.title('Histogram of Accuracy Improvement for Different Kernels')  # Uncomment if a title is needed

                # Save the plot
                plt.savefig(f"{save_path}/histogram_kernels_{model.replace(' ', '_')}_{kernel}.pdf")

    ##########
    # table with oracle results
    ##########
    print("\nCreate latex table with oracle results...")
    models = []
    beta_0 = []
    beta_1 = []
    beta_2 = []
    beta_3 = []
    beta_4 = []
    beta_5 = []
    beta_6 = []
    oracle = []
    orig_seed_0 = []

    for model in model_mapping:
        if model not in results:
            continue
        if 'oracle' in model:
            models.append(model.replace('oracle', ''))
            beta_0.append(results[model][f"acc_mean_0_{kernel}"].mean()*100)
            beta_1.append(results[model][f"acc_mean_1_{kernel}"].mean()*100)
            beta_2.append(results[model][f"acc_mean_2_{kernel}"].mean()*100)
            beta_3.append(results[model][f"acc_mean_3_{kernel}"].mean()*100)
            beta_4.append(results[model][f"acc_mean_4_{kernel}"].mean()*100)
            beta_5.append(results[model][f"acc_mean_5_{kernel}"].mean()*100)
            beta_6.append(results[model][f"acc_mean_6_{kernel}"].mean()*100)
            oracle.append(results[model]['acc'].mean())
            orig_seed_0.append(results[model_mapping[model]]['acc_seed_0'].mean())

    # pandas table
    table = pd.DataFrame({'Model': models,
                            '$\\beta_0$': beta_0,
                            '$\\beta_1$': beta_1,
                            '$\\beta_2$': beta_2,
                            '$\\beta_3$': beta_3,
                            '$\\beta_4$': beta_4,
                            '$\\beta_5$': beta_5,
                            '$\\beta_6$': beta_6,
                            'Oracle': oracle,
                            'Orig Encoder': orig_seed_0
                          })
    table = table.round(2)
    table = table.applymap(lambda x: str(x))
    print(table.to_latex(f"{save_path_tables}/result_table_oracle_{dataset}_{kernel}.tex", index=False))

    ####################
    # plot critical difference diagram
    ####################
    print("\nCreate critical difference diagram...")

    # create a new datafram with 3 columns: model_name, idx, acc
    df = pd.DataFrame(columns=['classifier_name', 'dataset_name', 'accuracy'])
    for model in results:
        df = pd.concat([df, pd.DataFrame({'classifier_name': model,
                                              'dataset_name': results[model][f"{dataset}_idx"],
                                              'accuracy': results[model]['acc']})], axis=0)
    # delete oracle models for statistical testing
    df = df[~df['classifier_name'].str.contains('oracle')]
    df = df[~(df['classifier_name'] == 'HDC-MultiROCKET auto')]
    df = df[~(df['classifier_name'] == 'MultiROCKET')]
    # df = df[~(df['classifier_name'] == 'HDC-HYDRA auto')]
    # df = df[~(df['classifier_name'] == 'HYDRA')]

    # change classifier names
    df['classifier_name'] = df['classifier_name'].replace('HDC-MiniROCKET auto', 'HDC-MiniROCKET')
    df['classifier_name'] = df['classifier_name'].replace('HDC-MultiROCKET-HYDRA auto', 'HDC-MultiROCKET-HYDRA')

    draw_cd_diagram(df_perf=df, title='Accuracy', labels=True)
    # move the cd diagram to the images folder
    os.system(f"mv cd-diagram.png {save_path}/cd_diagram_{dataset}_{kernel}.png")

    ########################
    ## save all results of all models in latex table
    ########################
    all_model_names = model_mapping.keys()
    # exclude names whit oracle in it
    all_model_names = [model for model in all_model_names if 'oracle' not in model]
    # create dataframe with dataset id, dataset name and all model names as columns
    overall_df = pd.DataFrame(columns=['Dataset ID', 'Dataset Name'] + list(all_model_names))
    for model in all_model_names:
        if model not in results:
            continue
        acc = results[model]['acc']
        # add one more value to the series
        acc = acc._append(pd.Series([np.mean(acc)], index=['mean']))
        overall_df[model] = acc
    overall_df['Dataset ID'] = [idx for idx in range(len(overall_df)-1)] + ['']
    overall_df['Dataset Name'] = [eval(f"{dataset}_PREFIX[{idx}]").replace('_',' ') for idx in range(len(overall_df)-1)] + ['mean']
    # Apply the highlighting
    overall_df = overall_df.round(2)
    # convert the highest acc in each row to bold
    for i in range(len(overall_df)):
        # max idx for columns from model name dict
        max_val = overall_df.iloc[i, 2:].values.max()
        max_idx = np.where(overall_df.iloc[i, :].values == max_val)[0]
        for idx in max_idx:
            overall_df.iloc[i, idx] = f"\\textbf{{{overall_df.iloc[i, idx]}}}"
    overall_df = overall_df.applymap(lambda x: str(x))
    # save as latex table
    print(overall_df.to_latex(f"{save_path_tables}/overall_table_{dataset}_{kernel}.tex", index=False))

