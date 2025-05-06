import os
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import json

from plot_config import mpl
from data.constants import UCR_NEW_PREFIX

pd.DataFrame.iteritems = pd.DataFrame.items # for compatibility with older pandas versions
force_recompute = False

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# define the folder paths for models
fs = 20

# path to the results folder of the original models
source = "/Users/scken/Nextcloud/project-users/scken/Experimental_Data/HDC-Rockets/results_journal_paper/UCR_NEW/_time_measure"

# path to the results folder of the HDC models
save_path = os.path.abspath(f"images/")
scale = 0
n_scales = 7
kernels = ['sinc']

# read all folders from orig and hdc path
model_paths = []
for folder in os.listdir(source):
    if folder=='trash':
        continue
    dir = os.path.join(source, folder)
    # check if dir is a directory and does not start with "_" (symbol for not included in paper)
    if os.path.isdir(dir) and not folder.startswith('_'):
        model_paths.append(dir)


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
    variant = config.get('variant', None)
    if 'hdc' in variant:
        model_variant = f"{model_variant}{variant.split('hdc')[1].replace('_', ' ')}"
    model_variant = abbreviations[model_variant]
    if model_variant not in model_columns:
        continue

    dataset = config.get('dataset', None)

    # read computational time data
    # find all files ending with time.xlsx
    time_files = glob.glob(f"{folder}/*time.xlsx")
    # read all files and average the time
    time_data = []
    for file in time_files:
        time_data.append(pd.read_excel(file))
    data_time = time_data[0]
    for i in range(1, len(time_data)):
        data_time = data_time + time_data[i]
    data_time = data_time / len(time_data)
    # check if data time is empty
    if data_time.empty:
        data_time = pd.DataFrame(columns=[f"{dataset}_idx", 'prep_time_test', 'inf_time'])
        data_time['prep_time_test'] = 0
        data_time['inf_time'] = 0

    results[model_variant] = data_time

    # add path to model
    results[model_variant]['model_path'] = folder



################################
# plot overall time comparison
################################
print("\nCreate overall time comparison...")
overall_time_fig, overall_time_ax = plt.subplots(1, 1, figsize=(8, 8))
models_to_plot = ['HDC-MiniROCKET auto', 'HDC-MultiROCKET auto', 'HDC-HYDRA auto']
max_val = 0
min_val = 100
min_ratio = 0
max_ratio = 0

for model in model_mapping:
    # Check if the model has to be plotted
    if model not in models_to_plot or model not in results:
        continue
    preproc_time_model_b = results[model]['prep_time_test'].values
    preproc_time_model_a = results[model_mapping[model]]['prep_time_test'].values
    inference_time_model_b = results[model]['inf_time'].values
    inference_time_model_a = results[model_mapping[model]]['inf_time'].values
    # Plot in overall figure
    overall_time_ax.scatter(preproc_time_model_a + inference_time_model_a,
                            preproc_time_model_b + inference_time_model_b,
                            label=f"{model.replace(' auto', '')} vs. {model_mapping[model]}")
    min_val = np.minimum(
        np.min([preproc_time_model_a + inference_time_model_a, preproc_time_model_b + inference_time_model_b]),
        min_val)
    max_val = np.maximum(
        np.max([preproc_time_model_a + inference_time_model_a, preproc_time_model_b + inference_time_model_b]),
        max_val)
    ratio = (preproc_time_model_a + inference_time_model_a) / (preproc_time_model_b + inference_time_model_b)
    max_ratio = np.maximum(np.max(ratio), max_ratio)
    min_ratio = np.minimum(np.min(ratio), min_ratio)

# Log scale
overall_time_ax.set_xscale('log')
overall_time_ax.set_yscale('log')
# Plot the correct diagonal line
overall_time_ax.plot([min_val, max_val], [min_val, max_val], ls="--", c=".3")
# Plot overall time comparison
overall_time_ax.set_ylabel("Inference time HDC-ROCKET models [s]", fontsize=fs)
overall_time_ax.set_xlabel("Inference time ROCKET models[s]", fontsize=fs)
# Add names for each diagonal (one half named by "model a is faster" and the other half by "model b is faster")
overall_time_ax.text(0.5, 0.05, "HDC Models are faster", horizontalalignment='center', verticalalignment='center',
                     transform=overall_time_ax.transAxes, fontsize=fs)
overall_time_ax.text(0.05, 0.5, "Original Models are faster", horizontalalignment='center', verticalalignment='center',
                     transform=overall_time_ax.transAxes, rotation=90, fontsize=fs)
# Increase font size of figure
for item in ([overall_time_ax.title, overall_time_ax.xaxis.label, overall_time_ax.yaxis.label] +
             overall_time_ax.get_xticklabels() + overall_time_ax.get_yticklabels()):
    item.set_fontsize(fs)
overall_time_ax.legend(fontsize=fs)
# Set correct limits
overall_time_ax.set_ylim(min_val - 0.01, max_val + max_val * 0.1)
overall_time_ax.set_xlim(min_val - 0.01, max_val + max_val * 0.1)
# Save overall_time_fig as PDF
plt.savefig(f"{save_path}/time_scatterplot_overall.pdf")
print(f"Min ratio: {min_ratio}, Max ratio: {max_ratio}")

################################
# plot overall training time comparison
################################
print("\nCreate overall training time comparison...")
overall_train_time_fig, overall_train_time_ax = plt.subplots(1, 1, figsize=(8, 8))
max_val_train = 0
min_val_train = 100
min_ratio_train = 0
max_ratio_train = 0

for model in model_mapping:
    if model not in models_to_plot or model not in results:
        continue
    if model_mapping[model] not in results:
        continue

    try:
        train_time_model_b = (
            results[model]['prep_time_train'] +
            results[model]['train_time']
        ).values
        train_time_model_a = (
            results[model_mapping[model]]['prep_time_train'] +
            results[model_mapping[model]]['train_time']
        ).values
    except KeyError:
        continue

    overall_train_time_ax.scatter(train_time_model_a, train_time_model_b,
                                  label=f"{model} vs. {model_mapping[model]}")
    min_val_train = np.minimum(np.min([train_time_model_a, train_time_model_b]), min_val_train)
    max_val_train = np.maximum(np.max([train_time_model_a, train_time_model_b]), max_val_train)
    ratio = train_time_model_a / train_time_model_b
    max_ratio_train = np.maximum(np.max(ratio), max_ratio_train)
    min_ratio_train = np.minimum(np.min(ratio), min_ratio_train)

overall_train_time_ax.set_xscale('log')
overall_train_time_ax.set_yscale('log')
overall_train_time_ax.plot([min_val_train, max_val_train], [min_val_train, max_val_train], ls="--", c=".3")
overall_train_time_ax.set_ylabel("Training time HDC-ROCKET models [s]", fontsize=fs)
overall_train_time_ax.set_xlabel("Training time ROCKET models [s]", fontsize=fs)
overall_train_time_ax.text(0.5, 0.05, "HDC Models are faster", horizontalalignment='center', verticalalignment='center',
                            transform=overall_train_time_ax.transAxes, fontsize=fs)
overall_train_time_ax.text(0.05, 0.5, "Original Models are faster", horizontalalignment='center', verticalalignment='center',
                            transform=overall_train_time_ax.transAxes, rotation=90, fontsize=fs)
for item in ([overall_train_time_ax.title, overall_train_time_ax.xaxis.label, overall_train_time_ax.yaxis.label] +
             overall_train_time_ax.get_xticklabels() + overall_train_time_ax.get_yticklabels()):
    item.set_fontsize(fs)
overall_train_time_ax.legend(fontsize=fs)
overall_train_time_ax.set_ylim(min_val_train - 0.01, max_val_train + max_val_train * 0.1)
overall_train_time_ax.set_xlim(min_val_train - 0.01, max_val_train + max_val_train * 0.1)
plt.savefig(f"{save_path}/time_scatterplot_overall_train.pdf")
print(f"Min training time ratio: {min_ratio_train}, Max training time ratio: {max_ratio_train}")


##################################
# scalability comparison
##################################
fs = 24
models_to_plot = ['MiniROCKET', 'MultiROCKET', 'HYDRA',
                    'HDC-MiniROCKET auto', 'HDC-MultiROCKET auto', 'HDC-HYDRA auto']
# load the dataset infos from csv
data_info = pd.read_csv(os.path.expanduser('dataset_shapes_UCR_NEW.csv'))


# create a plot: training size vs inference time (one plot per pair)
print("\nCreate scalability plot...")

model_pairs = [
    ('MiniROCKET', 'HDC-MiniROCKET auto'),
    ('MultiROCKET', 'HDC-MultiROCKET auto'),
    ('HYDRA', 'HDC-HYDRA auto'),
]

for base_model, hdc_model in model_pairs:
    scalability_fig, scalability_ax = plt.subplots(1, 1, figsize=(8, 8))
    fit_coeffs = {}
    for model in [base_model, hdc_model]:
        if model not in results:
            continue
        df = results[model]
        merged = pd.merge(df, data_info, left_on=f"UCR_NEW_idx", right_on="dataset_idx", how="inner")
        x = merged['n_train_samples'] * merged['n_train_steps']
        y = merged['prep_time_train'] + merged['train_time'] + merged['grid_search_time']
        scatter = scalability_ax.scatter(x, y, label=model.replace(" auto", ""))
        coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
        fit_coeffs[model] = coeffs
        fit_x = np.linspace(min(x), max(x), 100)
        fit_y = 10 ** (coeffs[0] * np.log10(fit_x) + coeffs[1])
        try:
            color = scatter.get_facecolor()[0]
            scalability_ax.plot(fit_x, fit_y, linestyle='-', color=color, linewidth=3)
        except Exception:
            scalability_ax.plot(fit_x, fit_y, linestyle='-', linewidth=3)
    ratio = 10 ** (fit_coeffs[hdc_model][1] - fit_coeffs[base_model][1])
    speed_label = f"{ratio:.1f}× {'faster' if ratio < 1 else 'slower'}"
    # Compute annotation point at median x
    merged = pd.merge(results[base_model], data_info, left_on=f"UCR_NEW_idx", right_on="dataset_idx", how="inner")
    x_vals = merged['n_train_samples'] * merged['n_train_steps']
    x_median = np.median(x_vals)
    y_base = 10 ** (fit_coeffs[base_model][0] * np.log10(x_median) + fit_coeffs[base_model][1])
    y_hdc = 10 ** (fit_coeffs[hdc_model][0] * np.log10(x_median) + fit_coeffs[hdc_model][1])
    # Draw arrow annotation
    scalability_ax.annotate(
        speed_label,
        xy=(x_median, y_base), xytext=(x_median, y_hdc),
        arrowprops=dict(arrowstyle="->", lw=2, color='black'),
        fontsize=fs,
        ha='center'
    )
    scalability_ax.set_xscale('log')
    scalability_ax.set_yscale('log')
    scalability_ax.set_xlabel('Training Size [samples * series length]', fontsize=fs)
    scalability_ax.set_ylabel('Total Training Time [s]', fontsize=fs)
    scalability_ax.legend(fontsize=fs)
    scalability_ax.grid(True, which="both", ls="--")
    scalability_ax.tick_params(axis='both', labelsize=fs)
    plt.savefig(f"{save_path}/time_scalability_plot_{base_model}_vs_{hdc_model}.pdf")
print("Saved scalability plots for each model pair.")


# scalability for testing
print("\nCreate scalability plot for testing...")

scalability_test_fig, scalability_test_ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot each model's testing time over test size
for model in models_to_plot:
    if model not in results:
        continue
    df = results[model]

    # Merge time info with dataset info based on dataset_idx
    merged = pd.merge(df, data_info, left_on=f"UCR_NEW_idx", right_on="dataset_idx", how="inner")

    # x = number of test samples * number of steps
    x = merged['n_test_samples'] * merged['n_test_steps']
    # y = total test time (prep + inference)
    y = merged['prep_time_test'] + merged['inf_time']

    # Scatter plot for the model
    scatter = scalability_test_ax.scatter(x, y)

    # Fit and plot a linear regression line
    coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
    fit_x = np.linspace(min(x), max(x), 100)
    fit_y = 10 ** (coeffs[0] * np.log10(fit_x) + coeffs[1])
    color = None
    try:
        color = scatter.get_facecolor()[0]
    except Exception:
        pass
    if color is not None:
        scalability_test_ax.plot(fit_x, fit_y, label=model.replace(" auto", ""), linestyle='-', color=color, linewidth=3)
    else:
        scalability_test_ax.plot(fit_x, fit_y, label=model.replace(" auto", ""), linestyle='-', linewidth=3)

# Log scale for clarity
scalability_test_ax.set_xscale('log')
scalability_test_ax.set_yscale('log')

scalability_test_ax.set_xlabel('Test Size [samples * series length]', fontsize=fs)
scalability_test_ax.set_ylabel('Total Test Time [s]', fontsize=fs)
scalability_test_ax.legend(fontsize=fs)
scalability_test_ax.grid(True, which="both", ls="--")
scalability_test_ax.tick_params(axis='both', labelsize=fs)

# Save scalability test plot
plt.savefig(f"{save_path}/time_scalability_plot_test.pdf")
print("Saved scalability test plot.")
