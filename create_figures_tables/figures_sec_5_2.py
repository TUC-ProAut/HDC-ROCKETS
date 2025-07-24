import os
import pandas as pd
from numpy.distutils.command.config import config

pd.DataFrame.iteritems = pd.DataFrame.items # for compatibility with older pandas versions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import roc_auc_score
import matplotlib.patches as mpatches
import json

from plot_config import mpl

force_recompute = False

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# define the folder paths for models
global_path = "../results/synthetic"
# path to the results folder of the HDC models
save_path = os.path.abspath(f".")


kernel = 'sinc'
scale = 0
n_scales = 7

abbreviations = {'MINIROCKET': 'MiniROCKET',
                 'HDC-MINIROCKET auto': 'HDC-MiniROCKET',
                 'HDC-MINIROCKET oracle': 'HDC-MiniROCKET oracle',
                 'MULTIROCKET': 'MultiROCKET',
                 'HDC-MULTIROCKET auto': 'HDC-MultiROCKET',
                 'HDC-MULTIROCKET oracle': 'HDC-MultiROCKET oracle',
                 'HYDRA': 'HYDRA',
                 'HDC-HYDRA auto': 'HDC-HYDRA',
                 'HDC-HYDRA oracle': 'HDC-HYDRA oracle',
                 'MULTIROCKET-HYDRA': 'MultiROCKET-HYDRA',
                 'HDC-MULTIROCKET-HYDRA auto': 'HDC-MultiROCKET-HYDRA',
                 'HDC-MULTIROCKET-HYDRA oracle': 'HDC-MultiROCKET-HYDRA oracle',
                 }

model_paths = []
# === Load and process model result logs ===
for folder in os.listdir(global_path):
    dir = os.path.join(global_path, folder)
    # check if dir is a directory and does not start with "_" (symbol for not included in paper)
    if os.path.isdir(dir) and not folder.startswith('_'):
        print(f"Processing model folder: {folder}")
        model_paths.append(dir)

df = pd.DataFrame(columns=['model', 'dataset', 'acc'])

# read the data
for folder in model_paths:
    # read the json config file
    with open(os.path.join(folder, "config_log.json"), 'r') as f:
        config = json.load(f)

    model_variant = config.get('model', None).replace('_', '-')
    variant = config.get('variant', None)
    if 'hdc' in variant:
        model_variant = f"{model_variant}{variant.split('hdc')[1].replace('_', ' ')}"
    model_variant = abbreviations[model_variant]

    dataset = config.get('dataset', None) + '_hard_case_' + str(config.get('hard_case'))

    # Extract dataset and accuracy from log
    # find key with "acc" in it
    acc_key = next((key for key in config.keys() if 'acc' in key), None)
    acc = float(config.get(acc_key))*100

    print(f"Appended results for model: {model_variant}, dataset: {dataset}, ACC: {acc:.2f}")
    df.loc[len(df)] = [model_variant, dataset, acc]

metrics = ['acc']

# Get colors from plot config
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
# Define color palette mapping model names to colors
palette = {
    'MiniROCKET': colors[0],
    'HDC-MiniROCKET': colors[0],
    'MULTIROCKET': colors[1],
    'HDC-MULTIROCKET': colors[1],
    'HYDRA': colors[2],
    'HDC-HYDRA': colors[2],
    'MULTIROCKET+HYDRA': colors[3],
    'HDC-MULTIROCKET+HDC-HYDRA': colors[3],
}

# Define hatching patterns for each model
hatches = {
    'MiniROCKET': '',
    'HDC-MiniROCKET': '/',
    'MULTIROCKET': '',
    'HDC-MULTIROCKET': '/',
    'HYDRA': '',
    'HDC-HYDRA': '/',
    'MULTIROCKET+HYDRA': '',
    'HDC-MULTIROCKET+HDC-HYDRA': '/',
}
# rename synthetic datasets names in the dataframe
df['dataset'] = df['dataset'].replace({
    'synthetic1_hard_case_False': 'Synthetic one peak',
    'synthetic1_hard_case_True': 'Synthetic one peak hard',
    'synthetic2_hard_case_False': 'Synthetic two peaks',
    'synthetic2_hard_case_True': 'Synthetic two peaks hard',
                                      })
sorted_model = ['MiniROCKET', 'HDC-MiniROCKET', 'MULTIROCKET', 'HDC-MULTIROCKET', 'HYDRA', 'HDC-HYDRA', 'MULTIROCKET+HYDRA', 'HDC-MULTIROCKET+HDC-HYDRA']
sorted_dataset = ['Synthetic one peak', 'Synthetic one peak hard', 'Synthetic two peaks', 'Synthetic two peaks hard']

# Loop over selected metrics (e.g., ACC)

# table for metric
metric='acc'
df_table = df.pivot(index='model', columns='dataset', values=metric)
# convert the first header (pivot) to normal header
df_table.columns = df_table.columns.get_level_values(0)
df_table = df_table.round(2)
df_table = df_table.applymap(lambda x: str(x))
# uppder case the first letter of each word
df_table.columns = [col.title() for col in df_table.columns]
df_table.columns = [col.replace(' ', '\n') for col in df_table.columns]
# set the name for the first column
df_table.index.name = 'Model'

print(f"Saving LaTeX table to: {save_path}/tables/synthetic_results_{metric}.tex")
df_table.to_latex(f"{save_path}/tables/synthetic_results_{metric}.tex")

# === Create tables and bar plots ===
fig, ax = plt.subplots()

# Sort models and datasets to maintain consistent plotting order
ordered_df = pd.DataFrame(columns=['model', 'dataset', metric])
for model in sorted_model:
    # check if model is in df
    if model not in df['model'].values:
        continue
    for dataset in sorted_dataset:
        ordered_df.loc[len(ordered_df)] = [model, dataset, df[(df['model'] == model) & (df['dataset'] == dataset)][metric].values[0]]
df = ordered_df

bars = sns.barplot(data=df, x='dataset', y=metric, hue='model', orient='v', ax=ax, palette=palette)
# Apply hatching patterns
for bar, model in zip(bars.patches, df['model']):
    bar.set_hatch(hatches[model])

# sort the bars by sorted_dataset
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels())

# Create custom legend handles
handles = [
    mpatches.Patch(facecolor=palette[model], hatch=hatches[model], label=model)
    for model in sorted_model
]
# Adjusting legend so that it does not overlap with the plot
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)

# Display the metric on each bar
for p in ax.patches:
    height = p.get_height()  # get bar length
    if height == 0:
        continue
    ax.text(p.get_x() + p.get_width() / 2., int(height), '{:1.0f}'.format(height),
            ha='center', va='bottom')
# Set the titles and labels
metric_name = 'Accuracy' if metric == 'acc' else 'AUC'
# ax.set_title(f"{metric_name} of the models on the synthetic datasets", fontsize=fs+2)
ax.set_ylabel(f"{metric_name}")
ax.set_ylabel('')
# plt.tight_layout()
# make space between legend and x label
plt.subplots_adjust(bottom=0.5)
# shift legend down
plt.ylabel("ACC")

print(f"Saving bar plot to: {save_path}/images/bar_plot_synthetic_results_{metric}.pdf")
plt.savefig(f"{save_path}/images/bar_plot_synthetic_results_{metric}.pdf", format='pdf')
plt.show()
