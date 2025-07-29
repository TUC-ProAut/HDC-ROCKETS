<div align="center">
<h1 align="center">
<br />HDC-ROCKETs</h1>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started) 
  - [Installation](#installation)
  - [Download the dataset](#download-the-dataset)
  - [Running](#running)
  - [Results](#results)
- [License](#license)

---

## Overview

Code to the paper [1]. The approach is based on the time series classification algorithm MiniROCKET [2], MultiROCKET [3] and HYDRA [4] and extend it with explicit time encoding by HDC.

[1] K. Schlegel, D. A. Rachkovskij, D. Kleyko, R. W. Gayler, P. Protzel, and P. Neubert, “Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing,” 2025.

[2] A. Dempster, D. F. Schmidt, and G. I. Webb, “MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification,” Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 248–257, 2021.

[3] C. W. Tan, A. Dempster, C. Bergmeir, and G. I. Webb, “MultiRocket: multiple pooling operators and transformations for fast and effective time series classification,” Data Mining and Knowledge Discovery, 2022, doi: 10.1007/s10618-022-00844-1.

[4] A. Dempster, D. F. Schmidt, and G. I. Webb, “Hydra: competing convolutional kernels for fast and accurate time series classification,” Data Mining and Knowledge Discovery, vol. 37, no. 5, pp. 1779–1805, 2023, doi: 10.1007/s10618-023-00939-3.


---

## Repository Structure

```sh
└── /
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   ├── defaults.yaml
    │   ├── HYDRA.yaml
    │   ├── MINIROCKET.yaml
    │   ├── MULTIROCKET.yaml
    │   ├── MULTIROCKET_HYDRA.yaml
    ├── create_figures_tables/
    │   ├── cd_diagram.py
    │   ├── create_eval_data.py
    │   ├── dataset_shapes_UCR_NEW.csv
    │   ├── figure_sec_4_3.py
    │   ├── figures_sec_5_2.py
    │   ├── figures_sec_5_3.py
    │   ├── figures_sec_5_3_example_data.py
    │   ├── figures_sec_5_56.py
    │   ├── plot_config.py
    │   ├── plot_ramifications.py
    │   ├── tables/         
    │   └── images/         
    ├── data/
    │   ├── constants.py
    │   └── dataset_utils.py
    ├── experimental_runs/
    │   ├── runs_hydra.sh
    │   ├── runs_minirocket.sh
    │   ├── runs_multirocket.sh
    │   ├── runs_multirocket_hydra.sh
    ├── models/
    │   ├── HYDRA_utils/
    │   ├── Minirocket_utils/
    │   ├── Multirocket_utils/
    │   ├── Multirocket_HYDRA/
    │   ├── hdc_utils.py
    │   ├── fit_scale_utils.py
    │   ├── Model_Pipeline.py
    │   └── model.pth
    ├── main.py
    ├── net_trail.py
    ├── results/
```

---

## Parameter Description

The table below provides an overview of the main configuration parameters defined in `configs/defaults.yaml`. These parameters control dataset loading, encoding, model settings, and experimental execution.

| **Category**              | **Parameter**                  | **Description**                                           |
|---------------------------|--------------------------------|-----------------------------------------------------------|
| **Dataset and Execution** | `dataset`                      | Dataset name to use (`UCR`, `UCR_NEW`, `synthetic`, etc.) |
|                           | `dataset_idx`                  | Index of dataset to use (for ensemble-style runs)         |
|                           | `complete_UCR` / `complete_UCR_NEW` / `complete_UEA` | Whether to run all datasets in the respective collection  |
|                           | `hard_case`                    | Use challenging version of synthetic dataset              |
|                           | `dataset_path`                 | Path to dataset folder                                    |
|                           | `results_path`                 | Path to output results                                    |
|                           | `run_name`                     | Optional run name (auto-set if null)                      |
|                           | `log_level`                    | Logging level (e.g., `INFO`)                              |
| **Model and Encoding**    | `model`                        | Model type to run (`MINIROCKET`, `MULTIROCKET`, etc.)     |
|                           | `vsa`                          | Vector Symbolic Architecture (`MAP`)                      |
|                           | `fpe_method`                   | Fractional Power Encoding method (`sinusoid`)             |
|                           | `scale`                        | Temporal encoding scale value                             |
|                           | `multi_scale`                  | Run multiple scale values at once                         |
|                           | `best_scale`                   | Automatically select best scale based on grid search      |
|                           | `scales_range`                 | List of scale values to evaluate                          |
|                           | `HDC_dim`                      | Hyperdimensional vector size                              |
|                           | `alpha_ranges`                 | Alpha range for Ridge Classifier                          |
|                           | `classifier`                   | Classifier used (`Ridge`)                                 |
| **Experiment Settings**   | `seed`                         | Random seed for reproducibility                           |
|                           | `seed_for_HDC`                 | Whether to seed HDC vector generation                     |
|                           | `seed_for_splits`              | Whether to seed data splits                               |
|                           | `stat_iterations`              | Number of iterations (e.g., seeds) to evaluate            |
|                           | `n_time_measures`              | How many times to repeat timing for computational runtime |
|                           | `batch_size`                   | Evaluation batch size                                     |
|                           | `max_jobs`                     | Maximum parallel jobs for multiprocessing                 |
| **Normalization**         | `normalize_input`              | Enable input normalization                                |
|                           | `predictors_min_max_norm`      | Min-max normalize predictors                              |
|                           | `predictors_z_score_with_mean` | Apply mean normalization                                  |
|                           | `predictors_z_score_with_std`  | Apply std normalization                                   |
|                           | `predictors_norm`              | Apply vector length normalization                         |
|                           | `norm_test_individually`       | Normalize test samples independently                      |
| **Model-Specific**        | `multi_dim`                    | Output dimension for MultiROCKET                          |
|                           | `HDC_dim_hydra`                | Dimensionality used in HYDRA variants                     |
|                           | `predictors_sparse_scaler`     | Use sparse scaler for predictors (if available)           |

---


---

## Getting Started

***Dependencies***

Dependencies are listed in the [requirements.txt](requirements.txt) file. 

### Installation
We recommend a virtual environment to run the code and using Python 3.10. 
1. Clone the . repository:

```sh
git clone https://github.com/scken/HDC_ROCKETS.git
```

1. Change to the project directory:

```sh
cd HDC_ROCKETS
```

1. Install the dependencies: We recommend using mamba (conda) as virtual environment to run the code and using Python 3.10.

```sh
mamba create -n hdc-rockets python=3.10
mamba activate hdc-rockets
mamba install numpy=1.26 scipy=1.10.0 pandas=2.0.3 matplotlib seaborn=0.13.2 scikit-learn=1.5.2 sktime=0.34.0 aeon=0.11.1 h5py
pip install multi-comp-matrix rocket-fft tsai torch-hd pytorch-lightning hydra-core==1.3.2 tables
```


### Download the dataset:

You can download the datasets and resample indices in two ways:

1. **Official TSML Benchmark Repository**  
   Follow the instructions at https://tsml-eval.readthedocs.io/en/latest/publications/2023/tsc_bakeoff/tsc_bakeoff_2023.html to download the full benchmark datasets (112 + 30) and the corresponding resample indices.

2. **Direct Download from Our Server**  
   Alternatively, you can download a prepackaged version directly from our cloud server:  
   [https://tuc.cloud/index.php/s/z5kZKSxose35sdK](https://tuc.cloud/index.php/s/z5kZKSxose35sdK)

```sh
wget https://tuc.cloud/index.php/s/z5kZKSxose35sdK/download -O dataset.zip
unzip dataset.zip -d data
rm dataset.zip
```

After downloading, extract and store all files in a single folder, e.g., `data/`.


### Running:

- main.py is reading the config files specified in "configs" folder contains more specific parameters for running the experiment (default and model-specific ones)

#### Arguments to run

- all parameters which are not specified in the command line will be taken from the config files
- to overwrite a parameter from the config file, specify it in the command line (e.g. scale=0) or in the files itself

#### UCR Datasets:

- run dataset 0 of UCR with scale=0

```sh
python3 main.py --config-name=MINIROCKET variant=orig complete_UCR=True
```

- Run the complete UCR Benchmark ensemble with different scales:

```sh
python3 main.py --config-name=MINIROCKET variant=hdc_oracle complete_UCR=True
```

- run the complete UCR with automatically selecting the best scale (based on best score)

```sh
python3 main.py --config-name=MINIROCKET variant=hdc_auto complete_UCR=True
```

#### Synthetic Dataset:

- run normal synthetic dataset 1

```sh
python3 main.py --config-name=MINIROCKET variant=orig dataset=synthetic
```

- run hard synthetic dataset 1

```sh
python3 main.py --config-name=MINIROCKET variant=orig dataset=synthetic_hard
```

#### Multiple experimental runs:

To run multiple configurations of the model, use the scripts in the experimental_runs folder.
The scripts are used to run multiple configurations of the model on different datasets.
They include options for synthetic datasets and the UCR dataset, with different configurations for each.

e.g.:

```sh
sh ./experimental_runs/runs_minirocket.sh
```

- this will run all synthetic datasets and UCR in different model configurations of MiniROCKET

### Results:

- the results will be written and saved in /results
- for each run a new folder will be created 
- within this folder the results will be saved in from of Excel spreadsheets, text files for logging and json file for storing the used hyperparameters 
- in addition, the code will copy all Python files to the result folder "Code" for reproducibility (snapshot of the code)


### Creating Figures and Tables

The folder `create_figures_tables/` contains scripts for visualizing results and generating LaTeX tables.
Run the following to generate synthetic result plots:

```sh
python3 ./create_figures_tables/figures_sec_5_2.py
```


---

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for details.

[↑ Return](#Top)