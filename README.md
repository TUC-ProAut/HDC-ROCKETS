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
    │   ├── tables/         # new: stores LaTeX output
    │   └── images/         # new: stores plots (e.g., bar plots)
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

1. Install the dependencies: We recommend using a virtual environment to run the code and using Python 3.8.

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


### Download the dataset:

Follow the instructions given in https://tsml-eval.readthedocs.io/en/latest/publications/2023/tsc_bakeoff/tsc_bakeoff_2023.html to download the datasets (112 + 30) and the resample indices. 
Store all in one folder, e.g. `data/Univariate_ts/`.


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