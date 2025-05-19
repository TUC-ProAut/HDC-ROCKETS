<div align="center">
<h1 align="center">
<br />HDC-ROCKETs</h1>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started) 
  - [Installation](#installation)
  - [Download the dataset](#download-the-dataset)
  - [Running](#running)
  - [Results](#results)
- [License](#license)

---

## Overview

Code to the paper [1]. The approach is based on the time series classification algorithm MiniROCKET [2] and extend it with explicit time encoding by HDC.

[1] ...

[2] A. Dempster, D. F. Schmidt, and G. I. Webb, “MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification,” Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 248–257, 2021.


---

## Repository Structure

```sh
└── /
    ├── LICENSE
    ├── README.md
    ├── configs
    │   ├── HYDRA.yaml
    │   ├── MINIROCKET.yaml
    │   ├── MULTIROCKET.yaml
    │   ├── MULTIROCKET_HYDRA.yaml
    │   └── defaults.yaml
    ├── create_figures_tables
    │   ├── cd_diagram.py
    │   ├── create_eval_data.py
    │   ├── dataset_shapes_UCR_NEW.csv
    │   ├── figure_sec_4_3.py
    │   ├── figures_sec_5_2.py
    │   ├── figures_sec_5_3.py
    │   ├── figures_sec_5_3_example_data.py
    │   ├── figures_sec_5_56.py
    │   ├── plot_config.py
    │   └── plot_ramifications.py
    ├── data
    │   ├── constants.py
    │   └── dataset_utils.py
    ├── experimental_runs
    │   ├── runs_hydra.sh
    │   ├── runs_minirocket.sh
    │   ├── runs_multirocket.sh
    │   ├── runs_multirocket_hydra.sh
    ├── main.py
    ├── models
    │   ├── HYDRA_utils
    │   ├── Minirocket_utils
    │   ├── Model_Pipeline.py
    │   ├── Multirocket_utils
    │   ├── hdc_utils.py
    │   ├── model.pth
    │   └── tabular.py
    ├── net_trail.py
    ├── requirements.txt
    ├── results
```

---

## Modules

<details closed><summary>Root</summary>

| File                  | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [plot_timestamp_sim.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/plot_timestamp_sim.py) | This code generates a plot of graded similarity in timesteps using the HDC-MiniROCKET model. It calculates cosine similarity between vector encodings at different timestamps and plots the results. The plot shows how similarity changes with timestamp difference as a percentage of the total series length.                                                                                                                                                                                                                            |
| [ts_viewer.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/ts_viewer.py)          | The code is a time series viewer application implemented using Tkinter and Matplotlib libraries in Python. It allows users to select different datasets, visualize time series data, and interactively explore different variables and class labels. The application provides options to visualize the mean of samples and supports scrolling to view large datasets.                                                                                                                                                                       |
| [requirements.txt](https://github.com/scken/HDC_MiniRocket_private/blob/master/requirements.txt)      | The code requires specific versions of the following libraries: scipy, matplotlib, numpy, openpyxl, sktime, pandas, numba, and scikit_learn. These libraries provide functionality for scientific computing, data visualization, machine learning, and data manipulation.                                                                                                                                                                                                                                                                   |
| [main.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/main.py)               | The code initializes and runs a time series classification experiment using the HDC-MiniROCKET model. It accepts command line arguments for dataset selection, path settings, normalization, scaling parameters, and model type. The code saves logs and results to specified folders, generates a random name for the run, and copies all Python files to the result folder. It then trains and evaluates the model on either a single dataset or all datasets in the UCR and UEA repositories, with the option to run on multiple scales. |
| [main_run.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/main_run.py)           | The code is a high-level class for training and evaluating time series classification models. It loads the dataset, trains the model, and evaluates its performance using metrics like accuracy, F1 score, and confusion matrix. The results are saved in Excel files for analysis. The code also includes a function to append a DataFrame to an existing Excel file.                                                                                                                                                                      |
| [plot_figures.m](https://github.com/scken/HDC_MiniRocket_private/blob/master/plot_figures.m)        | The code implements several functionalities. It reads data from an Excel file, performs calculations and generates plots for pairwise accuracy comparison, relative performance change, accuracy over different parameters, and evaluation of time effort for different algorithms.                                                                                                                                                                                                                                                         |

</details>

<details closed><summary>Experimental_runs</summary>

| File                      | Summary                                                                                                                                                                                                                                                                                              |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [runs_multirocket.sh](https://github.com/scken/HDC_MiniRocket_private/blob/master/experimental_runs/runs_multirocket.sh)       | This script is used to run multiple configurations for different datasets using the "main.py" file. It sets the global variables for the model and the results folder, and then runs the main script for Minirocket encoding with different arguments for each dataset and configuration.            |
| [runs_minirocket.sh](https://github.com/scken/HDC_MiniRocket_private/blob/master/experimental_runs/runs_minirocket.sh)        | This script is used to run multiple configurations for different datasets using the "main.py" file. It sets the global variables for the model and the results folder, and then runs the main script for Multirocket encoding with different arguments for each dataset and configuration.           |
| [runs_hydra.sh](https://github.com/scken/HDC_MiniRocket_private/blob/master/experimental_runs/runs_hydra.sh)             | This script is used to run multiple configurations for different datasets using the "main.py" file. It sets the global variables for the model and the results folder, and then runs the main script for Hydra encoding with different arguments for each dataset and configuration.                 |
| [runs_multirocket_hydra.sh](https://github.com/scken/HDC_MiniRocket_private/blob/master/experimental_runs/runs_multirocket_hydra.sh) | This script is used to run multiple configurations for different datasets using the "main.py" file. It sets the global variables for the model and the results folder, and then runs the main script for Multirocket and Hydra encoding with different arguments for each dataset and configuration. |

</details>

<details closed><summary>Models</summary>

| File              | Summary                                                                                                                                                                                                                                                                                                                              |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model_Pipeline.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/models/Model_Pipeline.py) | The code in models/Model_Pipeline.py defines a class called Model_Pipeline that is used for training and evaluating a defined model. It can be initialized with a specific encoder type like Minirocket, HDC-Minirocker, ... . It also contains methods for fitting the best scale parameter, creating the pose matrices, and so on. |


</details>

<details closed><summary>Multirocket_utils</summary>

| File                        | Summary                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [multirocket_multivariate.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/models/Multirocket_utils/multirocket_multivariate.py) | The script implements a Multirocket_Encoder class that is used for time series encoding. It utilizes the MultiRocketFeatures module, an implementation of the MultiRocket algorithm for time series classification. The Multirocket_Encoder module provides methods for forward pass, transformation, and fitting, to generate encoded features from input time series data. |

</details>

<details closed><summary>Minirocket_utils</summary>

| File                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [minirocket_multivariate.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/models/Minirocket_utils/minirocket_multivariate.py) | The code defines a Minirocket_Encoder class that is used for time series encoding. It utilizes the MiniRocketFeatures module, an implementation of the MiniRocket algorithm for time series classification. The Minirocket_Encoder module provides methods for forward pass, transformation, and fitting, to generate encoded features from input time series data.                                                                                                                                                                                                     |

</details>

<details closed><summary>Hydra_utils</summary>

| File                  | Summary                                                                                                                                                                                                                                                                                                                                         |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [hydra_multivariate.py](https://github.com/scken/HDC_MiniRocket_private/blob/master/models/HYDRA_utils/hydra_multivariate.py) | The code defines a Hydra_Encoder class that is used for time series encoding. It utilizes the HydraFeatures module, an implementation of the Hydra algorithm for time series classification. The Hydra_Encoder module provides methods for forward pass, transformation, and fitting, to generate encoded features from input time series data. |

</details>

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

- main.py contains some arguments (dataset, UCR index, scale, HDC dim, etc.)
- config.py contains more specific parameters for running the experiment

#### Arguments to run

- all parameters which are not specified in the command line will be taken from the config file
- to overwrite a parameter from the config file, specify it in the command line (e.g. scale=0)

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
./experimental_runs/runs_minirocket.sh
```

### Results:

- the results will be written and saved in /results
- for each run a new folder will be created 
- within this folder the results will be saved in from of Excel spreadsheets, text files for logging and json file for storing the used hyperparameters 
- in addition, the code will copy all Python files to the result folder "Code" for reproducibility (snapshot of the code)

---

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for details.

[↑ Return](#Top)