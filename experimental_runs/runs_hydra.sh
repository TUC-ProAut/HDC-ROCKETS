#!/bin/bash
# script for running multiple configurations

cd ..

# define global variable for results
MODEL="HYDRA"
HDC_MODEL="HDC"_$MODEL
RESULTS="results_comp"

# synthetic dataset 1
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic

# synthetic dataset 1 hard
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic_hard
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic_hard

# synthetic dataset 2
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic2
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic2

# synthetic dataset 2 hard
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic2_hard
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic2_hard

###################
#### UCR
###################
# orig.
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR_NEW=True

### HDC-Version
# all scales and kernels
python3 main.py --config-name=$MODEL variant=hdc_oracle results_path=$RESULTS complete_UCR_NEW=True

# data driven selection of scale and kernel
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True


