#!/bin/bash
# script for running multiple configurations

cd ..

# define global variable for results
MODEL="MINIROCKET"
HDC_MODEL="HDC"_$MODEL
RESULTS="results/synthetic"

# synthetic dataset 1
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic1
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic1

# synthetic dataset 1 hard
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic1 hard_case=True
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic1 hard_case=True

# synthetic dataset 2
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic2
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic2

# synthetic dataset 2 hard
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS dataset=synthetic2 hard_case=True
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS dataset=synthetic2 hard_case=True

###################
#### UCR
###################
KERNEL="sinc"
KERNEL_LIST='["'"$KERNEL"'"]'

RESULTS="results/UCR/_orig_models"
# orig. Minirocket
python3 main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR=True kernels=$KERNEL_LIST

### HDC-Version
RESULTS="results/UCR/_${KERNEL}"
# all scales and kernels
python3 main.py --config-name=$MODEL variant=hdc_oracle results_path=$RESULTS complete_UCR_NEW=True kernels=$KERNEL_LIST

# data driven selection of scale and kernel
python3 main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True kernels=$KERNEL_LIST
