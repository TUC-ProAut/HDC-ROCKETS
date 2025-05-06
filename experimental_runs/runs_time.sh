#!/bin/bash
# script for running multiple configurations

cd ..

# define global variable for results

RESULTS="results_time_measurements/"

## MINIROCKET
MODEL="MINIROCKET"
HDC_MODEL="HDC"_$MODEL

########################
# Time measurements
########################
TIME_RUNS=5
SCALES="[0,3,7]"

# UCR NEW
python main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW
python main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW scales_range=$SCALES

# MultiROCKET
MODEL="MULTIROCKET"
HDC_MODEL="HDC"_$MODEL


# UCR NEW
python main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW
python main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW scales_range=$SCALES

# Hydra
MODEL="HYDRA"
HDC_MODEL="HDC"_$MODEL


# UCR NEW
python main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW
python main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW scales_range=$SCALES

# MultiROCKET-HYDRA
MODEL="MULTIROCKET_HYDRA"
HDC_MODEL="HDC"_$MODEL


# UCR NEW
python main.py --config-name=$MODEL variant=orig results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW
python main.py --config-name=$MODEL variant=hdc_auto results_path=$RESULTS complete_UCR_NEW=True n_time_measures=$TIME_RUNS dataset=UCR_NEW scales_range=$SCALES



