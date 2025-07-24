# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

import os
import time
import shutil
import logging
from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import hashlib
import json
import sys
import inspect
from filelock import FileLock, Timeout

from net_trail import NetTrial

def get_run_name(config):
    # Base name using model and dataset
    setting = "{}_{}_{}_s{}".format(
        config.get('model', 'MODEL'),
        config.get('dataset', 'DATA'),
        config.get('variant', 'VARIANT'),
        config.get('scale', 'SCALE')
    )

    param_str = json.dumps(config, sort_keys=True)
    hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:4]

    # Optional: Append date or random word
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"{setting}_{timestamp}_{hash_suffix}"

@hydra.main(config_path="configs", config_name="MINIROCKET")
def main_app(cfg: DictConfig) -> None:
    # Setup config
    variant = cfg['variant']
    cfg_variant = cfg[variant]
    cfg.update(cfg_variant)
    config = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    log_level_str = config.get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logging.basicConfig(level=log_level, format='%(message)s')
    logger = logging.getLogger('log')
    logger.setLevel(log_level)
    logger.debug("Current directory: " + os.getcwd())

    # Generate run folder and log file
    if config.get('run_name') is None:
        name_of_run = get_run_name(config)
    else:
        name_of_run = config['run_name']
    config['results_folder'] = os.path.join(config['results_path'], name_of_run)
    os.makedirs(config['results_folder'], exist_ok=True)

    json_path = os.path.join(config['results_folder'], 'config_log.json')
    config['json_path'] = json_path  # Store the path for use in submodules

    # add config to json file
    config['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['run_name'] = name_of_run
    update_json_file(json_path, config)

    # Create logs folder in results folder
    logs_folder = os.path.join(config['results_folder'], 'logs')
    os.makedirs(logs_folder, exist_ok=True)

    # Use dataset index and seed to generate unique log name
    dataset_idx = config.get('dataset_idx', 'NA')
    seed = config.get('seed', 'NA')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name_of_run}_ds{dataset_idx}_se{seed}_{timestamp}.log"
    log_path = os.path.join(logs_folder, log_filename)

    # Add file handler for this job
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # Setup scales
    config['scales'] = np.logspace(*config['scales_range'], base=2) - 1

    # Backup code files
    backup_code(config['results_folder'])

    # Save the exact command used to run the script
    # Path to the original script
    script_path = os.path.basename(inspect.getfile(sys.modules['__main__']))

    # Full command including all CLI overrides
    original_command = f"python {script_path} " + " ".join(sys.argv[1:])

    # Also write a bash script to rerun the experiment
    bash_script_path = os.path.join(config['results_folder'], 'code/reproduce.sh')
    with open(bash_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(original_command + "\n")
    os.chmod(bash_script_path, 0o755)

    # Log run information
    logger.debug('_________________________')
    logger.debug('Time: ' + config['start_time'])
    logger.info('##### Name of run: ' + name_of_run + ' #####')
    logger.info('_________________________')
    log_config(logger, config)

    # Determine scales for the experiment
    scales = config['scales'] if config['multi_scale'] else [config['scale']]

    start_time = time.time()
    success = run_experiments(scales, config, logger)
    elapsed_time = time.time() - start_time
    update_json_file(json_path, {'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    update_json_file(json_path, {f'total_time_seed{seed}': elapsed_time})

    logger.debug("Success: " + str(success))
    logger.debug("_________________________")
    logger.debug("Time: " + str(datetime.now()))
    logger.debug(f"Total time: {elapsed_time:.2f} seconds")
    logger.debug("_________________________")

    # Merge new log lines into main_log.log, avoiding duplicates
    main_log_path = os.path.join(config['results_folder'], 'main_log.log')
    existing_lines = set()
    if os.path.exists(main_log_path):
        with open(main_log_path, "r") as f:
            existing_lines = set(line.strip() for line in f)

    with open(main_log_path, "a") as main_log:
        for fname in sorted(os.listdir(logs_folder)):
            if fname.endswith(".log") and fname != "main_log.log":
                with open(os.path.join(logs_folder, fname), "r") as f:
                    for line in f:
                        if line.strip() not in existing_lines:
                            main_log.write(line)
                            existing_lines.add(line.strip())

def backup_code(target_folder: str) -> None:
    """Backup code files to the results folder."""
    exclude_paths = ['logs', 'results', 'venv']
    for d, _, files in os.walk('.'):
        if any(os.path.normpath(d).startswith(os.path.normpath(ep)) for ep in exclude_paths):
            continue
        for f in files:
            if f.endswith(('.py', '.sh', '.m', '.yaml')):
                dest_folder = os.path.join(target_folder, 'code', d)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy(os.path.join(d, f), os.path.join(dest_folder, f))


def log_config(logger: logging.Logger, config: dict) -> None:
    """Log the configuration."""
    logger.debug("--- Config ---")
    for key, value in config.items():
        logger.debug(f"   - {key}: {value}")
    logger.debug('_________________________')


def run_experiments(scales: list, config: dict, logger: logging.Logger) -> int:
    """Run the experiments at different scales."""
    success = 1

    for s_idx, scale in enumerate(scales):
        config['scale'] = scale
        logger.debug(f"Scale = {scale}")

        if config.get('complete_UCR'):
            success &= run_dataset_experiment("UCR", 128, config, logger, s_idx)
        elif config.get('complete_UCR_NEW'):
            success &= run_dataset_experiment("UCR_NEW", 142, config, logger, s_idx)
        elif config.get('complete_UEA'):
            success &= run_dataset_experiment("UEA", 30, config, logger, s_idx)
        else:
            logger.debug(f"#### Normal Training on {config['dataset']} ####")
            trainer = NetTrial(config)
            trainer.config.update({'dataset_idx': config['dataset_idx'], 'scale_idx': s_idx})
            trainer.train()
            # add acc to the config json file
            seed = config.get('seed', 'NA')
            idx = config['dataset_idx']
            update_json_file(config['json_path'], {f'acc_idx{idx}_seed{seed}': trainer.acc})
            logger.debug(f"Mean acc: {trainer.acc}")

        # add success to the config json file
        seed = config.get('seed', 'NA')
        update_json_file(config['json_path'], {f'success_seed{seed}': success})

    return success


def run_dataset_experiment(dataset: str, iterations: int, config: dict, logger: logging.Logger, scale_idx: int) -> int:
    """Run experiments for specific datasets."""
    logger.debug(f"##### Full experiment on all {dataset} time series #####")
    mean_acc = 0
    success = 1

    for i in range(iterations):
        try:
            trainer = NetTrial(config)
            trainer.config.update({'dataset_idx': i, 'scale_idx': scale_idx, 'dataset': dataset})
            logger.info(f"Index: {i}")
            trainer.train()
            mean_acc += trainer.acc
            # add acc to the config json file
            seed = config.get('seed', 'NA')
            update_json_file(config['json_path'], {f'acc_idx{i}_seed{seed}': trainer.acc})
        except Exception as e:
            logger.error(f"Error: {e}")
            success = 0
        finally:
            del trainer

    logger.debug(f"-------------------------")
    logger.debug(f"Mean acc for {dataset}: {mean_acc / iterations}")
    return success


def update_json_file(json_path: str, updates: dict) -> None:
    """Safely update a JSON file with new key-value pairs only if keys do not already exist."""
    lock_path = json_path + '.lock'
    lock = FileLock(lock_path, timeout=60)
    try:
        with lock:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}

            for key, value in updates.items():
                data[key] = value

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
    except Timeout:
        print(f"Timeout occurred while trying to acquire lock for {json_path}")
    except Exception as e:
        print(f"Failed to update JSON file {json_path}: {e}")

if __name__ == '__main__':
    main_app()