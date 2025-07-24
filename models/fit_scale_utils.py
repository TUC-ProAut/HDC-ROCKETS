# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from copy import deepcopy
import numba
import time
import multiprocessing


from models.hdc_utils import *

import logging
logger = logging.getLogger('log')

def find_best_s_LOONMSE(scales, kernels, seed, alpha_range, fpe_method, dim, encoder, normalizer, input, label, max_jobs=64):
    """
    Finds the optimal encoding scale and kernel for High-Dimensional Computing (HDC) embeddings using Leave-One-Out Mean Squared Error (LOO-MSE) method.

    This method assesses and selects the best encoding strategy by iterating over available encoders, scales, and kernels. It uses RidgeClassifierCV for evaluating the scores and determining the optimal parameters.

    Args:
        scales: List of scale values to evaluate during grid search.
        kernels: List of kernel functions to evaluate during grid search.
        seed: Random seed for reproducibility.
        alpha_range: Range of alpha for Ridge regression values to evaluate.
        fpe_method: Method for computing FPE.
        dim: Dimensionality of the input data.
        encoder: Encoder to preprocess the input data.
        normalizer: Normalization method applied to the data.
        input: Input data to be used in the evaluation.
        label: Label data corresponding to the input.
        max_jobs: Maximum number of parallel jobs to run (default is 64).

    Returns:
        A tuple containing:
            - The best scale value that minimizes the error.
            - The best kernel function found through grid search.
            - The total time taken for the grid search process.
    """

    t = time.time()
    np.random.seed(seed)

    # get number of cpus and set up parallel jobs
    total_jobs = len(scales) * len(kernels)
    total_cpus = numba.get_num_threads()
    max_jobs = min(total_jobs, total_cpus, max_jobs)
    numba_threads = max(1, total_cpus - max_jobs)
    numba.set_num_threads(numba_threads)
    logger.debug(f"CPU Count: {total_cpus}, Grid Size: {total_jobs}, Using {max_jobs} parallel jobs.")

    results = Parallel(n_jobs=max_jobs)(
        delayed(evaluate_scale_kernel_external)(
            s, k, seed, alpha_range, dim,
            input,
            label,
            normalizer.steps,
            encoder,
            fpe_method
        ) for s in scales for k in kernels
    )
    best_sco = np.array(results).reshape(len(scales), len(kernels))

    # find row and column of best score
    best_sc, best_k = np.unravel_index(np.argmax(best_sco), best_sco.shape)
    best_s = scales[best_sc]
    best_k = kernels[best_k]

    # set number of threads for numba
    numba.set_num_threads(total_cpus)
    grid_search_time = time.time() - t

    return best_s, best_k, grid_search_time


def evaluate_scale_kernel_external(s, k, seed, alpha_range, dim, input_data, label_data, predictor_scaler, encoder, fpe_method):
    """
    Evaluates the performance of a RidgeClassifierCV model using a specific scaling factor and kernel.
    Args:
        s: Scaling factor used in the computation of poses during encoding.
        k: Kernel parameter for the encoding process.
        seed: Random seed for generating reproducible results.
        alpha_range: Range of alpha values used for RidgeClassifierCV.
        dim: Dimensionality of the high-dimensional space used by the encoder.
        input_data: Input dataset to be encoded and transformed.
        label_data: Labels corresponding to the input data for supervised learning.
        predictor_scaler: Configuration for the data normalization pipeline.
        encoder: Encoder object responsible for transforming the input data.
        fpe_method: FPE method used in the encoding process.

    Returns:
        The best score achieved by the RidgeClassifierCV model evaluated on the transformed data.
    """
    output = []

    encoder.compute_poses(n_steps=input_data.shape[-1], scale=s, seed=seed,
                          HDC_dim=dim,
                          fpe_method=fpe_method, kernel=k)
    output.append(encoder.transform(input_data))
    data_tf = np.concatenate(output, axis=1)

    # Normalization logic
    pipeline = Pipeline(predictor_scaler)
    data_tf = pipeline.fit_transform(data_tf)

    classifier = RidgeClassifierCV(alphas=alpha_range)
    classifier.fit(data_tf, label_data)
    return classifier.best_score_

