# ========================================================================================
# HDC-MiniROCKET: Extension of MiniROCKET with HDC for temporal Encoding
#
# Original MiniROCKET Authors:
# Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
# Source: MiniROCKET - A Very Fast (Almost) Deterministic Transform for Time Series Classification
# Paper: https://arxiv.org/abs/2012.08791
# Code: https://github.com/angus924/minirocket
#
# Original implementation adapted from the official MiniROCKET codebase (GPL-3.0 License).
#
# Extensions and modifications (2025) by:
# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
# Paper: Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Purpose: To integrate Hyperdimensional Computing (HDC) operations into the MiniROCKET pipeline.
# Notable additions:
#   - HDC temporal encoding using Fractional Power Encoding (FPE)
#   - Binding operations for MAP VSA model
#   - FPE bandwidth (scale) parameter grid search
#
# License: GNU General Public License v3.0 (GPL-3.0)
# This file and all modifications remain under the terms of the GPL-3.0 license.
# See https://www.gnu.org/licenses/gpl-3.0.en.html for details.
# ========================================================================================

import numpy as np
import pandas as pd

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation.panel import check_X
from numba import njit
from numba import prange
from numba import vectorize, int32
import scipy as sp
from numpy.fft import fft, ifft

from models.hdc_utils import *
from models.fit_scale_utils import *

class HDCMiniRocketMultivariate(_PanelToTabularTransformer):

    def __init__(
        self, num_features=10_000, max_dilations_per_kernel=32, random_state=None, vsa='MAP',fit_scale=True, normalizer=None,
            scales=None, kernels=None, scale=0, kernel='sinc', fpe_method='orig', max_jobs=64
    ):
        self.name = "HDCMiniRocket"
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = (
            np.int32(random_state) if isinstance(random_state, int) else None
        )
        self.poses = np.zeros((10,10),dtype=np.float32) # initial definition as placeholder for njit function
        self.vsa = vsa
        self.fit_scale = fit_scale
        self.normalizer = normalizer
        self.alpha_range = np.logspace(-3,3,10)

        self.best_scale = None
        self.best_kernel = None
        if scales is None:
            scales = np.logspace(0, 3, 7, base=2) - 1
        self.scales = scales
        if kernels is None:
            kernels = ['sinc']
        self.kernels = kernels
        self.fpe_method = fpe_method

        # single scale and kernel
        self.scale = scale
        self.kernel = kernel
        self.max_jobs = max_jobs


        super(HDCMiniRocketMultivariate, self).__init__()

    def fit(self, X, y=None):
        """Fits dilations and biases to input time series.
        And fit the best FPE scale (bandwidth) parameter for given data based on grid search
        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)

        Returns
        -------
        self
        """
        X = check_X(X, coerce_to_numpy=True).astype(np.float32)
        *_, n_timepoints = X.shape
        if n_timepoints < 9:
            # raise ValueError(
            #     (
            #         f"n_timepoints must be >= 9, but found {n_timepoints};"
            #         " zero pad shorter series so that n_timepoints == 9"
            #     )
            # )
            X = np.concatenate((np.zeros((X.shape[0], X.shape[1], 1),dtype=np.float32), X),-1)

        self.parameters = _fit_multi(
            X, self.num_features, self.max_dilations_per_kernel, self.random_state
        )
        self._is_fitted = True

        # fit the scale if set best_scale
        dim = self.num_features # HDC dimension
        if self.fit_scale:
            self.scale, self.kernel, self.grid_search_time = find_best_s_LOONMSE(self.scales, self.kernels, self.random_state,
                                                                    self.alpha_range, self.fpe_method, dim,
                                                                    self, self.normalizer, X, y, max_jobs=self.max_jobs)
        else:
            self.grid_search_time = 0

        # compute poses for time encoding
        self.compute_poses(n_timepoints, self.scale,dim, self.random_state,
                           fpe_method= self.fpe_method, kernel=self.kernel)

        return self

    def transform(self, X, y=None):
        """Transforms input time series.

        Parameters
        ----------
        X : pandas DataFrame, input time series (sktime format)
        y : array_like, target values (optional, ignored as irrelevant)

        Returns
        -------
        pandas DataFrame, transformed features
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True).astype(np.float32)
        if self.vsa == 'MAP':
            results = _transform_multi_map(X, self.parameters, self.poses)
        elif self.vsa == 'HRR':
            results = _transform_multi_hrr(X, self.parameters, self.poses)
        elif self.vsa == 'FHRR':
            results = _transform_multi_fhrr(X, self.parameters, self.poses)
        elif self.vsa == 'BSC':
            results = _transform_multi_bsc(X, self.parameters, self.poses)

        return results

    def compute_poses(self, n_steps, scale, HDC_dim, seed, fpe_method='orig', kernel='sinc'):
        # create pose matrix for time encoding
        scalars = np.linspace(0,1, n_steps, dtype=np.float32)
        poses = create_FPE(scalars, scale, HDC_dim, seed=seed, fpe_method=fpe_method, kernel=kernel)

        # load pose matrix to rocket transformer
        self.poses = poses


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],optional(int32))",  # noqa
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_multi(
    X,
    num_channels_per_combination,
    channel_indices,
    dilations,
    num_features_per_dilation,
    quantiles,
    seed,
):

    if seed is not None:
        np.random.seed(seed)

    n_instances, n_columns, n_timepoints = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[
                num_channels_start:num_channels_end
            ]

            _X = X[np.random.randint(n_instances)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(
                (num_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, num_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(n_timepoints, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _fit_multi(X, num_features=10_000, max_dilations_per_kernel=32, seed=None):
    if seed is not None:
        np.random.seed(seed)

    _, n_columns, n_timepoints = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(
        n_timepoints, num_features, max_dilations_per_kernel
    )

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(n_columns, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, num_combinations)
    ).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            n_columns, num_channels_this_combination, replace=False
        )

        num_channels_start = num_channels_end

    biases = _fit_biases_multi(
        X,
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        quantiles,
        seed,
    )

    return (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    )


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@vectorize("float32(float32,float32,float32)", nopython=True, cache=True)
def _PPV_HDC(v, b, t):
    ''' the adapted PPV calculation function to incorporate HDC operations (binding with timestamps) '''
    if v > b:
        return t
    else:
        return 0

@vectorize("float32(float32,float32,int32)", nopython=True, cache=True)
def _PPV_HDC_BSC(v, b, t):
    ''' The adapted PPV calculation function to incorporate HDC operations (binding with timestamps) '''
    return t ^ int32(v > b)

@vectorize("float32(float32,float32,float32,float64)", nopython=True, cache=True)
def _PPV_HDC_FHHR(v, b, t, p):
    ''' the adapted PPV calculation function to incorporate HDC operations (binding with timestamps) '''
    if v > b:
        return t
    else:
        return t+p

@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),float32[:,:])",  # noqa
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi(X, parameters, poses):
    n_instances, n_columns, n_timepoints = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.float32)

    for example_index in prange(n_instances):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C_complete = (
                    C_alpha
                    + C_gamma[index_0]
                    + C_gamma[index_1]
                    + C_gamma[index_2]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[
                            example_index, feature_index_start + feature_count
                        ] = _PPV(C, biases[feature_index_start + feature_count]).mean()

                else:
                    for feature_count in range(num_features_this_dilation):
                        features[
                            example_index, feature_index_start + feature_count
                        ] = _PPV(
                            C[padding:-padding],
                            biases[feature_index_start + feature_count],
                        ).mean()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features



@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),float32[:,:])",
    # noqa
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi_hrr(X, parameters, poses):
    # poses_array = poses
    # poses_array = fft(poses, axis=0)
    # transpose the poses matrix
    poses = poses.T
    poses_array = fft(poses, axis=0)

    n_instances, n_columns, n_timepoints = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.float32)

    for example_index in prange(n_instances):

        # initiate feature array without binding
        features_nobinding = np.zeros((num_features, n_timepoints), dtype=np.float64)

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                                            num_channels_start:num_channels_end
                                            ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                        C_alpha[channels_this_combination]
                        + C_gamma[index_0][channels_this_combination]
                        + C_gamma[index_1][channels_this_combination]
                        + C_gamma[index_2][channels_this_combination]
                )
                C_complete = (
                        C_alpha
                        + C_gamma[index_0]
                        + C_gamma[index_1]
                        + C_gamma[index_2]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features_nobinding[feature_index_start + feature_count] = _PPV(C, biases[
                            feature_index_start + feature_count]) * 2 - 1

                else:
                    for feature_count in range(num_features_this_dilation):
                        features_nobinding[feature_index_start + feature_count, padding:-padding] = _PPV(
                            C[padding:-padding], biases[feature_index_start + feature_count]) * 2 - 1

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end


        # NUMPY IMPLEMENTATION
        features_bundling = np.real(ifft(np.sum(fft(features_nobinding, axis=0) * poses_array, axis=1), axis=0))
        # features_bundling /= np.linalg.norm(features_bundling)

        features[example_index, :] = features_bundling.T

    return features

@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),float32[:,:])",  # noqa
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi_fhrr(X, parameters, poses):
    #poses = poses.ravel() # turns 9996x176 to 1759296
    n_instances, n_columns, n_timepoints = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.complex64)

    for example_index in prange(n_instances):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C_complete = (
                    C_alpha
                    + C_gamma[index_0]
                    + C_gamma[index_1]
                    + C_gamma[index_2]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        #pose = poses[
                        #    (feature_index_start + feature_count) * n_timepoints:
                        #    (feature_index_start + feature_count + 1) * n_timepoints]
                        pose = poses[:,feature_index_start + feature_count]

                        features[example_index, feature_index_start + feature_count] = np.exp(1j*_PPV_HDC_FHHR(C,biases[feature_index_start + feature_count],pose,np.pi).astype(np.float64)).sum()

                else:
                    for feature_count in range(num_features_this_dilation):
                        #pose = poses[
                        #    (feature_index_start + feature_count) * n_timepoints:
                        #    (feature_index_start + feature_count + 1) * n_timepoints]
                        pose = poses[:,feature_index_start + feature_count]

                        features[example_index, feature_index_start + feature_count] = np.exp(1j*_PPV_HDC_FHHR(C[padding:-padding],biases[feature_index_start + feature_count],pose[padding:-padding],np.pi).astype(np.float64)).sum()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end
    # concatenate cosine and sine
    features = np.concatenate((np.real(features), np.imag(features)), axis=1)

    #features = np.cos(features) + np.sin(features)

    return features

@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),float32[:,:])",  # noqa
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi_bsc(X, parameters, poses):
    n_instances, n_columns, n_timepoints = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    # cast pose to int32
    poses = poses.astype(np.int32)

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.float32)

    for example_index in prange(n_instances):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C_complete = (
                    C_alpha
                    + C_gamma[index_0]
                    + C_gamma[index_1]
                    + C_gamma[index_2]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        pose = poses[:, feature_index_start + feature_count].ravel()
                        ppv_hdc =  _PPV_HDC_BSC(C,
                                     biases[feature_index_start + feature_count],
                                     pose).sum()
                        features[
                            example_index, feature_index_start + feature_count
                        ] = ppv_hdc

                else:
                    for feature_count in range(num_features_this_dilation):
                        pose = poses[:, feature_index_start + feature_count].ravel()
                        ppv_hdc = _PPV_HDC_BSC(C[padding:-padding],
                                     biases[feature_index_start + feature_count],
                                     pose[padding:-padding]).sum()
                        features[
                            example_index, feature_index_start + feature_count
                        ] = ppv_hdc

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features

@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),float32[:,:])",  # noqa
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi_map(X, parameters, poses):
    n_instances, n_columns, n_timepoints = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)])
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    # flatten poses for more efficient access
    poses_shape = poses.shape
    poses = poses.T.ravel()

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.float32)

    for example_index in prange(n_instances):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_columns, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_columns, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C_complete = (
                    C_alpha
                    + C_gamma[index_0]
                    + C_gamma[index_1]
                    + C_gamma[index_2]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        pose = poses[
                            (feature_index_start + feature_count) * n_timepoints:
                            (feature_index_start + feature_count + 1) * n_timepoints]
                        ppv_hdc =  _PPV_HDC(C,
                                     biases[feature_index_start + feature_count],
                                     pose)
                        ppv_hdc = ppv_hdc.sum()
                        features[
                            example_index, feature_index_start + feature_count
                        ] = ppv_hdc


                else:
                    for feature_count in range(num_features_this_dilation):
                        pose = poses[
                               (feature_index_start + feature_count) * n_timepoints:
                               (feature_index_start + feature_count + 1) * n_timepoints]
                        ppv_hdc = _PPV_HDC(C[padding:-padding],
                                     biases[feature_index_start + feature_count],
                                     pose[padding:-padding])
                        ppv_hdc = ppv_hdc.sum()
                        features[
                            example_index, feature_index_start + feature_count
                        ] = ppv_hdc


                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features

