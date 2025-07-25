# ========================================================================================
# HDC-MultiROCKET: Extension of MultiROCKET with HDC for temporal Encoding
#
# Original MultiROCKET Authors:
# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
# Source: MultiROCKET - Multiple pooling operators and transformations for fast and effective time series classification
# Paper: https://arxiv.org/abs/2102.00457
# Code: https://github.com/angus924/minirocket
#
# Original implementation adapted from the official codebase (GPL-3.0 License).
#
# Extensions and modifications (2025) by:
# Kenny Schlegel (TU Chemnitz)
# Purpose: To integrate Hyperdimensional Computing (HDC) operations into the MultiROCKET pipeline.
# Notable additions:
#   - HDC temporal encoding using Fractional Power Encoding (FPE)
#   - Binding operations for MAP VSA model
#   - FPE bandwidth (scale) parameter grid search
#
# License: GNU General Public License v3.0 (GPL-3.0)
# This file and all modifications remain under the terms of the GPL-3.0 license.
# See https://www.gnu.org/licenses/gpl-3.0.en.html for details.
# ========================================================================================

import time
import numpy as np
from numba import njit, prange, vectorize
from sklearn.linear_model import RidgeClassifierCV


from models.fit_scale_utils import *

class HDCMultiRocket:
    def __init__(
            self,
            num_features=50000,
            verbose=0,
            random_state=None,
            num_features_per_kernel = 4,
            fit_scale=True, normalizer=None,
            scales=None, kernels=None, scale=0, kernel='sinc', fpe_method='orig',
            max_jobs=64
    ):
        self.name = "HDCMultiRocket"
        self.dim = num_features

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = num_features_per_kernel
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)
        self.random_state = (
            np.int32(random_state) if isinstance(random_state, int) else None
        )
        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        self.poses = np.zeros((10,10),dtype=np.float32) # initial definition as placeholder for njit function
        self.train_duration = 0
        self.test_duration = 0
        self.generate_kernel_duration = 0
        self.train_transforms_duration = 0
        self.test_transforms_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

        self.verbose = verbose

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

    def fit(self, x_train, y_train=None):
        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, x_train.shape))
        if x_train.shape[2] < 10:
            # handling very short series (like PensDigit from the MTSC archive)
            # series have to be at least a length of 10 (including differencing)
            _x_train = np.zeros((x_train.shape[0], x_train.shape[1], 10), dtype=x_train.dtype)
            _x_train[:, :, :x_train.shape[2]] = x_train
            x_train = _x_train
            del _x_train

        self.generate_kernel_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.train_transforms_duration = 0

        _start_time = time.perf_counter()
        xx = np.diff(x_train, 1)
        self.train_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        self.base_parameters = fit(
            x_train.astype(np.float32),
            num_features=self.num_kernels,
            seed=self.random_state
        )
        self.diff1_parameters = fit(
            xx.astype(np.float32),
            num_features=self.num_kernels,
            seed=self.random_state
        )
        self.generate_kernel_duration += time.perf_counter() - _start_time


        # fit the scale if set best_scale
        dim = self.dim # HDC dimension
        *_, n_timepoints = x_train.shape
        if self.fit_scale:
            self.scale, self.kernel, self.grid_search_time = find_best_s_LOONMSE(self.scales, self.kernels, self.random_state,
                                                                    self.alpha_range, self.fpe_method, dim,
                                                                    self, self.normalizer, x_train, y_train, max_jobs=self.max_jobs)
        else:
            self.grid_search_time = 0

        # compute poses for time encoding
        self.compute_poses(n_timepoints, self.scale,dim, self.random_state,
                           fpe_method= self.fpe_method, kernel=self.kernel)

    def transform(self, X, y=None, mask=None):
        start_time = time.perf_counter()
        # create first order derivation
        xx = np.diff(X, 1)
        x_train_transform = transform_hdc(
            X.astype(np.float32), xx.astype(np.float32),
            self.base_parameters, self.diff1_parameters,
            self.poses,
            self.n_features_per_kernel
        )


        x_train_transform = np.nan_to_num(x_train_transform)

        elapsed_time = time.perf_counter() - start_time
        if self.verbose > 1:
            print('[{}] Kernels applied!, took {}s'.format(self.name, elapsed_time))
            print('[{}] Transformed Shape {}'.format(self.name, x_train_transform.shape))

        if self.verbose > 1:
            print('[{}] Training'.format(self.name))

        return x_train_transform

    def compute_poses(self, n_steps, scale, HDC_dim, seed=None, fpe_method='orig', kernel='sinc'):
        # create pose matrix for time encoding
        scalars = np.linspace(0, 1, n_steps, dtype=np.float32)
        poses = create_FPE(scalars, scale, HDC_dim, seed=seed, fpe_method=fpe_method, kernel=kernel)

        # load pose matrix to rocket transformer
        self.poses = poses


@njit("float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],optional(int32))",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X,
                num_channels_per_combination,
                channel_indices,
                dilations,
                num_features_per_dilation,
                quantiles,
                seed):
    num_examples, num_channels, input_length = X.shape

    if seed is not None:
        np.random.seed(seed)

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

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

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros((num_channels_this_combination, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels_this_combination, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

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

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[
                                                                           feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)

@vectorize("float32(float32,float32,float32)", nopython=True, cache=True)
def _PPV_HDC(v, b, t):
    ''' the adapted PPV calculation function to incorporate HDC operations (binding with timestamps) '''
    # return ppv_hdc, mpv, lspv (mipv not necessary)
    if v > b:
        return t  # temporal encoding and mean positiv values
    else:
        return 0
@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _MEAN_HDC(v, b):
    ''' the adapted PPV calculation function to incorporate HDC operations (binding with timestamps) '''
    # return ppv_hdc, mpv, lspv (mipv not necessary)
    if v > b:
        return v+b  # temporal encoding and mean positiv values
    else:
        return 0.

def fit(X, num_features=10_000, max_dilations_per_kernel=32, seed=None):
    _, num_channels, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels,
                                                                                num_channels_this_combination,
                                                                                replace=False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices,
                         dilations, num_features_per_dilation, quantiles, seed)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases


@njit(
    "float32[:,:](float32[:,:,:],float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),"
    "float32[:,:],int32)",
    fastmath=True, parallel=True, cache=True)
def transform_hdc(X, X1, parameters, parameters1, poses, n_features_per_kernel=4):
    num_examples, num_channels, input_length = X.shape
    poses = np.ascontiguousarray(poses)

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

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

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        pose = poses[:, feature_index_start + feature_count]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        ppv_hdc = _PPV_HDC(C, _bias, pose)
                        ppv_hdc = ppv_hdc.sum() / C.shape[0]
                        features[example_index, end] = ppv_hdc
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0

                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        pose = poses[:, feature_index_start + feature_count]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += (_c[j] + _bias)
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        ppv_hdc = _PPV_HDC(_c, _bias, pose[padding:-padding])
                        ppv_hdc = ppv_hdc.sum() / _c.shape[0]
                        features[example_index, end] = ppv_hdc
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0


                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros((num_channels, input_length - 1), dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, num_channels, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]
                        pose = poses[:C.shape[-1], feature_index_start + feature_count]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += (C[j] + _bias)
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch= stretch

                        end = feature_index + n_features_per_transform
                        ppv_hd = _PPV_HDC(C, _bias, pose)
                        ppv_hd = ppv_hd.sum() / C.shape[0]
                        features[example_index, end] = ppv_hd
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]
                        pose = poses[:C.shape[-1], feature_index_start + feature_count]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += (_c[j] + _bias)
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        ppv_hdc = _PPV_HDC(_c, _bias, pose[padding:-padding])
                        ppv_hdc = ppv_hdc.sum() / _c.shape[0]
                        features[example_index, end] = ppv_hdc
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0

                feature_index_start = feature_index_end

    return features

