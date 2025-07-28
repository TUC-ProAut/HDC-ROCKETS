# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz


import pickle
import numpy as np
import logging
from data.constants import *
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from models.Model_Pipeline import Model
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from aeon.datasets._data_loaders import load_from_tsfile
import h5py
import torch
import gc
import os
import random


# config logger
logger = logging.getLogger('log')

####################################
# datasets loading
####################################
def load_dataset(dataset,config):
    """
    load the specific data set (from the data/ folder)
    @param dataset: specifies the data set [string]
    @param config: configure struct with necessary parameters [struct]
    @return: set of training and test data [tuple]
    """
    # load preprocessed data
    if dataset == "UCR":
        X_train, X_test, y_train, y_test = load_UCR_dataset(ucr_index=config['dataset_idx'],
                                                            dataset_path=config['dataset_path'])
    elif dataset == "UCR_NEW":
        X_train, X_test, y_train, y_test = load_new_UCR_dataset(ucr_index=config['dataset_idx'],
                                                            dataset_path=config['dataset_path'],
                                                            config=config)
    elif dataset == "synthetic1":
        X_train, X_test, y_train, y_test = load_synthetic_dataset(hard_case=config['hard_case'], config=config)
    elif dataset == "synthetic2":
        X_train, X_test, y_train, y_test = load_synthetic2_dataset(hard_case=config['hard_case'], config=config)

    # remove nan if in dataset
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0

    if config['normalize_input']:
        logger.debug('     Normalize input data...')
        # Normalizing dimensions independently
        for j in range(X_train.shape[1]):
            # mean of non nan values
            mean = np.nanmean(X_train[:, j])
            std = np.nanstd(X_train[:, j])
            std = np.where(std ==0 , 1, std)
            X_train[:, j] = (X_train[:, j] - mean) / std
            X_test[:, j] = (X_test[:, j] - mean) / std

    # check if length is smaller than 9
    if X_train.shape[2] < 9:
        logger.debug("     Sequence length is smaller than 9, padding with zeros...")
        X_train = np.pad(X_train, ((0, 0), (0, 0), (0, 9 - X_train.shape[2])), 'constant')
        X_test = np.pad(X_test, ((0, 0), (0, 0), (0, 9 - X_test.shape[2])), 'constant')

    return (X_train, X_test, y_train, y_test, config)

def load_resample_indices(dataset_path, dataset_id, seed):
    # load txt file with the indices
    idx_train = np.loadtxt(
        f"{os.path.expanduser(dataset_path)}/_UCR_resample_idx/{UCR_NEW_PREFIX[dataset_id]}/resample{seed}Indices_TRAIN.txt").astype(int)
    idx_test = np.loadtxt(
        f"{os.path.expanduser(dataset_path)}/_UCR_resample_idx/{UCR_NEW_PREFIX[dataset_id]}/resample{seed}Indices_TEST.txt").astype(int)

    return idx_train, idx_test

def load_UCR_dataset(ucr_index, dataset_path):
    """
    load the specific data set (from the data/ folder)
    @param ucr_index: specifies the data set [int]
    @param dataset_path: path to the data set [string]
    @return: set of training and test data [tuple]
    """
    logger.debug(f"     Loading UCR train / test dataset : {str(UCR_SETS[ucr_index])}")
    dataset_path = dataset_path + '/Univariate_ts'

    X_train, y_train = load_UCR_UEA_dataset(name=UCR_PREFIX[ucr_index],
                                            split='train',
                                            return_X_y=True,
                                            extract_path=os.path.expanduser(dataset_path))
    X_test, y_test = load_UCR_UEA_dataset(name=UCR_PREFIX[ucr_index],
                                          split='test',
                                          return_X_y=True,
                                          extract_path=os.path.expanduser(dataset_path))

    X_train, X_test = df2np(X_train, X_test)
    # concat label to create different classes
    labels = np.concatenate((y_train,y_test))
    labels = pd.Categorical(pd.factorize(labels)[0])
    y_train = np.array(labels[0:y_train.shape[0]])
    y_test = np.array(labels[y_train.shape[0]:])

    logger.debug("     Finished processing train dataset..")

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    nb_dims = X_train.shape[1]
    length = X_train.shape[2]

    logger.debug(f"     Number of train samples: {str(train_size)} Number of test samples:  {str(test_size)}")
    logger.debug(f"     Number of variables: {str(nb_dims)}")
    logger.debug(f"     Sequence length: {str(length)}")

    return X_train, X_test, y_train, y_test


def load_new_UCR_dataset(ucr_index, dataset_path, config):
    """
    load the specific data set (from the data/ folder)
    @param ucr_index: specifies the data set [int]
    @param dataset_path: path to the data set [string]
    @return: set of training and test data [tuple]
    """
    logger.debug(f"     Loading NEW UCR train / test dataset : {str(UCR_NEW_PREFIX[ucr_index])}")
    dataset_path = dataset_path + '/Univariate_ts'

    X_train, y_train = load_from_tsfile(
        f"{os.path.expanduser(dataset_path)}/{UCR_NEW_PREFIX[ucr_index]}/{UCR_NEW_PREFIX[ucr_index]}_TRAIN.ts")
    X_test, y_test = load_from_tsfile(
        f"{os.path.expanduser(dataset_path)}/{UCR_NEW_PREFIX[ucr_index]}/{UCR_NEW_PREFIX[ucr_index]}_TEST.ts")

    # concat label to create different classes
    labels = np.concatenate((y_train,y_test))
    labels = pd.Categorical(pd.factorize(labels)[0])
    y_train = np.array(labels[0:y_train.shape[0]])
    y_test = np.array(labels[y_train.shape[0]:])

    logger.debug("     Finished processing train dataset..")

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    nb_dims = X_train.shape[1]
    length = X_train.shape[2]

    logger.debug(f"     Number of train samples: {str(train_size)} Number of test samples:  {str(test_size)}")
    logger.debug(f"     Number of variables: {str(nb_dims)}")
    logger.debug(f"     Sequence length: {str(length)}")

    # use splits created by the bake of redux journal paper
    if config['seed_for_splits']:
        # Merge and convert to contiguous memory layout
        X = np.ascontiguousarray(np.concatenate([X_train, X_test]))
        y = np.ascontiguousarray(np.concatenate([y_train, y_test]))

        # Load split indices
        idx_train, idx_test = load_resample_indices(dataset_path, ucr_index, config['seed'])

        # Efficient indexing using np.take
        X_train = np.take(X, idx_train, axis=0)
        X_test = np.take(X, idx_test, axis=0)
        y_train = np.take(y, idx_train, axis=0)
        y_test = np.take(y, idx_test, axis=0)

        # Optional cleanup
        del X, y

    return X_train, X_test, y_train, y_test


def load_synthetic_dataset(hard_case=False, config=None):
    '''
    create synthetic dataset with impulse as signal and additional noise
    @param hard_case: if set to true, the output is a difficult dataset selected from the original synthetic samples
    @return: tuple of training and test samples and the labels (X_train, X_test, y_train, y_test)
    '''
    config['dataset_idx'] = 0

    # copy the config params
    cfg = config.copy()

    ## create synthetic signal
    length = 500
    n_samples = 500
    a = 0.03

    t = np.linspace(-10, 10, length)
    X_ = np.zeros((n_samples, 1, length), dtype=np.float32)

    for i in range(length):
        X_[i, :, :] = np.roll((t.shape[0] / np.sqrt(np.pi) * a) * np.exp(-(t ** 2 / a ** 2)), i, -1)

    np.random.seed(config['seed'])
    X = X_ + 1 * np.random.randn(X_.shape[0], X_.shape[1], X_.shape[2]).astype(np.float32)

    y = np.zeros((length))
    y[0:int(length / 2)] = 1

    X = np.roll(X, int(length / 2), 0)

    if hard_case:
        # calculate the similarity matrix of original MiniROCKET transform encodings
        cfg['n_steps'] = length
        cfg['n_channels'] = 1
        cfg['kernel'] = 'sinc'
        cfg['model'] = 'MINIROCKET'
        MINIROCKET = Model(cfg)
        MINIROCKET.encoder.fit(X)
        X_MR = MINIROCKET.transform_data(X, train=True)
        sim_mat_mr = cosine_similarity(X_MR, X_MR)

        # chose those indices with a high similarity between class one and two (empirically chosen)
        q = 0.96  # 96th quantile
        special_mat = sim_mat_mr[:250, 250:]
        thresh = np.quantile(special_mat, q)
        rows, cols = np.where((special_mat > thresh) & (special_mat < 1))
        cols = cols + 249
        idx = np.concatenate((np.unique(rows), np.unique(cols)))
        X_train, X_test, y_train, y_test = train_test_split(X[idx], y[idx], test_size=0.2, random_state=42)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def load_synthetic2_dataset(hard_case=False, config=None):
    '''
    create synthetic dataset with impulse as signal and additional noise
    @param hard_case: if set to true, the output is a difficult dataset selected from the original synthetic samples
    @return: tuple of training and test samples and the labels (X_train, X_test, y_train, y_test)
    '''
    config['dataset_idx'] = 0

    # copy the config params
    cfg = config.copy()

    ## create synthetic signal
    length = 500

    # Parameters
    amplitude = 4  # Amplitude of the peaks
    noise_level = 1 # Add some noise

    # Generate symmetric time series data
    t = np.arange(length)
    n_samples = 200
    y = []
    X = np.zeros((n_samples, 1, length))
    np.random.seed(config['seed'])

    for i in range(n_samples):
        x = np.random.normal(0, noise_level, length)
        if i % 2 == 0:
            s = np.random.randint(length//8, 2*length//8)
            x[s] = amplitude
            x[-s] = amplitude
            # Peaks near the beginning and end
            X[i, :] = x
            y += [0]
        else:
            s = np.random.randint(2.5*length//8, 3*length//8)
            x[s] = amplitude
            x[-s] = amplitude
            # Peaks near the beginning and end
            X[i, :] = x
            y += [1]

    y = np.array(y)
    # sort the samples by class
    idx = np.argsort(y)
    X = X[idx]
    y = y[idx]

    if hard_case:
        # calculate the similarity matrix of original MiniROCKET transform encodings
        cfg['n_steps'] = length
        cfg['n_channels'] = 1
        cfg['kernel'] = 'sinc'
        cfg['scale'] = 0
        cfg['model'] = 'MULTIROCKET'
        MULTIROCKET = Model(cfg)
        MULTIROCKET.encoder.fit(X,y)
        X_MR = MULTIROCKET.transform_data(X, train=True)
        sim_mat_mr = cosine_similarity(X_MR, X_MR)

        # chose those indices with a high similarity between class one and two
        q = 0.96  # 96th quantile
        special_mat = sim_mat_mr[:n_samples//2, n_samples//2:]
        thresh = np.quantile(special_mat, q)
        rows, cols = np.where((special_mat > thresh) & (special_mat < 1))
        cols = cols + n_samples//2
        idx = np.concatenate((np.unique(rows), np.unique(cols)))
        X_train, X_test, y_train, y_test = train_test_split(X[idx], y[idx], test_size=0.2, random_state=42)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test


def df2np(X_train, X_test):
    '''
    convert pandas dataframe to numpy array
    @param X_train:
    @param X_test:
    @return:
    '''

    # train samples
    X = X_train
    num_samples = X.shape[0]
    num_dim = X.shape[1]
    s_length = X['dim_0'][0].size * 10 # set it bigger than necessary --> cut the tensor afterwards (to do so, we can handly different timeseries lengths)

    data_train = np.ones([num_samples, num_dim, s_length]) * np.nan
    longest_series = 0

    for s in range(num_samples):
        idx = 0
        for c in X.columns:
            series = X[c][s].values
            if len(series) > longest_series:
                longest_series = len(series)
            data_train[s, idx, :len(series)] = series
            idx += 1

    # test samples
    X = X_test
    num_samples = X.shape[0]
    num_dim = X.shape[1]

    data_test = np.ones([num_samples, num_dim, s_length]) * np.nan

    for s in range(num_samples):
        idx = 0
        for c in X.columns:
            series = X[c][s].values
            if len(series) > longest_series:
                longest_series = len(series)
            data_test[s, idx, :len(series)] = series
            idx += 1

    # calculate the max length of all series
    data = np.concatenate([data_train, data_test],axis=0)

    X_train = data_train[:,:,:longest_series]
    X_test = data_test[:, :, :longest_series]

    return X_train, X_test


class CustomDatasetPaths(Dataset):
    def __init__(self, data, config):
        self.config = config
        torch.manual_seed(config['seed'])
        if self.config['gaussian_ids']:
            self.channel_ids = torch.randn((config['n_channels'], config['HDC_dim'])).to(torch.float32)
        else:
            # rand values from {-1,1}
            self.channel_ids = torch.randint(0, 2, (config['n_channels'], config['HDC_dim'])).to(torch.float32)
            self.channel_ids[self.channel_ids == 0] = -1

        self.mask = torch.ones((config['n_channels']), dtype=torch.bool)

        if isinstance(data, tuple):
            self.data = data[0]
            self.labels = data[1]
            self.read_from_file = False
            self.used_idx = np.arange(self.data.shape[0])
        else:
            self.data = []
            self.idx_file_map = {}
            self.idx_per_file = {}
            self.paths = data
            self.use_cache = False
            self.read_from_file = True
            counter = 0

            for path in self.paths:
                sample = path
                self.data.append(sample)
                example = h5py.File(sample, 'r')
                num_samples = example['X'].shape[0]
                for i in range(num_samples):
                    self.idx_file_map[counter] = sample
                    self.idx_per_file[counter] = i
                    counter += 1

            self.used_idx = np.arange(counter)

        self.num_classes = self.config['n_classes']
        self.max_cache_size = self.config['max_cache_size']
        self.cached_data = {}
        self.use_cache = False
        self.for_encoding = False

    def __len__(self):
        return len(self.used_idx)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, idx):
        true_idx = self.used_idx[idx]

        if self.use_cache:
            # If caching is enabled, load the data from cache if available
            if true_idx in self.cached_data:
                sample, label = self.cached_data[true_idx]
            else:
                # If not in cache, load the data from file
                sample, label = self._load_data(true_idx)
                # Check if cache size exceeds the maximum
                if len(self.cached_data) < self.max_cache_size:
                    # Cache the loaded data
                    self.cached_data[true_idx] = (sample, label)
        else:
            # If caching is not enabled, load the data from file
            sample, label = self._load_data(true_idx)

        one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot_label[label] = 1
        return sample, one_hot_label

    def _load_data(self, true_idx):
        file_path = self.idx_file_map[true_idx] if self.read_from_file else None

        if self.read_from_file:
            with h5py.File(file_path, 'r') as file:
                sample = torch.tensor(file['X'][self.idx_per_file[true_idx]]).to(torch.float32)
                label = file['y'][self.idx_per_file[true_idx]]
        else:
            sample = torch.tensor(self.data[true_idx]).to(torch.float32)
            label = self.labels[true_idx]

        if not self.for_encoding:
            if self.config['channel_ids']:
                sample = sample * self.channel_ids

            if self.config['use_wsp']:
                sample = sample[self.mask]

            if self.config['bundle_channels'] and not self.config['use_wsp']:
                sample = sample.sum(-2)

            if self.config['concatenate_channels']:
                sample = sample.flatten()

        return sample, label

    def clear_cache(self):
        self.cached_data = {}
        gc.collect()

    def get_labels(self):
        if self.read_from_file:
            labels = []
            for i in self.used_idx:
                file_path = self.idx_file_map[i] if self.read_from_file else None
                with h5py.File(file_path, 'r') as file:
                    label = file['y'][self.idx_per_file[i]]
                labels.append(label)
            labels = np.array(labels)
        else:
            labels = [self.labels[i] for i in self.used_idx]
            labels = np.array(labels)
        return labels

    def get_data(self):
        data = [self.__getitem__(i)[0] for i in range(len(self.used_idx))]
        data = np.stack(data)
        return data

    def preload_cache(self):
        for i in range(len(self.used_idx)):
            sample, label = self._load_data(i)
            # Check if cache size exceeds the maximum
            if len(self.cached_data) < self.max_cache_size:
                # Cache the loaded data
                self.cached_data[i] = (sample, label)
