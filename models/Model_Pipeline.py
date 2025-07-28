# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

from os import cpu_count

import scipy
import logging

from sklearn.linear_model import RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import torch
from time import time
from joblib import Parallel, delayed
from copy import deepcopy
import os
import numba

from models.Minirocket_utils.minirocket_multivariate import MiniRocketMultivariate
from models.Minirocket_utils.hdc_minirocket_multivariate import HDCMiniRocketMultivariate
from models.Multirocket_utils.multirocket_multivariate import MultiRocket
from models.Multirocket_utils.hdc_multirocket_multivariate import HDCMultiRocket
# from models.HYDRA_utils.hydra_multivariate import HYDRA
# from models.HYDRA_utils.hdc_hydra_multivariate import HDC_HYDRA
# from models.Multirocket_HYDRA.hdc_multirocket_HYDRA import HDCMultirocketHYDRA
# from models.Multirocket_HYDRA.multirocket_hydra import MultirocketHYDRA
from models.hdc_utils import *

# config logger
logger = logging.getLogger('log')



class Model:
    """
        Model class defines a machine learning pipeline based on predefined configurations including data preprocessing,
        feature extraction, model encoding, and classifier training.

        Attributes:
            config: A dictionary containing the configuration parameters for the model such as model type, device type,
                    normalizer settings, training parameters, and others.
            predictor_scaler: A Scikit-learn pipeline that handles preprocessing of predictors
                              (e.g., min-max scaling, z-score normalization).
            encoder: The feature extraction model initialized based on the chosen configuration.
            deep_model: A boolean that is True if the model is a deep learning-based model, else False.
            classifier: The classifier used after encoding for predictions, such as Ridge regression or custom classifier.
            train_preproc: The time taken for preprocessing and encoding the training data.
            grid_search_time: The time spent on hyperparameter grid search for feature extraction encoders.
            training_time: The processing time taken for training either the encoder or classifier.
            X_train_tf: The preprocessed training dataset transformed by the encoder pipeline.
    """
    def __init__(self,config):
        # params
        self.config = config

        # output the number of threads for numby
        logger.debug(f"Numba threads: {numba.get_num_threads()}")

        # set processing pipeline for predictors
        pipe = []
        if self.config['predictors_min_max_norm']:
            pipe.append(('min_max', MinMaxScaler(feature_range=(-1, 1), clip=True)))
        pipe.append(('z_score', StandardScaler(with_mean=self.config['predictors_z_score_with_mean'],
                                               with_std=self.config['predictors_z_score_with_std'])))

        if self.config['predictors_norm']:
            pipe.append(('norm', Normalizer()))
        self.predictor_scaler = Pipeline(pipe)

        # get device
        if torch.cuda.is_available():
            self.config['device'] = 'cuda'
        else:
            self.config['device'] = 'cpu'


        self.deep_model = False

        # init the used model
        if self.config['model'] == 'HDC_MINIROCKET':
            self.encoder = HDCMiniRocketMultivariate(random_state=self.config['seed'],
                                                                        num_features=self.config['HDC_dim'], vsa=self.config['vsa'],
                                                                        fpe_method=self.config['fpe_method'],
                                                                        scales=self.config['scales'],
                                                                        kernels=self.config['kernels'],
                                                                        scale=self.config['scale'],
                                                                        kernel='sinc',
                                                                        max_jobs=self.config['max_jobs'],
                                                                        fit_scale=self.config['best_scale'], normalizer=self.predictor_scaler)
        elif self.config['model'] == 'MINIROCKET':
            self.encoder = MiniRocketMultivariate(random_state=self.config['seed'],
                                                             num_features=self.config['HDC_dim'])

        elif self.config['model'] == 'MULTIROCKET':
            self.encoder = MultiRocket(random_state=self.config['seed'], num_features=self.config['multi_dim'])
        elif self.config['model'] == 'HDC_MULTIROCKET':
            self.encoder = HDCMultiRocket(random_state=self.config['seed'],
                                                              num_features=self.config['multi_dim'],
                                                              fpe_method=self.config['fpe_method'],
                                                              scales=self.config['scales'],
                                                              kernels=self.config['kernels'],
                                                              scale=self.config['scale'],
                                                              kernel='sinc',
                                                              fit_scale=self.config['best_scale'],
                                                              max_jobs= self.config['max_jobs'],
                                                              normalizer=self.predictor_scaler)

        # elif self.config['model'] == 'HYDRA':
        #     self.encoder = HYDRA(self.config['n_steps'], self.config['n_channels'], random_state= self.config['seed'])
        # elif self.config['model'] == 'HDC_HYDRA':
        #     self.encoder = HDC_HYDRA(self.config['n_steps'], self.config['n_channels'], dim=self.config['HDC_dim_hydra'],
        #                             fpe_method=self.config['fpe_method'],
        #                             scales=self.config['scales'],
        #                             kernels=self.config['kernels'],
        #                             scale=self.config['scale'],
        #                             kernel='sinc',
        #                             fit_scale=self.config['best_scale'],
        #                             normalizer=self.predictor_scaler,
        #                             use_sparse_scaler=self.config['predictors_sparse_scaler'],
        #                             batch_size=self.config['batch_size'],
        #                             max_jobs=self.config['max_jobs'],
        #                                            )
        #
        # elif self.config['model'] == 'MULTIROCKET_HYDRA':
        #     self.encoder = MultirocketHYDRA(self.config['n_steps'], self.config['n_channels'],
        #                                     num_features=self.config['multi_dim'], random_state= self.config['seed'],
        #                                     batch_size= self.config['batch_size'], use_sparse_scaler= self.config['predictors_sparse_scaler'])
        # elif self.config['model'] == 'HDC_MULTIROCKET_HYDRA':
        #     self.encoder = HDCMultirocketHYDRA(self.config['n_steps'], self.config['n_channels'],
        #                                        num_features=self.config['multi_dim'],
        #                                        HDC_HYDRA_dim=self.config['HDC_dim_hydra'],
        #                                        random_state=self.config['seed'],
        #                                        use_sparse_scaler=self.config['predictors_sparse_scaler'],
        #                                        fit_scale=self.config['best_scale'], normalizer=self.predictor_scaler,
        #                                        scales=self.config['scales'], kernels=self.config['kernels'],
        #                                        scale=self.config['scale'], kernel='sinc',
        #                                        fpe_method=self.config['fpe_method'],
        #                                        max_jobs=self.config['max_jobs'],
        #                                        batch_size= self.config['batch_size'])

    def transform_data(self, input, train=False):
        """
        Transform the input data using the encoder defined in the model.
        Args:
            input: Input data to be processed by the ensemble of encoders.
            train: A boolean flag indicating whether the model is in training mode. Defaults to False.

        Returns:
            A numpy array representing the processed embeddings, normalized if required.
        """
        # encode the input
        output = self.encoder.transform(input)

        if train:
            # normalize the HDC embeddings
            output = self.predictor_scaler.fit_transform(output)
        if not train and not self.config['norm_test_individually']:
            # use params from training to normalize test set
            if self.predictor_scaler is not None:
                output = self.predictor_scaler.transform(output)
        if not train and self.config['norm_test_individually']:
            # fit test set individually
            if self.predictor_scaler is not None:
                output = self.predictor_scaler.fit_transform(output)

        return output


    def train_model(self,X_train,y_train):
        """
        Train the model using the provided training data and labels.
        Args:
            X_train: The training data containing the input features.
            y_train: The labels corresponding to the X_train dataset.
        """
        ####### Rocket-based encoders
        t_proc = []

        # time measurement for training
        for i in range(self.config['n_time_measures']):
            t = time()
            # fit encoder
            self.encoder.fit(X_train, y_train)
            if hasattr(self.encoder, 'kernel') and hasattr(self.encoder, 'scale'):
                self.config['kernel'] = self.encoder.kernel
                self.config['scale'] = self.encoder.scale
            else:
                self.config['kernel'] = None
                self.config['scale'] = None

            if hasattr(self.encoder, 'grid_search_time'):
                # if the encoder has a grid search time attribute, use it
                self.grid_search_time = self.encoder.grid_search_time
            else:
                self.grid_search_time = 0

            logger.debug(f"Encoder {self.encoder.name} fitted with scale: {self.config['scale']} and kernel: {self.config['kernel']}")
            # transform the training data
            self.X_train_tf = self.transform_data(X_train, train=True)
            t_proc.append((time() - t))
        processing_time = np.median(t_proc)
        self.train_preproc = processing_time

        # classifier
        self.classifier = RidgeClassifierCV(alphas=np.logspace(self.config['alpha_ranges'][0],
                                                                self.config['alpha_ranges'][1],
                                                                self.config['alpha_ranges'][2]))

        t_train = []
        for i in range(self.config['n_time_measures']):
            t = time()
            self.classifier.fit(self.X_train_tf, y_train)
            t_train.append(time() - t)

        self.training_time = np.median(t_train)

        return

    def eval_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test data and return predictions and scores.
        Args:
            X_test: Test data features.
            y_test: Test data labels.

        Returns:
            pred: Predicted labels for the test data.
            scores: Decision function scores for the test data.
        """
        ####### Rocket-based encoders
        t_proc = []
        # time measurement
        for i in range(self.config['n_time_measures']):
            t = time()
            # encode without recomputing the pose matrix for timestamps
            self.X_test_tf = self.transform_data(X_test, train=False)
            t_proc.append((time() - t))
        processing_time = np.median(t_proc)
        self.test_preproc = processing_time

        t_test = []
        for i in range(self.config['n_time_measures']):
            t = time()
            pred = self.classifier.predict(self.X_test_tf)
            t_test.append(time() - t)
        self.testing_time = np.median(t_test)

        # evaluate the results
        logger.debug('Results on test data: ')
        report = classification_report(y_test.astype(int), pred, output_dict=True)
        logger.debug(classification_report(y_test.astype(int), pred))

        scores = self.classifier.decision_function(self.X_test_tf)

        # print time results
        logger.debug("Train preproc time: " + str(self.train_preproc))
        logger.debug("Test preproc time: " + str(self.test_preproc))
        logger.debug("Training time: " + str(self.training_time))
        logger.debug("Evaluation: " + str(self.testing_time))

        return pred, scores
