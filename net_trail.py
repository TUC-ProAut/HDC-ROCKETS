# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

from data.dataset_utils import *
import logging
from models.Model_Pipeline import Model
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix
import pandas as pd
import os
from filelock import FileLock, Timeout
import json

# self.config logger
logger = logging.getLogger('log')

class NetTrial():
    """
    Represents a machine learning trial class designed for data handling, model initialization, training, and evaluation.

    Attributes:
      config (dict): A dictionary containing configurations for the dataset, model, and training process.
      data (tuple): A tuple containing training and testing datasets.
      X_train (array-like): Training input data.
      X_test (array-like): Testing input data.
      y_train (array-like): Training labels.
      y_test (array-like): Testing labels.
      model (object): This represents the neural network model instance used for training and evaluation.
      acc (float): Accuracy of the model on the test dataset.
    """

    def __init__(self,config):
        # set self.config parameter
        self.config = config

    def load_data(self):
        """
        Loads the dataset and prepares the training and testing data.

        For synthetic datasets (e.g., 'synthetic1', 'synthetic_hard', 'synthetic2', 'synthetic2_hard'),
        sets the dataset index to 0. Loads the dataset based on the configuration and extracts training
        and testing features (X) and labels (y). Updates the configuration with the number of classes,
        number of channels, and the number of time steps based on the training data. Initializes the
        model with the updated configuration.

        Raises:
            ValueError: If the dataset specified in the configuration is not recognized.
        """
        # load dataset initially
        if (self.config['dataset'] == 'synthetic1' or self.config['dataset'] == 'synthetic_hard' or
            self.config['dataset'] == 'synthetic2' or self.config['dataset'] == 'synthetic2_hard'):
            self.config['dataset_idx'] = 0
        self.data = load_dataset(self.config['dataset'], self.config)
        self.X_train = self.data[0]
        self.X_test = self.data[1]
        self.y_train = self.data[2]
        self.y_test = self.data[3]

        self.config['n_classes'] = len(np.unique(self.y_train))
        self.config['n_channels'] = self.X_train.shape[1]
        self.config['n_steps'] = self.X_train.shape[2]

        self.model = Model(self.config)

    def train(self):
        """
        Executes training, evaluation, and statistical resampling for a machine learning model,
        handles multiple kernel configurations, and persistently stores results, scores, and runtime statistics.

        The method's workflow includes:

        1. Iterates over different kernel configurations if the `best_scale` flag is disabled.
        2. For each kernel, reloads data, trains the model, computes metrics (accuracy, F1-score),
           and logs these for each statistical iteration determined by `stat_iterations`.
        3. Persistently saves model evaluation scores and results in HDF5 format.
        4. Writes accuracy metrics and related configurations to Excel files.
        5. Stores runtime statistics, including preprocessing, training, inference,
           and grid search times to separate files.

        Logs relevant progress and error information for debugging purposes.
        """

        # iterate over kernels only if best_scale option is false
        if not self.config['best_scale']:
            kernels = self.config['kernels']
        else:
            kernels = ['auto']

        for k_idx, kernel in enumerate(kernels):
            if not self.config['best_scale']:
                self.config['kernel'] = kernel
            logger.debug('Kernel: ' + str(kernel))

            acc_stat = []
            f1_stat = []
            # statistical resampling (repeat training and evaluation with different seeds)
            for stat_it in range(self.config['stat_iterations']):
                logger.debug('Statistial iteration: ' + str(stat_it))
                if self.config['stat_iterations']>1:
                    self.config['seed'] = stat_it

                # reload data with new seed (for random split)
                self.load_data()

                col_idx = k_idx + 1 + self.config['scale_idx'] * len(self.config['kernels'])

                # self.config['HDC_dim = X_train.shape[1]
                logger.debug('Training dataset shape: ' + str(self.X_train.shape) + str(self.y_train.shape))
                logger.debug('Test dataset shape: ' + str(self.X_test.shape) + str(self.y_test.shape))

                self.config['train_count'] = len(self.X_train)
                self.config['test_data_count'] = len(self.X_test)
                self.model.config = self.config

                # train the model
                self.model.train_model(self.X_train,self.y_train)

                # evaluate the model
                y_pred, y_scores = self.model.eval_model(self.X_test, self.y_test)
                # evaluate the results
                logger.debug('Results on test data: ')
                report = classification_report(self.y_test.astype(int), y_pred, output_dict=True)

                # accuracy and f1 score
                acc = report['accuracy']
                self.acc = acc
                f1 = f1_score(self.y_test.astype(int), y_pred, average='weighted')

                logger.info('Accuracy: ' + str(acc))
                logger.info('F1 score: ' + str(f1))

                acc_stat.append(np.mean(acc))
                f1_stat.append((np.mean(f1)))

                idx = self.config['dataset_idx']

                try:
                    file_path = (f"{self.config['results_folder']}/"
                                 f"results"
                                 f"_{self.config['dataset']}"
                                 f"_seed_{self.config['seed']}")
                    # create directory if not exists
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))

                    ################### save scores ######################
                    if np.unique(self.y_test).shape[0] == 2 and y_scores.ndim == 1:
                        y_scores = y_scores[:,None]
                    scores = pd.DataFrame(np.concatenate((self.y_test[:,None].astype(np.int32),y_scores.astype(np.float32)),axis=1))
                    # convert to 16 bit to save space
                    scores = scores.astype(np.float16)

                    h5_path = f"{self.config['results_folder']}/scores_{self.config['seed']}.h5"
                    lock = FileLock(h5_path + ".lock", timeout=120)

                    try:
                        with lock:
                            with pd.HDFStore(h5_path, complevel=9, complib='zlib') as store:
                                store.append(
                                    f"scores{self.config['seed']}_{self.config['dataset']}_idx_{self.config['dataset_idx']}_scale_{self.config['scale_idx']}_kernel_{kernel}",
                                    scores
                                )
                    except Timeout:
                        logger.error(f"[HDFStore] Timeout: Could not acquire lock for '{h5_path}'")


                    #################### save to excel ######################
                    file_acc = file_path + '_acc.xlsx'
                    file_time = file_path + '_time.xlsx'
                    acc_df = pd.DataFrame({'data': acc}, index=[0])
                    idx_df = pd.DataFrame({'data': idx}, index=[0])
                    # write index
                    append_df_to_excel(file_acc, pd.DataFrame({self.config['dataset'] + '_idx': []}),
                                       startcol=0, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({self.config['dataset'] + '_idx': []}),
                                       startcol=0, index=False, header=True,
                                       startrow=0)

                    if self.config['best_scale']:
                        header_name = pd.DataFrame({'acc_at_best_scale' :[],
                                                    'best_scale':[],
                                                    'best_kernel':[]})
                    else:
                        header_name = pd.DataFrame({'scale_idx' + str(self.config['scale_idx']) +
                                                    '_' + kernel: []})
                    # files for the normal results
                    append_df_to_excel(file_acc, header_name,
                                       startcol=col_idx, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_acc, acc_df,
                                       index=False, header=False, startrow=idx + 1,
                                       startcol=col_idx)

                    if self.config['best_scale']:
                        append_df_to_excel(file_acc, pd.DataFrame({'best_scale': str(self.config['scale'])}, index=[0]),
                                           index=False, header=False, startrow=idx + 1,
                                           startcol=2)
                        append_df_to_excel(file_acc, pd.DataFrame({'best_kernel': str(self.config['kernel'])}, index=[0]),
                                              index=False, header=False, startrow=idx + 1,
                                                startcol=3)


                    # files for the acc results
                    append_df_to_excel(file_acc, idx_df,
                                       startcol=0, index=False, header=False,
                                       startrow=idx + 1)

                    # write run-time results to file
                    append_df_to_excel(file_time, idx_df,
                                       startcol=0, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'prep_time_train': []}),
                                       startcol=1, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.train_preproc}, index=[0]),
                                       startcol=1, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'prep_time_test': []}),
                                       startcol=2, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.test_preproc}, index=[0]),
                                       startcol=2, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'train_time': []}),
                                       startcol=3, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.training_time}, index=[0]),
                                       startcol=3, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'inf_time': []}),
                                       startcol=4, index=False, header=True,
                                       startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.testing_time}, index=[0]),
                                       startcol=4, index=False, header=False,
                                       startrow=idx + 1)
                    append_df_to_excel(file_time, pd.DataFrame({'grid_search_time': []}),
                                        startcol=5, index=False, header=True,
                                        startrow=0)
                    append_df_to_excel(file_time, pd.DataFrame({'data': self.model.grid_search_time}, index=[0]),
                                        startcol=5, index=False, header=False,
                                        startrow=idx + 1)

                except Exception as e:
                    logger.error(f"Error: {e}")


    def eval(self):
        """
        Evaluates the model and logs evaluation results.

        This method performs the following tasks:
        1. Logs the shape of the test dataset for debugging purposes.
        2. Updates the total count of test data samples in the configuration.
        3. Sets the model's configuration to match the current context.
        4. Evaluates the model using the provided training and testing data and computes metrics such as accuracy, F1 score, and the confusion matrix.
        5. Logs computing time data related to model inference into a text file with fields like dataset index, preprocessing method, and testing time.
        6. Logs the averaged accuracy results for debugging purposes.

        Args:
            self: The instance of the class containing this method.
        """


        logger.debug('Test dataset shape: ' + str(self.X_test.shape) + str(self.y_test.shape))

        self.config['test_data_count'] = len(self.X_test)
        self.model.config = self.config

        # evaluate the model
        acc, f1, confusion_matrix = self.model.eval_model(self.X_train,self.y_train,self.X_test,self.y_test,fold=it)


        with open('results/computing_time_inference_' + self.config['dataset'] + '_' + self.config['model'] + '.txt',
                  'a') as file:
            file.write(str(self.config['dataset_idx']) + '\t' +
                       str(self.model.test_preproc) + '\t'
                       + str(self.model.testing_time) + '\n'
                       )

        logger.debug('Accuracy results: ' + str(np.mean(acc)))



def append_df_to_excel(filename, df, sheet_name='Sheet1', startrow=None,
                       truncate_sheet=False, lock_timeout=120,
                       **to_excel_kwargs):
    """
    Safely append a DataFrame [df] to an existing Excel file [filename] using a file lock.
    If the file does not exist, it will be created.

    Parameters:
      filename : File path or existing ExcelWriter (e.g., 'results.xlsx')
      df : DataFrame to append
      sheet_name : Target sheet in the Excel file (default: 'Sheet1')
      startrow : Starting row for writing (default: append at bottom)
      truncate_sheet : Remove and recreate the sheet if True
      lock_timeout : Max time to wait for the file lock (in seconds)
      to_excel_kwargs : Additional kwargs passed to DataFrame.to_excel()
    """
    # Strip engine if provided
    to_excel_kwargs.pop('engine', None)

    lock_path = filename + ".lock"
    lock = FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock:
            if os.path.isfile(filename):
                writer = pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists="overlay")
            else:
                # Make parent dir if needed
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                writer = pd.ExcelWriter(filename, mode='w', engine='openpyxl')

            # Write data
            df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, **to_excel_kwargs)
            writer.close()

    except Timeout:
        logger.error(f"[ExcelWriter] Timeout: Could not acquire lock for '{filename}'")
