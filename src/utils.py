# -*- coding: utf-8 -*-
import os
import yaml
import pickle
import argparse

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from usad.spot import SPOT

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


class ConfigHandler:

    def __init__(self, makedirs=True):
        # load default config
        dir_ = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dir_, 'config.yml')
        with open(config_path, 'r') as f:
            self._config_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

        # update config according to executing parameters
        parser = argparse.ArgumentParser()
        for field, value in self._config_dict.items():
            parser.add_argument(f'--{field}', default=value)
        self._config = parser.parse_args()

        # complete config
        self._trans_format()
        if self._config.x_dims is None:
            self._config.x_dims = get_data_dim(self._config.dataset)
        if makedirs:
            self._complete_dirs()

    def _trans_format(self):
        """
        convert invalid formats of config to valid ones
        """
        config_dict = vars(self._config)
        for item, value in config_dict.items():
            if value == 'None':
                config_dict[item] = None
            elif isinstance(value, str) and is_number(value):
                if value.isdigit():
                    value = int(value)
                else:
                    value = float(value)
                config_dict[item] = value

    def _complete_dirs(self):
        """
        complete dirs in config
        """
        if self._config.save_dir:
            self._config.save_dir = self._make_dir(self._config.save_dir, 'save')

        if self._config.result_dir:
            self._config.result_dir = self._make_dir(self._config.result_dir, 'result')
        
        if self._config.cluster_save_dir:
            self._config.cluster_save_dir = self._make_dir(self._config.cluster_save_dir, 'cluster')

    def _make_dir(self, dir_, dir_type):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        par_dir = os.path.dirname(cur_dir)
        dir_ = os.path.join(par_dir, dir_)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        
        if dir_type == 'save' or dir_type == 'result':
            this_dir = os.path.join(dir_, f'{self._config.model}_{self._config.cluster_model}_{self._config.cluster_distance}_pf{self._config.prefix}'
                                    +f'_cws{self._config.cluster_window_size}_czd{self._config.cluster_z_dims}_cme{self._config.cluster_max_epoch}_cbs{self._config.cluster_batch_size}'
                                    +f'_ws{self._config.window_size}_zd{self._config.z_dims}_gme{self._config.global_max_epoch}_lme{self._config.local_max_epoch}_bs{self._config.batch_size}')
        elif dir_type == 'cluster':
            this_dir = os.path.join(dir_, f'{self._config.cluster_model}_pf{self._config.prefix}'
                                    +f'_cws{self._config.cluster_window_size}_czd{self._config.cluster_z_dims}_cme{self._config.cluster_max_epoch}_cbs{self._config.cluster_batch_size}')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)

        return this_dir

    @property
    def config(self):
        return self._config

def get_data_dim(dataset):
    if dataset == 'smap':
        return 25
    elif dataset == 'msl':
        return 55
    elif str(dataset).startswith('smd'):
        return 38
    elif str(dataset).startswith('asd'):
        return 19
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('service'):
        return 3
    elif str(dataset).startswith('explore'):
        return 24
    elif str(dataset).startswith('new'):
        return 19
    elif str(dataset).startswith('ZTE'):
        return int(str(dataset).split('-')[1])
    elif dataset == 'too_much_na':
        return 34
    elif dataset == 'prepare_data':
        return 3
    elif dataset == 'remove_anomaly_top_point_to_lower_thres_202106':
        return 20
    elif dataset == '202107_C100120001_small_disturbance_should_not_report_err':
        return 3
    elif dataset == '202108_history_and_yesterday_thres_diff_weight':
        return 12
    elif str(dataset).startswith('History_Performance_Host_NIC_20201219160457'):
        if dataset.split('_')[-1] != 'all':
            return 10
        else:
            return 120
    else:
        raise ValueError('unknown dataset '+str(dataset))

def get_data(dataset, max_train_size=None, max_test_size=None, do_preprocess=True, train_start=0,
             test_start=0, prefix="processed", x_dims=None, quntile=None, verbose=False):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    
    if verbose:
        print('load data of:', dataset)
        print("train: ", train_start, train_end)
        print("test: ", test_start, test_end)

    if x_dims is None:
        x_dim = get_data_dim(dataset)
    else:
        x_dim = x_dims
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f)
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f) 
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f)
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    train_data, test_data = preprocess(train_data, test_data, do_preprocess, quntile)
    
    if verbose:
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
    if test_label is not None and verbose:
        print("test label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)

def preprocess(df_train, df_test, do_preprocess, quntile=None):
    """
    normalize raw data
    """
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num(df_test)
    if do_preprocess:
        if quntile is None:
            scaler = MinMaxScaler()
            scaler = scaler.fit(df_train)
            df_train = scaler.transform(df_train)
            df_test = scaler.transform(df_test)
        else:
            df_min = np.percentile(np.sort(df_train, axis=0), 100 - quntile, axis=0)
            df_max = np.percentile(np.sort(df_train, axis=0), quntile, axis=0)

            scaler = df_max - df_min
            scaler[np.where(scaler == 0)[0]] = 1

            df_train = (df_train - df_min) / scaler
            df_test = (df_test - df_min) / scaler

    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    return df_train, df_test

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def paint(datas, scores=None, names=None, on_dim=False, thresholds=None, store_path=None):
    data = np.array(datas, dtype=float)
    scores = np.array(scores, dtype=float)

    total_row = data.shape[0]
    
    if on_dim:
        pair_num = 2
    else:
        pair_num = 1

    if scores is not None:
        if on_dim:
            total_row *= pair_num
            total_row += 1
        else:
            total_row += 1

    fig = plt.figure(figsize=(60, 60))
    for i in range(1, data.shape[0] + 1):

        ax = fig.add_subplot(total_row, 1, (i - 1) * pair_num + 1)
        ax.plot(data[i - 1], color='black', lw=4)
        if names is None:
            ax.set_ylabel('dim ' + str(i-1), rotation=0, labelpad=20)
        else:
            ax.set_ylabel('dim ' + str(names[i-1]), rotation=0, labelpad=20)
        
        if on_dim:
            ax = fig.add_subplot(total_row, 1, (i - 1) * pair_num + 2)
            ax.plot(scores[i - 1])
            if names is None:
                ax.set_ylabel('score ' + str(i-1), rotation=0, labelpad=20)
            else:
                ax.set_ylabel('score ' + str(names[i-1]), rotation=0, labelpad=20)
    
    ax = fig.add_subplot(total_row, 1, total_row)
    if on_dim:
        scores = np.sum(scores, axis=0)
    scores = np.reshape(scores, -1)
    ax.plot(scores, color='blue')

    if thresholds is not None:
        ax.plot(thresholds, color='red')
        anomaly_point = []
        for i in range(len(scores)):
            if scores[i] > thresholds[i]:
                anomaly_point.append(i)
        ax.scatter(anomaly_point, scores[anomaly_point], color='red', marker='.')

    if store_path is not None:
        plt.savefig(store_path, dpi=300)
    else:
        plt.show()

def pot_detect(init_score, score, q=1e-3, level=0.98):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, verbose=False)  # initialization step
    ret = s.run(dynamic=False)  # run
    print('total ' + str(len(ret['alarms'])) + ' anomaly points detect')

    return ret
