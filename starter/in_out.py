#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pickle
import pandas as pd
from logger_config import log


def load_dataframe_from_file(file_path):
    """
    Reading Data  from the file into Dataset, uses ", " as separator

    Parameters
    ----------
    file_path : str
        input csv file name
        
    Returns
    -------
        
    """
    #check for datasets, compile them together, and write to an output file
    log.info(f"reading data from {file_path}")
    return pd.read_csv(file_path, sep=', ', engine='python')


def load_f1_score(latestscore_txt_file):
    with open(latestscore_txt_file, 'r') as file:
        old_f1_score = float(file.readline().strip())
    log.info(f"Reading old f1 score: {old_f1_score} from {latestscore_txt_file}")
    return old_f1_score


def load_X_y(data_path):
    log.info(f"loading data from {data_path}")
    X = pd.read_csv(data_path).drop(columns=['corporation'])
    y = X.pop('exited')
    return X, y


def load_model(pickle_model_path):
    """
    Loads a machine learning model from a file.

    Parameters
    ----------
    pickle_model_path : str
        The path to the file containing the model.

    Returns
    -------
    model
        The loaded machine learning model.
    """
    log.info(f"loading model from {pickle_model_path}")
    with open(pickle_model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, pickle_model_path):
    """
    Stores a machine learning model to a file.

    Parameters
    ----------
    model : ???
        Trained machine learning model.
    pickle_model_path : str
        The path to the file where the model should be stored.
    """
    log.info(f"saving model {pickle_model_path}")
    with open(pickle_model_path, "wb") as f:
        pickle.dump(model, f)


def write_list_to_file(str_list, file_path):
    log.info(f"writing to file {file_path}")
    with open(file_path, 'w') as f:
        for file_path in str_list:
            f.write('%s\n' % file_path)


