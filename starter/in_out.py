#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pickle
import pandas as pd
from logger_config import log

def to_full_path(path):

    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, path)
    log.info(f"current working directory is {cwd}, full path for {path} is {absolute_path}")
    return absolute_path

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


def load_artifact(artifact_path):
    """
    Loads from a file.

    Parameters
    ----------
    artifact_path : str
        The path to the file containing the artifact.

    Returns
    -------
    artifact
        The loaded machine learning artifact.
    """
    full_path=to_full_path(artifact_path)
    log.info(f"loading from {full_path}")
    with open(full_path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact


def save_artifact(artifact, artifact_path):
    """
    Stores to a file.

    Parameters
    ----------
    artifact : ???
        Trained machine learning artifact.
    artifact_path : str
        The path to the file where the artifact should be stored.
    """
    log.info(f"saving {artifact_path}")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)


def write_list_to_file(str_list, file_path):
    log.info(f"writing to file {file_path}")
    with open(file_path, 'w') as f:
        for file_path in str_list:
            f.write('%s\n' % file_path)


