import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as rfc
import starter.ml.data as data
from logger_config import log
import in_out

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = rfc(random_state=42)
    model.fit(X_train, y_train)
    return model

def inference(X, model, encoder, lb, label=None):
    # data preprocessing
    X_preporcessed, y_preprocessed, _, _ = data.process_data(
        X,
        categorical_features=data.cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    log.debug(f"features preporcessed to {X_preporcessed}")

    # prediction
    prediction = model.predict(X_preporcessed)
    return y_preprocessed, prediction


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_on_dataslices(test, y_test, pred_test):
    output = in_out.generate_table_header()

    test['y_test'] = y_test
    test['pred_test'] = pred_test

    for category in data.cat_features:
        slices = np.unique(test[category])
        log.info(f"Calculating metrics for slices {slices} in category {category}")

        for slice in slices:
            y_test_slice = test[test[category] == slice]

            # number of rows in testset with given slice value
            slice_shape = y_test_slice.shape[0]
            log.debug(f"slice contains {slice_shape:>5} rows with {category}.{slice}")

            # if any testdata for the given slice in testset
            if slice_shape != 0:
                # Proces the test data with the process_data function.
                precision, recall, fbeta = compute_model_metrics(
                    y_test_slice['y_test'], y_test_slice['pred_test']
                 )
                output += in_out.new_row(category, slice, slice_shape, precision, recall, fbeta)
    return output


