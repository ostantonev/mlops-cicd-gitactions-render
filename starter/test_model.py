import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as rfc
from starter.ml.model import train_model, compute_model_metrics

def test_train_model():
    # Generate dummy data for testing
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Check if the model is trained successfully
    assert isinstance(model, rfc)

def test_compute_model_metrics():
    # Generate dummy data for testing
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    
    # Compute model metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # Check if precision, recall, and fbeta are floats
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    
    # Check if precision, recall, and fbeta are within valid ranges
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1