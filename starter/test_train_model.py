import os
import pytest
from train_model import main

@pytest.fixture(scope="module")
def trained_model():
    # Run the main function to train the model
    main()

    # Check if trained model file exists
    model_path = "model/trained_model.pkl"
    assert os.path.exists(model_path)

    # Return the path to the trained model
    return model_path

def test_trained_model_exists(trained_model):
    # Check if the trained model file exists
    assert os.path.exists(trained_model)

def test_trained_model_is_not_empty(trained_model):
    # Check if the trained model file is not empty
    assert os.path.getsize(trained_model) > 0