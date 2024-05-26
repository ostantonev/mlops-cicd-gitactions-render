from fastapi.testclient import TestClient
from main import app
from logger_config import log

# Create client test
client = TestClient(app)

def test_alive_client():
    """
        Test the server is alive
    """
    response = client.get("/")
    log.info(f"Response for / is {response.text}")
    # Test successfull response
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    assert response.json() == {"Hello": "World(main)"}

def test_predict_endpoint():
    # Create a test client using TestClient
    client = TestClient(app)
    
    # Create a sample Person object to send in the request body
    person_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    
    # Make a POST request to the /predict endpoint with the sample Person data
    response = client.post("/predict", json=person_data)
    log.info(f"Response for /predict is {response.text}")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"    
    # Assert that the response contains the expected prediction result
    assert response.json() == {"salary_prediction": "<=50K"}
