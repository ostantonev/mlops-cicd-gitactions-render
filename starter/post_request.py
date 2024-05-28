import json
import requests
from logger_config import log

person_data = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 193524,
    "education": "Doctorate",
    "education-num": 16,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native_country": " United-States",
}

# URL = "http://localhost:10000/"
URL = "https://mlops-cicd-gitactions-render.onrender.com/"


# response = requests.get(URL)
# log.info(f"response code={response.status_code}, response text={response.text}")

response = requests.post(URL+"predict", data=json.dumps(person_data))
log.info(f"response code={response.status_code}, response text={response.text}")