from fastapi import FastAPI
import pandas as pd
from logger_config import log
import in_out
from starter.ml.data import Person
from starter.ml.model import inference


model=in_out.load_artifact('model/trained_model.pkl')
encoder=in_out.load_artifact('model/encoder.pkl')
lb=in_out.load_artifact('model/lb.pkl')

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello":"World(main)"}

@app.post("/predict")
def predict(person: Person):
    log.debug(f"got predict request  for person {person}")

    X = pd.DataFrame([person.dict()])
    _, prediction = inference(model, encoder, lb, X)

    return {"salary_prediction": "<=50K" if prediction == 0 else ">50K"}





