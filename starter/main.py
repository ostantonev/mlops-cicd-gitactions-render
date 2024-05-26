from fastapi import FastAPI
import pandas as pd
from logger_config import log
import in_out
from starter.ml.data import process_data, cat_features, Person

model=in_out.load_artifact('model/trained_model.pkl')
encoder=in_out.load_artifact('model/encoder.pkl')
lb=in_out.load_artifact('model/lb.pkl')

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello":"World(main)"}

@app.post("/predict")
def predict(person: Person):
    X = pd.DataFrame([person.dict()])
    log.debug(f"got predict request  for person {person}")


    # data preprocessing
    X_preporcessed, _, _, _ = process_data(
        X,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    log.debug(f"request preporcessed to {X_preporcessed}")

    # prediction
    prediction = model.predict(X_preporcessed)
    return {"salary_prediction": "<=50K" if prediction == 0 else ">50K"}

