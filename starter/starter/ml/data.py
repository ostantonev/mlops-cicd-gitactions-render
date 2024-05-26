import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from pydantic import BaseModel, Field

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

class Person(BaseModel):
# see https://archive.ics.uci.edu/dataset/20/census+income
    age: int = Field(default=40)
    workclass: str = Field(default="Private")
    fnlgt: int = Field(default=200000)
    education: str = Field(default="11th")
    education_num: int = Field(default=11, alias="education-num")
    marital_status: str = Field(default="Separated", alias="marital-status")
    occupation: str = Field(default="Sales")
    race: str = Field(default="White")
    relationship: str = Field(default="Wife")
    sex: str = Field(default="Female")
    capital_gain: int = Field(default=5000, alias="capital-gain")
    capital_loss: int = Field(default=800, alias="capital-loss")
    hours_per_week: int = Field(default=45, alias="hours-per-week")
    native_country: str = Field(default="United-States", alias="native-country")

    class Config:
        # see https://fastapi.tiangolo.com/tutorial/schema-extra-example/
        schema_extra = {
            "examples": [
                {
                    "age": 40,
                    "workclass": "Private",
                    "fnlwgt": 200000,
                    "education": "11th",
                    "education_num": 11,
                    "marital_status": "Separated",
                    "occupation": "Sales",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 5000,
                    "capital_loss": 800,
                    "hours_per_week": 45,
                    "native_country": "United-States",
                }
            ]
        }


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we"re doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
