# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model
from logger_config import log
import in_out
# Add the necessary imports for the starter code.

# Add code to load in the data.
data=in_out.load_dataframe_from_file('data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Train and save a model.
model = train_model(X_train, y_train)
in_out.save_model(model, 'model/trained_model.pkl')