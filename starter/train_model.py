# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, cat_features
from starter.ml.model import train_model, inference, compute_model_metrics_on_dataslices
from logger_config import log
import in_out
# Add the necessary imports for the starter code.


def main():
    # Add code to load in the data.
    data=in_out.load_dataframe_from_file('data/census.csv')

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    in_out.save_artifact(encoder, 'model/encoder.pkl')
    in_out.save_artifact(lb, 'model/lb.pkl')

    # Train and save a model.
    model = train_model(X_train, y_train)
    in_out.save_artifact(model, 'model/trained_model.pkl')

    # calculate metrics on test data
    y_test, pred_test = inference(test, model, encoder, lb, label="salary")
    output_table= compute_model_metrics_on_dataslices(test, y_test, pred_test)
    log.info(f"output_table=\n{output_table}")
    in_out.write_list_to_file([output_table], 'model/slice_output.txt')


   

if __name__ == "__main__":
    main()