import mlflow
import dagshub
import json

import dagshub
dagshub.init(repo_owner='shit192004', repo_name='Taxi-Demand-Prediction', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/shit192004/Taxi-Demand-Prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

# set model name
model_path = load_model_information("run_information.json")["model_uri"]

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)


def test_load_model_from_registry():
    assert model is not None, "Failed to load model from registry"
    