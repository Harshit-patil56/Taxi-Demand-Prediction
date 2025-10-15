import mlflow
import dagshub
import json
import pandas as pd
import joblib
import shutil
from pathlib import Path
import logging
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error

# ---------------------------
# Init DagsHub + MLflow
# ---------------------------
dagshub.init(repo_owner='shit192004', repo_name='Taxi-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/shit192004/Taxi-Demand-Prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")
set_config(transform_output="pandas")

# ---------------------------
# Logger
# ---------------------------
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------------------
# Helpers
# ---------------------------
def load_model(path):
    return joblib.load(path)

def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {"run_id": run_id, "artifact_path": artifact_path, "model_uri": model_uri}
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"

    # Read test data
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")

    df.set_index("tpep_pickup_datetime", inplace=True)
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"]

    # Load encoder & transform
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")

    # Load model & predict
    model_path = root_path / "models/model.joblib"
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    y_pred = model.predict(X_test_encoded)

    # Compute loss
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Loss: {loss}")

    # Minimal MLflow logging (safe for DagsHub)
    with mlflow.start_run(run_name="model"):
        # params & metric
        try:
            mlflow.log_params(model.get_params())
        except Exception:
            # some sklearn models (or wrappers) might not expose get_params the same way; ignore if fails
            logger.info("Could not log model params via get_params().")

        mlflow.log_metric("MAPE", float(loss))

        # model signature (optional but helpful)
        try:
            model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
        except Exception:
            model_signature = None

        # Save model locally to a clean folder, then upload the folder as artifacts
        models_dir = root_path / "tmp_logged_models" / "demand_prediction"
        if models_dir.exists():
            shutil.rmtree(models_dir)
        models_dir.parent.mkdir(parents=True, exist_ok=True)

        # save_model writes files locally (no registry API calls)
        if model_signature is not None:
            mlflow.sklearn.save_model(sk_model=model, path=str(models_dir), signature=model_signature)
        else:
            mlflow.sklearn.save_model(sk_model=model, path=str(models_dir))

        # Upload the saved folder as artifacts (DagsHub supports artifact upload)
        mlflow.log_artifacts(str(models_dir), artifact_path="demand_prediction")

        # collect run info and write JSON
        run = mlflow.active_run()
        run_id = run.info.run_id
        artifact_path = "demand_prediction"
        model_uri = f"runs:/{run_id}/{artifact_path}"

        json_file_save_path = root_path / "run_information.json"
        save_run_information(run_id=run_id, artifact_path=artifact_path, model_uri=model_uri, path=json_file_save_path)
        logger.info("Mlflow logging complete and run_information.json saved")


