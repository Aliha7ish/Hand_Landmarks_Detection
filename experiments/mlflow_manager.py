import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# -------------------------------------------------
# Set Experiment
# -------------------------------------------------
def set_experiment(experiment_name: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(experiment_name)


# -------------------------------------------------
# Start Run Context Manager
# -------------------------------------------------
def start_run(run_name: str):
    # nested=True allows a run to start even if a parent run is active
    return mlflow.start_run(run_name=run_name, nested=True)



# -------------------------------------------------
# Log Parameters
# -------------------------------------------------
def log_params(params: dict):
    mlflow.log_params(params)


# -------------------------------------------------
# Log Metrics
# -------------------------------------------------
def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


# -------------------------------------------------
# Log Model
# -------------------------------------------------
def log_model(model, X_example=None, artifact_path="model"):
    if X_example is not None:
        signature = infer_signature(X_example, model.predict(X_example))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X_example
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path
        )


# -------------------------------------------------
# Log Single Artifact (image, csv, etc.)
# -------------------------------------------------
def log_artifact(path: str, artifact_path="artifacts"):
    mlflow.log_artifact(path, artifact_path=artifact_path)


# -------------------------------------------------
# Log Label Encoder
# -------------------------------------------------
def log_label_encoder(label_encoder, artifact_path="label_encoder"):
    """
    Saves encoder temporarily then logs it.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        encoder_path = os.path.join(tmp_dir, "label_encoder.pkl")
        joblib.dump(label_encoder, encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path=artifact_path)


# -------------------------------------------------
# Log Sample DataFrame as CSV
# -------------------------------------------------
def log_dataframe(df, artifact_path="data", artifact_name="temp_dataset.csv"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_path = os.path.join(tmp_dir, artifact_name)
        df.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path, artifact_path=artifact_path)