import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn

# -------------------------------------------------
# Set Experiment
# -------------------------------------------------
def set_experiment(experiment_name: str):
    mlflow.set_experiment(experiment_name)


# -------------------------------------------------
# Start Run Context Manager
# -------------------------------------------------
def start_run(run_name: str):
    return mlflow.start_run(run_name=run_name)


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
def log_model(model, artifact_path="model"):
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


# -------------------------------------------------
# Log Dataset (file or directory)
# -------------------------------------------------
def log_dataset(path: str, artifact_path="dataset"):
    """
    Log dataset file or folder.
    """
    mlflow.log_artifacts(path, artifact_path=artifact_path)


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
def log_dataframe_sample(df, artifact_name="dataset_sample.csv"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_path = os.path.join(tmp_dir, artifact_name)
        df.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path, artifact_path="dataset_sample")