from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import plotly.figure_factory as ff
from visualizations import *

class ModelTrainer:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}

    # ---------------------------
    # Train single model
    # ---------------------------
    def train(self, model, model_name):
        model.fit(self.X_train, self.y_train)

        self.results[model_name] = {
            "model": model,
            "metrics": {}
        }

        return model

    # ---------------------------
    # Evaluate
    # ---------------------------
    def evaluate(self, model, X, y):
        y_pred = model.predict(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="macro"),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1_score": f1_score(y, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y, y_pred)
        }

    # ---------------------------
    # Evaluate and store (GENERALIZED)
    # ---------------------------
    def evaluate_and_store(self, model_name, X, y, dataset_name):

        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.results[model_name]["model"]

        metrics = self.evaluate(model, X, y)

        self.results[model_name]["metrics"][dataset_name] = metrics
    
    # ---------------------------
    # Grid Search
    # ---------------------------
    def grid_search(self, models_with_params_dict, cv=3, scoring="f1_macro"):
        """
        Perform GridSearch for multiple models and store best estimators.
        """

        for model_name, v in models_with_params_dict.items():

            grid = GridSearchCV(
                estimator=v["model"],
                param_grid=v["params"],
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )

            grid.fit(self.X_train, self.y_train)

            best_model = grid.best_estimator_

            # Store using SAME structure as train()
            self.results[model_name] = {
                "model": best_model,
                "metrics": {},  # prepare for train/dev/test storage
                "best_params": grid.best_params_,
                "best_cv_score": grid.best_score_
            }

        return self.results


    # ---------------------------
    # Print Results
    # ---------------------------
    def print_results(self, dataset_name):

        for model_name, info in self.results.items():

            metrics = info["metrics"].get(dataset_name)

            if metrics is None:
                print(f"\n⚠ No metrics stored for {model_name} on {dataset_name}")
                continue

            print(f"\n📌 {model_name.upper()} on {dataset_name}")

            for metric, value in metrics.items():
                if metric != "confusion_matrix":
                    print(f"{metric}: {value:.4f}")

    # ---------------------------
    # Confusion Matrix
    # ---------------------------

    def plot_confusion_matrix(
        trainer,
        model_name,
        label_encoder,
        dataset_name="development",
        cell_size=80
    ):

        metrics = trainer.results[model_name]["metrics"].get(dataset_name)
        if metrics is None:
            raise ValueError(f"No metrics stored for '{dataset_name}'")

        cm = metrics["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        classes = label_encoder.classes_.tolist()
        n_classes = len(classes)

        annotations = []
        for i in range(n_classes):
            row = []
            for j in range(n_classes):
                text = f"{cm[i,j]}<br>({cm_norm[i,j]*100:.1f}%)"
                row.append(text)
            annotations.append(row)

        fig = ff.create_annotated_heatmap(
            z=cm_norm,
            x=classes,
            y=classes,
            annotation_text=annotations,
            colorscale="Blues",
            showscale=True
        )

        fig.update_layout(
            title={
                "text": f"{model_name} | {dataset_name} Confusion Matrix",
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },

            # Remove default xaxis_title
            xaxis=dict(side="top"),

            # Keep Y axis label normally
            yaxis_title="True Label",

            # Dynamic size
            width=n_classes * cell_size,
            height=n_classes * cell_size,

            # Extra spacing
            margin=dict(t=150, l=100, r=40, b=80),

            annotations=fig.layout.annotations + (
                dict(
                    text="Predicted Label",
                    x=0.5,
                    y=1.084,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                ),
            )
        )

        return fig


    def get_model(self, model_name):
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found.")
        
        return self.results[model_name]["model"]
