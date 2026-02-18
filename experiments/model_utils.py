from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
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
    # Confusion Matrix (GENERALIZED)
    # ---------------------------
    def plot_confusion_matrix(self, model_name, label_encoder, dataset_name):
        metrics = self.results[model_name]["metrics"].get(dataset_name)

        if metrics is None:
            raise ValueError(f"No metrics stored for '{dataset_name}'")

        cm = metrics["confusion_matrix"]

        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create ONE figure with TWO subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # -------------------
        # Raw Counts
        # -------------------
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_
        )
        disp.plot(ax=axes[0], cmap="Blues", xticks_rotation=90, values_format="d")
        axes[0].set_title(f"Confusion Matrix (Counts)\n{model_name} | {dataset_name}")

        # -------------------
        # Normalized
        # -------------------
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=label_encoder.classes_
        )
        disp_norm.plot(ax=axes[1], cmap="Blues", xticks_rotation=90, values_format=".0%")
        axes[1].set_title(f"Confusion Matrix (Normalized)\n{model_name} | {dataset_name}")

        plt.tight_layout()
        plt.show()


    def get_model(self, model_name):
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found.")
        
        return self.results[model_name]["model"]
