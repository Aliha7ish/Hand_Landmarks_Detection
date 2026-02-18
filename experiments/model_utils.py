from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from visualizations import *

def train_model(model, X_train, y_train):
    """
    Train the given model using the provided training data.
    
    :param model: sklearn model instance, the machine learning model to be trained.
    :param X_train: df.DataFrame, the training features.
    :param y_train: df.Series, The training labels

    return: the trained model. 
    """

    model.fit(X_train, y_train)

    return model
    
def evaluate_model(model, X, y):
    """
    Evaluate the performance of the trained model on the development set and return the evaluation metrics.
    
    :param model: trained model instance, the machine learning model to be evaluated.
    :param X: df.DataFrame, the development features.
    :param y: df.Series, The development labels.

    Returns:
    metrics: dict, dictionary containing the performance metrics as accuracy, precision, recall, and F1-score. 

    """

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro")
    recall = recall_score(y, y_pred, average="macro")
    f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)

    # strore them into metrics dict
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return metrics


def display_metrics(model, X, y, clf_name):
    rf_metrics_train = evaluate_model(model, X, y)

    for metric_name, metric_value in rf_metrics_train.items():
        if metric_name != "confusion_matrix":
            print(f"{clf_name} {metric_name} = {metric_value:.2f}")


def run_model_selection_pipeline(models_with_params_dict, 
                                 X_train, y_train, 
                                 X_val, y_val,
                                 cv=3,
                                 scoring="f1_macro"):
    
    results = {}

    for model_name, v in models_with_params_dict.items():

        grid = GridSearchCV(
            estimator=v["model"],
            param_grid=v["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        metrics = evaluate_model(best_model, X_val, y_val)

        results[model_name] = {
            "best_params": grid.best_params_,
            "best_cv_score": grid.best_score_,
            "model": best_model,
            "metrics": metrics
        }

    return results



def plot_confusion_matrix(cm, label_encoder, model_name=None):
    """
    Plot raw and normalized confusion matrices.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (raw counts)
    label_encoder : LabelEncoder
        Fitted LabelEncoder for decoding class names
    """
    # Raw counts
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format="d")
    plt.title("Confusion Matrix (Counts)")
    plt.show()

    # Normalized percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                       display_labels=label_encoder.classes_)
    disp_norm.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format=".0%")
    plt.title(f"Confusion Matrix (Normalized) | {model_name}")
    plt.show()

def plot_two_class_misclassified_grid(model, X, y_true, label_encoder, class1, class2, n_per_cell=1):
    """
    Plot a 2x2 mini confusion matrix for two selected classes with images.

    Parameters
    ----------
    model : sklearn-like model
        Fitted classifier with .predict().
    X : pd.DataFrame or np.ndarray
        Feature data (hand landmarks).
    y_true : np.ndarray
        True labels (encoded).
    label_encoder : LabelEncoder
        Fitted LabelEncoder to decode class names.
    class1, class2 : str
        Names of the two classes to compare.
    n_per_cell : int
        Number of images to show per cell.
    """
    # Encode the classes
    c1_label = label_encoder.transform([class1])[0]
    c2_label = label_encoder.transform([class2])[0]

    y_pred = model.predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    # Grid positions for True vs Pred
    grid_positions = [
        (c1_label, c1_label),  # True A, Pred A
        (c1_label, c2_label),  # True A, Pred B
        (c2_label, c1_label),  # True B, Pred A
        (c2_label, c2_label),  # True B, Pred B
    ]

    for i, (true_lbl, pred_lbl) in enumerate(grid_positions):
        idxs = np.where((y_true == true_lbl) & (y_pred == pred_lbl))[0]
        axes[i].axis('off')  # hide axis by default

        if len(idxs) == 0:
            axes[i].set_title(f"T: {label_encoder.inverse_transform([true_lbl])[0]}\n"
                              f"P: {label_encoder.inverse_transform([pred_lbl])[0]}\n(No samples)")
            continue

        # Pick up to n_per_cell examples
        for j, idx in enumerate(idxs[:n_per_cell]):
            row = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            xs, ys = extract_hand_landmarks(row)
            axes[i].scatter(xs, ys, c='red')
            for start, end in HAND_CONNECTIONS:
                axes[i].plot([xs[start], xs[end]], [ys[start], ys[end]], c='black')

            axes[i].invert_yaxis()
            axes[i].axis('off')
            axes[i].set_title(f"T: {label_encoder.inverse_transform([true_lbl])[0]}\n"
                              f"P: {label_encoder.inverse_transform([pred_lbl])[0]}")

    plt.tight_layout()
    plt.show()
