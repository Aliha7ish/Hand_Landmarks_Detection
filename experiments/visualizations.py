import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import denormalize_landmarks


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def extract_hand_landmarks(row):
    """
    Extract (x, y) hand landmarks from a dataframe row or numpy array.

    Parameters
    ----------
    row : pd.Series or np.ndarray
        One sample containing x1..x21, y1..y21

    Returns
    -------
    xs : np.ndarray
        X coordinates of 21 landmarks
    ys : np.ndarray
        Y coordinates of 21 landmarks
    """
    if isinstance(row, pd.Series):
        xs = np.array([row[f"x{i}"] for i in range(1, 22)])
        ys = np.array([row[f"y{i}"] for i in range(1, 22)])
    else:  # numpy array
        xs = row[0:42:2]  # x1, x2, ... x21
        ys = row[1:42:2]  # y1, y2, ... y21
    return xs, ys


def plot_hand_skeleton(xs, ys, ax):
    """
    Plot a hand skeleton using MediaPipe landmark connections.

    Parameters
    ----------
    xs : np.ndarray
        X coordinates of landmarks
    ys : np.ndarray
        Y coordinates of landmarks
    ax : matplotlib.axes.Axes
        Axis to plot on
    """
    ax.scatter(xs, ys, s=20)

    for start, end in HAND_CONNECTIONS:
        ax.plot(
            [xs[start], xs[end]],
            [ys[start], ys[end]],
            linewidth=1
        )

    ax.invert_yaxis()
    ax.axis("off")


def visualize_class_samples(df, label, n_samples=5):
    """
    Visualize hand landmark samples for a single class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hand landmarks and labels
    label : str
        Class label to visualize
    n_samples : int
        Number of samples to visualize
    """
    samples = df[df["label"] == label].sample(n_samples, random_state=42)

    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    fig.suptitle(f"Class: {label}", fontsize=14)

    for ax, (_, row) in zip(axes, samples.iterrows()):
        xs, ys = extract_hand_landmarks(row)
        plot_hand_skeleton(xs, ys, ax)

    plt.tight_layout()
    plt.show()


def visualize_samples_per_class(df, n_samples=5):
    """
    Visualize hand landmark samples for each class in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hand landmarks and labels
    n_samples : int
        Number of samples per class
    """
    class_order = df["label"].value_counts().index

    for label in class_order:
        visualize_class_samples(df, label, n_samples)


def plot_class_distribution_bar(df, label_col="label", figsize=(10, 6)):
    """
    Plot the number of samples per class using a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing class labels
    label_col : str
        Name of the label column
    figsize : tuple
        Figure size
    """
    class_counts = df[label_col].value_counts().reset_index()
    class_counts.columns = ["label", "count"]

    plt.figure(figsize=figsize)
    sns.barplot(
        data=class_counts,
        x="count",
        y="label"
    )

    plt.title("Samples per Class (Count)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Class Label")
    plt.tight_layout()
    plt.show()


def plot_class_distribution_pie(df, label_col="label", figsize=(8, 8)):
    """
    Plot the percentage distribution of samples per class using a pie chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing class labels
    label_col : str
        Name of the label column
    figsize : tuple
        Figure size
    """
    class_counts = df[label_col].value_counts()
    percentages = class_counts / class_counts.sum() * 100

    plt.figure(figsize=figsize)
    plt.pie(
        percentages,
        labels=percentages.index,
        autopct="%.1f%%",
        startangle=90
    )

    plt.title("Samples per Class (Percentage)")
    plt.axis("equal")  # Ensures circular pie
    plt.tight_layout()
    plt.show()




def plot_correct_landmarks(
    model,
    df_features,
    y_true,
    le,
    model_name="Model",
    dataset_name="Test",
    accuracy=None,
    n_samples=9
):

    y_pred = model.predict(df_features)
    correct_idx = np.where(y_pred == y_true)[0]

    if len(correct_idx) == 0:
        print("No correct predictions found!")
        return

    selected_idx = np.random.choice(
        correct_idx,
        size=min(n_samples, len(correct_idx)),
        replace=False
    )

    cols = int(np.sqrt(n_samples))
    rows = int(np.ceil(len(selected_idx) / cols))

    subplot_titles = []

    for idx in selected_idx:
        true_label = le.inverse_transform([y_true[idx]])[0]
        subplot_titles.append(true_label)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    for i, idx in enumerate(selected_idx):

        row = i // cols + 1
        col = i % cols + 1

        if hasattr(df_features, "iloc"):
            row_data = df_features.iloc[idx]
        else:
            row_data = df_features[idx]

        xs, ys = extract_hand_landmarks(row_data)

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=4),
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Hand connections
        for start, end in HAND_CONNECTIONS:
            fig.add_trace(
                go.Scatter(
                    x=[xs[start], xs[end]],
                    y=[ys[start], ys[end]],
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, autorange="reversed", row=row, col=col)

    # Hide unused cells
    total_cells = rows * cols
    for j in range(len(selected_idx), total_cells):
        row = j // cols + 1
        col = j % cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    # -----------------------------
    # Main Title
    # -----------------------------
    main_title = f"{model_name} | {dataset_name} | Correct Predictions"
    if accuracy is not None:
        main_title += f"<br>Accuracy: {accuracy:.4f}"

    fig.update_layout(
        height=rows * 250,
        width=cols * 250,
        title=dict(
            text=main_title,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(t=120),
        showlegend=False
    )

    fig.show()

    return fig



def plot_two_class_table(
    model,
    X,
    y,
    label_encoder,
    class1,
    class2,
    model_name="Model",
    dataset_name="Test",
    f1_score=None,
    n_per_cell=4,
    cell_padding=1.5
):

    cl_a = label_encoder.transform([class1])[0]
    cl_b = label_encoder.transform([class2])[0]

    y_pred = model.predict(X)

    if isinstance(X, np.ndarray):
        X_aa = X[(y == cl_a) & (y_pred == cl_a)]
        X_ab = X[(y == cl_a) & (y_pred == cl_b)]
        X_ba = X[(y == cl_b) & (y_pred == cl_a)]
        X_bb = X[(y == cl_b) & (y_pred == cl_b)]
    else:
        X_aa = X[(y == cl_a) & (y_pred == cl_a)].values
        X_ab = X[(y == cl_a) & (y_pred == cl_b)].values
        X_ba = X[(y == cl_b) & (y_pred == cl_a)].values
        X_bb = X[(y == cl_b) & (y_pred == cl_b)].values

    cases = [X_aa, X_ab, X_ba, X_bb]

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=[
            f"True {class1} / Pred {class1}",
            f"True {class1} / Pred {class2}",
            f"True {class2} / Pred {class1}",
            f"True {class2} / Pred {class2}",
        ]
    )

    for idx, X_case in enumerate(cases):

        row = idx // 2 + 1
        col = idx % 2 + 1

        rows_mini = int(np.ceil(np.sqrt(n_per_cell)))
        cols_mini = rows_mini

        for i, row_data in enumerate(X_case[:n_per_cell]):

            # xs, ys = extract_hand_landmarks(row_data)
            xs, ys = denormalize_landmarks(row_data)

            r = i // cols_mini
            c = i % cols_mini
            offset_x = c * cell_padding * 50
            offset_y = r * cell_padding * 50

            plot_scale = 100  # tweak until hands are visible
            xs = xs * plot_scale + offset_x
            ys = ys * plot_scale + offset_y

            # Points
            fig.add_trace(
                go.Scatter(
                    x=xs + offset_x,
                    y=ys + offset_y,
                    mode="markers",
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=row,
                col=col
            )

            # Connections
            for start, end in HAND_CONNECTIONS:
                fig.add_trace(
                    go.Scatter(
                        x=[xs[start] + offset_x, xs[end] + offset_x],
                        y=[ys[start] + offset_y, ys[end] + offset_y],
                        mode="lines",
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, autorange="reversed", row=row, col=col)

    # -----------------------------
    # Clean Main Title
    # -----------------------------
    main_title = f"{model_name} | {dataset_name} | {class1} vs {class2}"
    if f1_score is not None:
        main_title += f"<br>F1 Score: {f1_score:.4f}"

    fig.update_layout(
        height=900,
        width=1000,
        title=dict(
            text=main_title,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(t=120)
    )

    fig.show()

    return fig


def plot_dataset_split_distribution(y_train, y_dev, y_test, save_path=None):
    """
    Plot a bar chart showing the number of samples
    in train, development, and test sets using seaborn.

    Parameters
    ----------
    y_train : array-like
        Training labels
    y_dev : array-like
        Development labels
    y_test : array-like
        Test labels
    save_path : str, optional
        If provided, saves the figure to this path.
    """

    # Count samples
    data = {
        "Dataset Split": ["Train", "Development", "Test"],
        "Number of Samples": [len(y_train), len(y_dev), len(y_test)]
    }

    df = pd.DataFrame(data)

    # Styling
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="Dataset Split",
        y="Number of Samples"
    )

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)

    plt.title("Dataset Split Distribution", fontsize=14)
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.show()


def plot_model_performance_subplots(trainer, model_name, save_path=None):
    """
    Create 2x2 subplot visualization for Accuracy, Precision,
    Recall, and F1-score across Train, Dev, and Test sets.
    """

    sns.set_theme(style="whitegrid")

    # Extract metrics
    metrics = trainer.results[model_name]["metrics"]

    data = []
    for dataset in metrics:
        if dataset == "confusion_matrix":
            continue
        for metric_name, value in metrics[dataset].items():
            if metric_name != "confusion_matrix":
                data.append({
                    "Dataset": dataset,
                    "Metric": metric_name,
                    "Score": value
                })

    df = pd.DataFrame(data)

    metric_list = ["accuracy", "precision", "recall", "f1_score"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metric_list):
        ax = axes[i]

        subset = df[df["Metric"] == metric]

        sns.barplot(
            data=subset,
            x="Dataset",
            y="Score",
            ax=ax
        )

        # Increase y-limit slightly to prevent overlap
        ax.set_ylim(0, 1.08)

        # Add padding to subplot title
        ax.set_title(
            metric.replace("_", " ").capitalize(),
            pad=12
        )

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        sns.despine(ax=ax)


    # Clean main title only
    fig.suptitle(f"{model_name} Performance Across Datasets", fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_model_comparison_radar(trainer, dataset_name="development"):
    """
    Plot a radar chart comparing multiple models' performance metrics.

    Args:
        trainer (ModelTrainer): An instance of the trained ModelTrainer class.
        dataset_name (str): Dataset to use for metrics ('train', 'development', 'test').
    """

    metrics_names = ["accuracy", "precision", "recall", "f1_score"]

    fig = go.Figure()

    for model_name, info in trainer.results.items():
        metrics = info["metrics"].get(dataset_name)
        if metrics is None:
            continue

        # Collect metrics in the same order
        values = [metrics[m] for m in metrics_names]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the loop
            theta=metrics_names + [metrics_names[0]],
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # metrics are between 0 and 1
            )
        ),
        title=f"Model Comparison on {dataset_name.capitalize()} Set",
        showlegend=True,
        width=700,
        height=700
    )

    return fig