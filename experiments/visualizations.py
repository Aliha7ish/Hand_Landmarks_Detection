import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from matplotlib.lines import Line2D

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

def plot_correct_landmarks(model, df_features, y_true, le, n_samples=9):
    """
    Plot hand skeletons of correctly classified samples.

    Parameters
    ----------
    model : sklearn-like model
        Trained classifier with .predict()
    df_features : pd.DataFrame
        DataFrame containing x1..x21, y1..y21
    y_true : np.ndarray or pd.Series
        True labels (encoded)
    le : LabelEncoder
        LabelEncoder for decoding labels
    n_samples : int
        Number of samples to plot
    """
    y_pred = model.predict(df_features)
    correct_idx = np.where(y_pred == y_true)[0]

    if len(correct_idx) == 0:
        print("No correct predictions found!")
        return

    selected_idx = np.random.choice(correct_idx, size=min(n_samples, len(correct_idx)), replace=False)

    cols = int(np.sqrt(n_samples))
    rows = int(np.ceil(len(selected_idx)/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, idx in enumerate(selected_idx):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        xs, ys = extract_hand_landmarks(df_features.iloc[idx])
        plot_hand_skeleton(xs, ys, ax)

        true_label = le.inverse_transform([y_true[idx]])[0]
        ax.set_title(f"{true_label}", fontsize=10)

    # Hide any unused subplots
    for j in range(len(selected_idx), rows*cols):
        row = j // cols
        col = j % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_two_class_table(model, X, y, label_encoder, class1, class2, n_per_cell=4, cell_padding=1.5):
    """
    Plot a beautiful 2x2 table for two classes with hand skeletons.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        True labels
    label_encoder : LabelEncoder
        To decode class names
    class1 : str
        First class
    class2 : str
        Second class
    n_per_cell : int
        Number of images per cell
    cell_padding : float
        Spacing multiplier between mini-images
    """
    cl_a = label_encoder.transform([class1])[0]
    cl_b = label_encoder.transform([class2])[0]

    # Predictions
    y_pred = model.predict(X)

    # Four cases
    if isinstance(X, pd.DataFrame):
        X_aa = X[(y == cl_a) & (y_pred == cl_a)]
        X_ab = X[(y == cl_a) & (y_pred == cl_b)]
        X_ba = X[(y == cl_b) & (y_pred == cl_a)]
        X_bb = X[(y == cl_b) & (y_pred == cl_b)]
    else:
        X_aa = X[(y == cl_a) & (y_pred == cl_a)]
        X_ab = X[(y == cl_a) & (y_pred == cl_b)]
        X_ba = X[(y == cl_b) & (y_pred == cl_a)]
        X_bb = X[(y == cl_b) & (y_pred == cl_b)]

    cases = [X_aa, X_ab, X_ba, X_bb]
    # case_titles = ["True A / Pred A", "True A / Pred B", "True B / Pred A", "True B / Pred B"]

    # Create main 2x2 grid
    fig, main_axes = plt.subplots(2, 2, figsize=(10, 8))
    main_axes = main_axes.flatten()

    for idx, (X_case, ax) in enumerate(zip(cases, main_axes)):
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(case_titles[idx], fontsize=10)

        # Mini-grid inside each cell
        rows = int(np.ceil(np.sqrt(n_per_cell)))
        cols = rows

        for i, row_data in enumerate(X_case[:n_per_cell]):
            xs, ys = extract_hand_landmarks(row_data)

            # Compute offsets for mini-grid
            r = i // cols
            c = i % cols
            offset_x = c * cell_padding * 50
            offset_y = r * cell_padding * 50

            ax.scatter(xs + offset_x, ys + offset_y, s=20, c='red')
            for start, end in HAND_CONNECTIONS:
                ax.plot([xs[start]+offset_x, xs[end]+offset_x],
                        [ys[start]+offset_y, ys[end]+offset_y],
                        linewidth=1, c='blue')

        ax.invert_yaxis()
        ax.axis("off")

    # Add True/Predicted labels outside grid
    fig.text(0.5, 0.95, "Predicted Label", ha='center', fontsize=14)
    fig.text(0.05, 0.5, "True Label", va='center', rotation='vertical', fontsize=14)

    # Add ticks with class names
    fig.text(0.27, 0.97, class1, ha='center', fontsize=12)
    fig.text(0.73, 0.97, class2, ha='center', fontsize=12)
    fig.text(0.01, 0.75, class1, va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.25, class2, va='center', rotation='vertical', fontsize=12)

    # Draw separating lines between cells (table-like)
    fig.add_artist(Line2D([0.5, 0.5], [0.1, 0.9], color='black', linewidth=2, transform=fig.transFigure))
    fig.add_artist(Line2D([0.1, 0.9], [0.5, 0.5], color='black', linewidth=2, transform=fig.transFigure))

    plt.tight_layout(rect=[0.1, 0.1, 0.95, 0.9])
    plt.show()
