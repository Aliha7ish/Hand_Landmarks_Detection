import pandas as pd
import numpy as np

def load_data(path):
    """
    Load the train, dev, test datasets from the specified path 
    and return the features and labels for each set.
    
    path: str, the directory path where the CSV files are located.

    Returns:
    X_train, y_train, X_dev, y_dev, X_test, y_test: DataFrames and Series for features and labels.

    """
    
    # load train, dev, test sets
    train_df = pd.read_csv(f"{path}/hand_landmarks_train.csv")
    dev_df = pd.read_csv(f"{path}/hand_landmarks_dev.csv")
    test_df = pd.read_csv(f"{path}/hand_landmarks_test.csv")

    # unback the features and labels
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    
    X_dev = dev_df.drop(columns=["label"])
    y_dev = dev_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


# preprocessing steps
def preprocess_landmarks(data):
    """
    preprocessing function.
    
    Accepts:
    - Single sample (42,)
    - Numpy array (N, 42)
    - Pandas DataFrame (with or without 'label')
    
    Returns:
    - Same type structure (without label modification)
    """

    if isinstance(data, pd.DataFrame):
        
        df = data.copy()
        
        label = None
        if 'label' in df.columns:
            label = df['label']
            df = df.drop(columns=['label'])
        
        X = df.values
        X_processed = _process_numpy(X)
        
        df_processed = pd.DataFrame(X_processed, columns=df.columns)
        
        if label is not None:
            df_processed['label'] = label.values
            
        return df_processed
    
    elif isinstance(data, np.ndarray):
        return _process_numpy(data)

    else:
        raise TypeError("Input must be numpy array or pandas DataFrame")


def _process_numpy(X):
    
    X = np.array(X)
    
    # Single sample case (42,)
    if X.ndim == 1:
        X = X.reshape(1, -1)
        single_sample = True
    else:
        single_sample = False
    
    # Reshape to (N, 21, 2)
    X = X.reshape(-1, 21, 2)
    
    # recenter
    wrist = X[:, 0:1, :]        # shape (N,1,2)
    X = X - wrist
    
    # normalize
    middle_tip = X[:, 12, :]    # shape (N,2)
    scale = np.linalg.norm(middle_tip, axis=1).reshape(-1, 1, 1)
    
    scale[scale == 0] = 1  # avoid division by zero
    X = X / scale
    
    # Back to (N, 42)
    X = X.reshape(-1, 42)
    
    if single_sample:
        return X[0]
    
    return X

def denormalize_landmarks(row_data):
    """
    Denormalize a single preprocessed hand sample for plotting.
    
    Parameters
    ----------
    row_data : np.ndarray
        Shape (42,) for a single hand sample (preprocessed)
    
    Returns
    -------
    xs, ys : np.ndarray
        Denormalized x and y coordinates for plotting
    """
    # Reshape to (21,2)
    sample = row_data.reshape(21, 2)
    
    # Use middle finger tip to estimate original scale
    middle_tip = sample[12]
    scale = np.linalg.norm(middle_tip)
    if scale == 0:
        scale = 1.0
    
    xs = sample[:, 0] * scale
    ys = sample[:, 1] * scale
    
    return xs, ys
