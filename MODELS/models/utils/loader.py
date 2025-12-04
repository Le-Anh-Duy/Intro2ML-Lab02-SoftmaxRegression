import numpy as np

def load_dataset(path: str):
    """
    Load the dataset from a NumPy .npz file.

    Args:
        path (str): The file path to the dataset.

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test).
    """
    data = np.load(path)

    X_train = np.array(data["X_train"], dtype=np.float32)
    y_train = np.array(data["y_train"])
    X_test = np.array(data["X_test"], dtype=np.float32)
    y_test = np.array(data["y_test"])

    return (X_train, y_train, X_test, y_test)