import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy score comparing predicted and true labels.

    Args:
        y_true (np.ndarray): Ground truth class labels (1D integer array).
        y_pred (np.ndarray): Predicted class labels (1D integer array).

    Returns:
        float: The accuracy score (between 0.0 and 1.0).
    """
    return np.sum(y_pred == y_true) / y_true.size

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute the confusion matrix.
    
    Args:
        y_true (np.ndarray): Ground truth class labels (1D integer array).
        y_pred (np.ndarray): Predicted class labels (1D integer array).
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: The confusion matrix of shape (num_classes, num_classes).
    """
    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Iterate through each pair of (true label, predicted label) and accumulate counts
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
        
    return confusion_matrix