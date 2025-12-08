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

def precision_score(y_true, y_pred, num_classes=10, average='macro'):
    """
    Compute Precision.
    TP / (TP + FP)
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    # TP: Main diagonal of confusion matrix
    tp = np.diag(cm)
    # FP: Column sum minus TP (false positives for each class)
    fp = np.sum(cm, axis=0) - tp
    
    # Compute Precision for each class (handle division by zero)
    denominator = tp + fp
    # np.divide with where: if denominator != 0 then divide, else return 0
    per_class_precision = np.divide(tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator!=0)
    
    if average == 'macro':
        return np.mean(per_class_precision)
    return per_class_precision

def recall_score(y_true, y_pred, num_classes=10, average='macro'):
    """
    Compute Recall.
    TP / (TP + FN)
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    # TP: Main diagonal of confusion matrix
    tp = np.diag(cm)
    # FN: Row sum minus TP (false negatives for each class)
    fn = np.sum(cm, axis=1) - tp
    
    # Compute Recall for each class
    denominator = tp + fn
    per_class_recall = np.divide(tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator!=0)
    
    if average == 'macro':
        return np.mean(per_class_recall)
    return per_class_recall

def f1_score(y_true, y_pred, num_classes=10, average='macro'):
    """
    Compute F1-Score.
    2 * (P * R) / (P + R)
    """
    # Get precision and recall as arrays (average=None)
    p = precision_score(y_true, y_pred, num_classes, average=None)
    r = recall_score(y_true, y_pred, num_classes, average=None)
    
    # Compute F1 for each class
    denominator = p + r
    per_class_f1 = np.divide(2 * (p * r), denominator, out=np.zeros_like(p, dtype=float), where=denominator!=0)
    
    if average == 'macro':
        return np.mean(per_class_f1)
    return per_class_f1

def print_class_report(y_true, y_pred, model_name):
    classes = range(10)

    p = precision_score(y_true, y_pred, num_classes=10, average=None)
    r = recall_score(y_true, y_pred, num_classes=10, average=None)
    f1 = f1_score(y_true, y_pred, num_classes=10, average=None)
    
    print(f"\n{'='*20} DETAILED REPORT: {model_name} {'='*20}")
    print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 46)
    for i in classes:
        print(f"{i:<6} | {p[i]:.4f}     | {r[i]:.4f}     | {f1[i]:.4f}")
    print("-" * 46)