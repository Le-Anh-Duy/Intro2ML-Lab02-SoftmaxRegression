import numpy as np

class SoftmaxRegression:
    """Base Softmax Regression classifier using NumPy."""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train the model. Override this in child classes."""
        pass
    
    def predict(self, X):
        """Predict class labels. Override this in child classes."""
        pass
    
    def preprocess_features(self, X):
        """Preprocess features. Override this in child classes."""
        pass
