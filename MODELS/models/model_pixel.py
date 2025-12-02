import numpy as np
from softmax_regression import SoftmaxRegression

class PixelSoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, learning_rate=0.5, **kwargs):
        super().__init__(num_features, num_classes, learning_rate=learning_rate, **kwargs)
    def _flatten_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess data: Flatten image and normalize pixel values.

        Args:
            X (np.ndarray): Input images of shape (N, 28, 28) or (N, 784).

        Returns:
            np.ndarray: Normalized flattened features (N, 784).
        """
        # Flatten
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Normalize
        epsilon = 1e-8
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        return (X - mean) / (std + epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, epochs=100):
        """
        Train the Pixel-based model.

        Overridden to preprocess features (flatten & normalize) before training.

        Args:
            X (np.ndarray): Raw input images.
            y (np.ndarray): Training labels.
            verbose (bool): Display progress bar.
            epochs (int): Number of training iterations.
        """
        X_proc = self._flatten_normalize(X)
        
        super().fit(X_proc, y, verbose=verbose, epochs=epochs)

    def predict(self, X: np.ndarray) -> int:
        """
        Predict class labels for raw input images.

        Args:
            X (np.ndarray): Raw input images.

        Returns:
            np.ndarray: Predicted class indices.
        """
        X_proc = self._flatten_normalize(X)
        return super().predict(X_proc)