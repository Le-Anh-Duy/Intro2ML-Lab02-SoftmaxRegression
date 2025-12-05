import numpy as np
from models.softmax_regression import SoftmaxRegression

class PixelSoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)

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
        
        return np.asarray(X / 255.0, dtype=np.float32)
    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, learning_rate=0.0001, epochs=100):
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
        
        super().fit(X_proc, y, verbose=verbose, learning_rate=learning_rate, epochs=epochs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        """
        Predict class labels for raw input images.

        Args:
            X (np.ndarray): Raw input images.

        Returns:
            np.ndarray: Predicted class indices.
        """
        X_proc = self._flatten_normalize(X)
        return super().predict(X_proc, use_best=use_best)
    
    def predict_proba(self, X: np.ndarray, use_best=True) -> np.ndarray:
        """
        Predict class probabilities for raw input images.

        Args:
            X (np.ndarray): Raw input images.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        X_proc = self._flatten_normalize(X)
        return super().predict_proba(X_proc, use_best=use_best)
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize normalized pixel values.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,).
            
        Returns:
            np.ndarray: Normalized image (28, 28).
        """
        # Reshape if needed
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(1, -1)
        elif sample_image.ndim == 2 and sample_image.shape[0] != 1:
            sample_image = sample_image.reshape(1, -1)
        
        # Apply preprocessing (flatten and normalize)
        processed = self._flatten_normalize(sample_image)
        
        # Reshape back to 28x28
        return processed.reshape(28, 28)
    
    def save_best_model(self, model_path: str) -> bool:
        """Save PixelSoftmax model (only needs weights)."""
        return super().save_best_model(model_path)
    
    def load_weight(self, weight_path: str) -> bool:
        """Load PixelSoftmax model (only needs weights)."""
        return super().load_weight(weight_path)