import numpy as np
import cv2
from softmax_regression import SoftmaxRegression

class EdgeSoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)

    def _extract_sobel_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess data: Apply Sobel Edge Detection -> Flatten -> Normalize.

        Args:
            X (np.ndarray): Input images (N, 784) or (N, 28, 28).

        Returns:
            np.ndarray: Normalized edge features (N, 784).
        """
        # 1. Reshape
        if X.ndim == 2:
            N = X.shape[0]
            side = int(np.sqrt(X.shape[1])) 
            images = X.reshape(N, side, side)
        else:
            images = X
            N = images.shape[0]

        edge_features = []

        # 2. Sobel Edge Detection
        for i in range(N):
            img = images[i]
            
            # Calculate gradient
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            magnitude = np.sqrt(gx**2 + gy**2)
            
            edge_features.append(magnitude.flatten())

        X_edge = np.array(edge_features)

        # 3. Normalize
        epsilon = 1e-8
        mean = np.mean(X_edge, axis=0)
        std = np.std(X_edge, axis=0)
        
        return (X_edge - mean) / (std + epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, learning_rate=0.0001, epochs=100):
        """
        Train the Edge-based model.
        """
        if verbose:
            print("Feature Extraction: Computing Sobel Edges...")
            
        X_proc = self._extract_sobel_normalize(X)
        
        super().fit(X_proc, y, verbose=verbose, learning_rate=learning_rate, epochs=epochs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        """
        Predict class labels using edge features.
        """
        X_proc = self._extract_sobel_normalize(X)
        return super().predict(X_proc, use_best=use_best)
    
    def predict_proba(self, X: np.ndarray, use_best=True) -> np.ndarray:
        """
        Predict class probabilities using edge features.
        """
        X_proc = self._extract_sobel_normalize(X)
        return super().predict_proba(X_proc, use_best=use_best)