import numpy as np
import cv2
from models.softmax_regression import SoftmaxRegression

class EdgeSoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__(num_features, num_classes, *args, **kwargs)

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
        
        return np.asarray(X_edge / 255.0)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, *args, **kwargs):
        """
        Train the Edge-based model.
        """
        print("Feature Extraction: Computing Sobel Edges...")
            
        X_proc = self._extract_sobel_normalize(X)
        X_val_proc = self._extract_sobel_normalize(X_val) if X_val is not None else None
        
        super().fit(X_proc, y, X_val=X_val_proc, y_val=y_val, *args, **kwargs)

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
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize Sobel edge detection result.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,).
            
        Returns:
            np.ndarray: Edge magnitude image (28, 28) normalized to [0, 1].
        """
        # Reshape if needed
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(1, -1)
        elif sample_image.ndim == 2 and sample_image.shape[0] != 1:
            sample_image = sample_image.reshape(1, -1)
        
        # Apply Sobel edge detection preprocessing
        processed = self._extract_sobel_normalize(sample_image)
        
        # Reshape back to 28x28
        return processed.reshape(28, 28)
    
    def save_best_model(self, model_path: str) -> bool:
        """Save EdgeSoftmax model (only needs weights)."""
        return super().save_best_model(model_path)
    
    def load_weight(self, weight_path: str) -> bool:
        """Load EdgeSoftmax model (only needs weights)."""
        return super().load_weight(weight_path)