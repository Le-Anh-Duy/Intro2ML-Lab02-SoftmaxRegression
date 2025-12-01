import numpy as np
import cv2
from base import SoftmaxRegression

class EdgeSoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, learning_rate=0.5, **kwargs):
        super().__init__(num_features, num_classes, learning_rate=learning_rate, **kwargs)

    def _extract_sobel_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess data: Apply Sobel Edge Detection -> Flatten -> Normalize.

        Args:
            X (np.ndarray): Input images (N, 784) or (N, 28, 28).

        Returns:
            np.ndarray: Normalized edge features (N, 784).
        """
        # 1. Reshape về dạng ảnh (N, 28, 28) để xử lý ảnh
        if X.ndim == 2:
            N = X.shape[0]
            # Giả sử ảnh vuông 28x28
            side = int(np.sqrt(X.shape[1])) 
            images = X.reshape(N, side, side)
        else:
            images = X
            N = images.shape[0]

        edge_features = []

        # 2. Loop qua từng ảnh để tính Sobel
        for i in range(N):
            img = images[i]
            
            # Lưu ý: OpenCV Sobel cần type phù hợp, float là tốt nhất để tính toán
            # Tính đạo hàm theo phương X (dọc)
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            # Tính đạo hàm theo phương Y (ngang)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Tính độ lớn vector gradient (Magnitude)
            # Đây chính là "độ đậm" của cạnh
            magnitude = np.sqrt(gx**2 + gy**2)
            
            edge_features.append(magnitude.flatten())

        X_edge = np.array(edge_features)

        # 3. Normalize (Standardization giống PixelSoftmax)
        # Giúp Gradient Descent hội tụ nhanh hơn
        epsilon = 1e-8
        mean = np.mean(X_edge, axis=0)
        std = np.std(X_edge, axis=0)
        
        return (X_edge - mean) / (std + epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, epochs=100):
        """
        Train the Edge-based model.
        """
        if verbose:
            print("Feature Extraction: Computing Sobel Edges...")
            
        X_proc = self._extract_sobel_normalize(X)
        
        # Gọi hàm fit của class cha (SoftmaxRegression)
        super().fit(X_proc, y, verbose=verbose, epochs=epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using edge features.
        """
        X_proc = self._extract_sobel_normalize(X)
        return super().predict(X_proc)