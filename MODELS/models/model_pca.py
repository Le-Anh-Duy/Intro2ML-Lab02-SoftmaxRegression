import numpy as np
from models.softmax_regression import SoftmaxRegression

class PCASoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, **kwargs):
        super().__init__(num_features, num_classes, **kwargs)

        self.n_components = num_features
        self.components = None # Đây là ma trận U_k (các hướng chính)
        self.pca_mean = None       # Đây là vector trung bình mu

        self.scaler_mean = None
        self.scaler_std = None
        
    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.pca_mean
        return np.dot(X_centered, self.components)
    
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
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        
        return (X - self.scaler_mean) / (self.scaler_std + epsilon)

    def _preprocess(self, X: np.ndarray):
        """
            Preprocess for PCA reduction
            
            :param self: Description
            :param X: Description
            :type X: np.ndarray
        """
        X_proc = self._flatten_normalize(X)
        X_processed = self._transform(X_proc)

        return X_processed

    def _PCA_fit(self, X: np.ndarray):
        """
        Tính toán các trục chính từ dữ liệu X (thường là tập Train)
        X shape: (N_samples, 784)
        """
        # 1. Tính mean và center dữ liệu
        self.pca_mean = np.mean(X, axis=0)
        X_centered = X - self.pca_mean

        # 2. Tính ma trận hiệp phương sai
        # rowvar=False nghĩa là mỗi cột là một feature, mỗi dòng là một sample
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Tính Eigenvalues và Eigenvectors
        # eigh dùng cho ma trận đối xứng (như cov_matrix), nhanh hơn eig thường
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sắp xếp giảm dần (Lưu ý: eigh thường trả về tăng dần nên cần đảo ngược)
        sorted_index = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_index]
        
        # 5. Lấy K vector đầu tiên
        self.components = sorted_eigenvectors[:, :self.n_components]

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
        self._PCA_fit(X_proc)
        X_train = self._transform(X_proc)
        print(X_train.shape)

        super().fit(X_train, y, verbose=verbose, learning_rate=learning_rate, epochs=epochs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        """
        Predict class labels for raw input images.

        Args:
            X (np.ndarray): Raw input images.

        Returns:
            np.ndarray: Predicted class indices.
        """
        X_proc = self._flatten_normalize(X)
        X_pred = self._transform(X_proc)
        return super().predict(X_pred, use_best=use_best)
    
    def predict_proba(self, X: np.ndarray, use_best=True) -> np.ndarray:
        """
        Predict class probabilities for raw input images.

        Args:
            X (np.ndarray): Raw input images.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        X_proc = self._flatten_normalize(X)
        X_pred = self._transform(X_proc)
        return super().predict_proba(X_pred, use_best=use_best)