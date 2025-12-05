import numpy as np
from models.softmax_regression import SoftmaxRegression
import os
class PCASoftmax(SoftmaxRegression):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__(num_features, num_classes, *args, **kwargs)

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
        
        return np.asarray(X / 255.0, dtype = np.float32)

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

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
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

        super().fit(X_train, y, *args, **kwargs)

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
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize image after PCA transformation and reconstruction.
        Shows information preserved in the reduced n_components dimensions.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,).
            
        Returns:
            np.ndarray: Reconstructed image (28, 28) after PCA round-trip.
        """
        if self.components is None:
            raise ValueError("PCA has not been fitted yet")
        
        # Reshape if needed
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(1, -1)
        elif sample_image.ndim == 2 and sample_image.shape[0] != 1:
            sample_image = sample_image.reshape(1, -1)
        
        # Apply normalization (same as in fit/predict)
        X_proc = self._flatten_normalize(sample_image)  # Shape: (1, 784)
        
        # Center the data using PCA mean
        X_centered = X_proc - self.pca_mean  # Shape: (1, 784)
        
        # Project to PCA space (reduce to n_components dimensions)
        # components shape: (784, n_components)
        X_pca = X_centered @ self.components  # Shape: (1, n_components)
        
        # Reconstruct back to 784-dimensional pixel space
        # This shows what information is preserved after dimensionality reduction
        X_reconstructed = X_pca @ self.components.T + self.pca_mean  # Shape: (1, 784)
        
        # Clip to valid range [0, 1] and reshape to 28x28
        X_reconstructed = np.clip(X_reconstructed, 0, 1)
        
        return X_reconstructed.reshape(28, 28)
    
    def save_best_model(self, model_path: str) -> bool:
        """
        Save PCASoftmax model including PCA parameters.
        Saves: best_weights, components, pca_mean, scaler_mean, scaler_std
        """
        try:
            if self.best_weights is None:
                print("Error: No best weights to save.")
                return False
            
            # Handle .npz extension
            if not model_path.endswith('.npz'):
                model_path = model_path.replace('.npy', '.npz')
            
            # Save all parameters
            np.savez(
                model_path,
                best_weights=self.best_weights,
                components=self.components,
                pca_mean=self.pca_mean,
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
                n_components=np.array([self.n_components]),  # Save as array for npz
                num_classes=self.num_classes,
                num_features=self.num_features
            )
            
            print(f"PCASoftmax model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_weight(self, weight_path: str) -> bool:
        """
        Load PCASoftmax model including PCA parameters.
        Loads: best_weights, components, pca_mean, scaler_mean, scaler_std
        """
        try:
            # Handle file extension
            if not os.path.exists(weight_path):
                if os.path.exists(weight_path + ".npz"):
                    weight_path += ".npz"
                elif os.path.exists(weight_path.replace('.npy', '.npz')):
                    weight_path = weight_path.replace('.npy', '.npz')
            
            # Load file
            data = np.load(weight_path, allow_pickle=True)
            
            # Load all parameters
            self.best_weights = data['best_weights']
            self.weights = self.best_weights.copy()
            self.components = data['components']
            self.pca_mean = data['pca_mean']
            self.scaler_mean = data['scaler_mean']
            self.scaler_std = data['scaler_std']
            
            if 'n_components' in data:
                self.n_components = int(data['n_components'][0])
            if 'num_classes' in data:
                self.num_classes = int(data['num_classes'])
            if 'num_features' in data:
                self.num_features = int(data['num_features'])
            
            print(f"PCASoftmax model loaded successfully from {weight_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found at {weight_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False