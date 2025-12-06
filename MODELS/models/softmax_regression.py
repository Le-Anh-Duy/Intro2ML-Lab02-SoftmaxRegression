import numpy as np
from rich.progress import *
import os
from models.optimizer import *

class SoftmaxRegression:
    def __init__(self, num_features: int, num_classes: int, optimizer = GD()):
        """
        Initialize the Softmax Regression model.

        Args:
            num_features (int): The number of features in the input data.
            num_classes (int): The number of target classes.
            learning_rate (float): The step size for gradient descent optimization. Defaults to 0.01.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.weights: np.ndarray = None
        self.best_weights: np.ndarray = None
        self._min_loss = None
        self.loss_history = []
        self.acc_history = []

    def _initialize_weights(self, resume = False):
        """
        Initialize weights to zeros.
        
        The shape is (num_features + 1, num_classes) to accommodate the bias term.
        """
        if self.weights is not None and resume:
            self.best_weights = self.weights.copy()
            return
        self.weights = np.zeros((self.num_features + 1, self.num_classes), dtype=np.float32)
        self.best_weights = self.weights.copy()

    def get_X_biased(self, X: np.ndarray):
        """
        Add a column of ones to the input data to handle the bias term.

        Args:
            X (np.ndarray): Input feature matrix of shape (n, m).

        Returns:
            np.ndarray: Feature matrix with bias term, shape (n, m + 1).
        """
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax function numerically stable.

        Args:
            z (np.ndarray): Linear combination of weights and input (logits).

        Returns:
            np.ndarray: Probability distribution over classes.
        """
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Convert integer class labels to one-hot encoded vectors.

        Args:
            y (np.ndarray): Array of integer class labels.

        Returns:
            np.ndarray: One-hot encoded matrix of shape (n, num_classes).
        """
        one_hot = np.zeros((y.size, self.num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def _cross_entropy_loss(self, y_target: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Cross-Entropy Loss.

        Args:
            y_target (np.ndarray): One-hot encoded ground truth labels.
            y_pred (np.ndarray): Predicted probabilities from softmax.

        Returns:
            float: The average cross-entropy loss.
        """
        eps = 1e-9
        m = y_target.shape[0]
        return -np.sum(y_target * np.log(y_pred.clip(eps, 1 - eps))) / m
    
    def _accuracy(self, X: np.ndarray, y: np.ndarray, use_best=True) -> float:
        """
        Compute the accuracy of the model on the given data.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): True class labels.
            use_best (bool): Whether to use the best weights. Defaults to True.

        Returns:
            float: The accuracy score (between 0 and 1).
        """
        y_pred = self.predict(X, use_best=use_best)
        return np.sum(y_pred == y) / y.size

    def _generate_batches(self, X, y, batch_size, shuffle=True):
        """
        Generator trả về từng batch dữ liệu đã được shuffle.
        Tương đương với dataset.shuffle().batch() trong TF.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples) # Tạo mảng chỉ mục [0, 1, 2, ..., N-1]

        if shuffle:
            np.random.shuffle(indices) # Xáo trộn chỉ mục

        # Duyệt qua các chỉ mục theo từng bước nhảy (batch_size)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Lấy ra mảng chỉ mục cho batch hiện tại
            batch_indices = indices[start_idx:end_idx]
            
            # Trả về dữ liệu tương ứng với chỉ mục đó
            yield X[batch_indices], y[batch_indices]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, verbose=True, batch_size=64, learning_rate=0.01, epochs=100, resume = False):
        """
        Train the model using Gradient Descent.
        """
        self._initialize_weights(resume)
        self._min_loss = 2e9

        # Integrate bias to X to remove bias during calculating
        X_biased = self.get_X_biased(X)
        y_target = self._one_hot_encode(y)
        
        # Tính số lượng batch (steps) cho mỗi epoch
        total_steps = (X_biased.shape[0] + batch_size - 1) // batch_size

        # 1. Khởi tạo Progress Context ra ngoài vòng lặp epochs
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            disable=(not verbose)
        ) as progress:
            
            # Tạo task lần đầu tiên
            task = progress.add_task(f"Epoch 0/{epochs}", total=total_steps, metrics="loss: --")

            for i in range(epochs):
                # 2. Reset lại task ở đầu mỗi epoch
                # Đưa 'completed' về 0, cập nhật tiêu đề Epoch mới
                progress.reset(task, description=f"Epoch {i+1}/{epochs}", total=total_steps, metrics="loss: --")

                for X_batch, y_batch in self._generate_batches(X_biased, y_target, batch_size):
                    N = X_batch.shape[0]
                    
                    # --- Training Logic ---
                    z = X_batch @ self.weights
                    y_pred = self._softmax(z)
                    dz = y_pred - y_batch
                    dw = X_batch.T @ dz / N
                    self.weights = self.optimizer.apply(self.weights, dw, learning_rate)
                    loss = self._cross_entropy_loss(y_batch, y_pred)

                    if loss < self._min_loss:
                        self._min_loss = loss
                        self.best_weights = self.weights.copy()
                    # -----------------------

                    # Cập nhật tiến trình từng bước (advance=1)
                    progress.update(task, advance=1, metrics=f"loss: {loss:.4f}")
                
                # End of epoch logic
                self.loss_history.append(loss)

                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_acc = np.sum(val_pred == y_val) / y_val.size
                    
                    # LƯU Ý QUAN TRỌNG:
                    # Dùng progress.console.print thay vì print thường
                    # để in log lên trên thanh progress bar mà không làm vỡ giao diện
                    progress.console.print(f"Epoch {i+1}/{epochs} - Validation Accuracy: {val_acc*100:.2f}%")

    def predict(self, X: np.ndarray, use_best=True) -> int:
        """
        Predict class labels for the given input data.

        Args:
            X (np.ndarray): Input feature matrix.
            use_best (bool): Whether to use the best weights. Defaults to True.

        Returns:
            np.ndarray: Predicted class indices.
        """
        z = np.dot(self.get_X_biased(X), self.best_weights if use_best else self.weights)
        # print(z)
        y_pred = self._softmax(z)
        return y_pred.argmax(axis=1)

    def predict_proba(self, X: np.ndarray, use_best=True) -> np.ndarray:
        """
        Predict class probabilities for the given input data.

        Args:
            X (np.ndarray): Input feature matrix.
            use_best (bool): Whether to use the best weights. Defaults to True.

        Returns:
            np.ndarray: Matrix of class probabilities.
        """
        z = np.dot(self.get_X_biased(X), self.best_weights if use_best else self.weights)
        return self._softmax(z)
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize preprocessed features for a sample image.
        Base class simply normalizes and reshapes the image.
        Subclasses should override to show their specific preprocessing.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,) or (1, 784).
            
        Returns:
            np.ndarray: Preprocessed features as (28, 28) image for visualization.
        """
        # Reshape if flattened
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(28, 28)
        elif sample_image.ndim == 3:
            sample_image = sample_image.reshape(28, 28)
        
        # Normalize to [0, 1]
        img = sample_image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def save_best_model(self, model_path: str) -> bool:
        """
        Save model to .npz file (supports multiple parameters).
        Base SoftmaxRegression only saves best_weights.
        Subclasses should override to save additional parameters.
        """
        try:
            if self.best_weights is None:
                print("Error: No best weights to save.")
                return False
            
            # Handle .npz extension
            if not model_path.endswith('.npz'):
                model_path = model_path.replace('.npy', '.npz')
            
            # Save to npz with num_classes and num_features
            np.savez(
                model_path, 
                best_weights=self.best_weights,
                num_classes=self.num_classes,
                num_features=self.num_features
            )
            
            print(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_weight(self, weight_path: str) -> bool:
        """
        Load model from .npz file.
        Base SoftmaxRegression only loads best_weights.
        Subclasses should override to load additional parameters.
        """
        try:
            # Handle file extension
            if not os.path.exists(weight_path):
                if os.path.exists(weight_path + ".npz"):
                    weight_path += ".npz"
                elif os.path.exists(weight_path.replace('.npy', '.npz')):
                    weight_path = weight_path.replace('.npy', '.npz')
            
            # Load file
            data = np.load(weight_path)
            
            # Load best_weights
            if 'best_weights' in data:
                self.best_weights = data['best_weights']
                self.weights = self.best_weights.copy()
                
                # Load num_classes and num_features if available
                if 'num_classes' in data:
                    self.num_classes = int(data['num_classes'])
                if 'num_features' in data:
                    self.num_features = int(data['num_features'])
            else:
                # Fallback for old .npy files
                self.best_weights = data['arr_0'] if 'arr_0' in data else data
                self.weights = self.best_weights.copy()
            
            print(f"Weights loaded successfully from {weight_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found at {weight_path}")
            return False
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False    
