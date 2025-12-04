import numpy as np
from rich.progress import *
import os

class SoftmaxRegression:
    def __init__(self, num_features: int, num_classes: int):
        """
        Initialize the Softmax Regression model.

        Args:
            num_features (int): The number of features in the input data.
            num_classes (int): The number of target classes.
            learning_rate (float): The step size for gradient descent optimization. Defaults to 0.01.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights: np.ndarray = None
        self.best_weights: np.ndarray = None
        self._min_loss = None
        self.loss_history = []
        self.acc_history = []

    def _initialize_weights(self):
        """
        Initialize weights to zeros.
        
        The shape is (num_features + 1, num_classes) to accommodate the bias term.
        """
        self.weights = np.zeros((self.num_features + 1, self.num_classes))
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

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, learning_rate=0.0001, epochs=100):
        """
        Train the model using Gradient Descent.

        Args:
            X (np.ndarray): Training features of shape (n, m).
            y (np.ndarray): Training labels of shape (n,).
            verbose (bool): Whether to display the progress bar. Defaults to True.
            epochs (int): Number of training iterations. Defaults to 100.
        """
        self._initialize_weights()
        self._min_loss = 2e9

		# Integrate bias to X to remove bias during calculating
        X_biased = self.get_X_biased(X)

        N = X_biased.shape[0]
        y_target = self._one_hot_encode(y)

        for i in range(epochs):
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}"),
                disable=(not verbose)
            ) as progress:
                task = progress.add_task(
                    f"Epoch {i+1}/{epochs}", 
                    total=1, 
                    metrics="loss: --  acc: --"
                )
                
                # Compute the score vector
                z = X_biased @ self.weights
                # Convert the score vector to distribution vector
                y_pred = self._softmax(z)

                # Compute the derivative of z
                dz = y_pred - y_target
                
                # Compute the gradient
                dw = X_biased.T @ dz / N

                # Update weights by gradient descent
                self.weights -= learning_rate * dw
                
                # Evaluate loss and accuracy during training
                loss = self._cross_entropy_loss(y_target, y_pred)
                acc = self._accuracy(X, y, use_best=False)
                self.loss_history.append(loss)
                self.acc_history.append(acc)

                # Save best weights
                if loss < self._min_loss:
                    self._min_loss = loss
                    self.best_weights = self.weights.copy()

                progress.update(task, advance=1, metrics=f"loss: {loss:.4f}  acc: {acc:.4f}")

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
    
    def save_best_model(self, model_path: str) -> bool:
        """
        Save (best_weights) into file .npy
        """
        try:
            if self.best_weights is None:
                print("Error: No best weights to save.")
                return False
            np.save(model_path, self.best_weights)
            
            print(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            # Bắt mọi lỗi (I/O, permission, v.v.)
            print(f"Error saving model: {e}")
            return False

    def load_weight(self, weight_path: str) -> bool:
        """
        Đọc trọng số từ file .npy và gán vào model hiện tại
        """
        try:
            # Xử lý trường hợp người dùng quên điền đuôi .npy
            if not os.path.exists(weight_path) and os.path.exists(weight_path + ".npy"):
                weight_path += ".npy"
            
            # Load file
            weights = np.load(weight_path)
            
            # Gán trọng số vừa load vào biến trọng số chính của class (self.W)
            self.W = weights
            
            # (Tùy chọn) Cập nhật luôn best_weights để đồng bộ
            self.best_weights = weights
            
            print(f"Weights loaded successfully from {weight_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found at {weight_path}")
            return False
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False    