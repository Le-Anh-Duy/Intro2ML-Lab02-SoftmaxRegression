import numpy as np
import matplotlib.pyplot as plt
from models.softmax_regression import SoftmaxRegression

class BlockSoftmax(SoftmaxRegression):
    def __init__(self, num_classes, grid_size=(7, 7), **kwargs):
        """
        grid_size: Kích thước lưới chia (mặc định 7x7 ô)
        """
        # Tính số lượng features mới
        self.grid_size = grid_size
        self.num_block_features = grid_size[0] * grid_size[1]
        
        super().__init__(num_features=self.num_block_features, num_classes=num_classes, **kwargs)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Biến đổi batch ảnh (N, 784) thành (N, grid_h * grid_w)
        Sử dụng kỹ thuật reshape của NumPy để tính trung bình khối cực nhanh.
        """
        # 1. Reshape về ảnh gốc (N, 28, 28)
        if X.ndim == 2:
            X = X.reshape(-1, 28, 28)
            
        N, H, W = X.shape
        grid_h, grid_w = self.grid_size
        
        # Kiểm tra xem ảnh có chia hết cho lưới không
        # Ví dụ: 28 chia hết cho 7 (mỗi ô 4 pixel), chia hết cho 4 (mỗi ô 7 pixel)
        assert H % grid_h == 0 and W % grid_w == 0, \
            f"Ảnh {H}x{W} không chia hết cho lưới {grid_h}x{grid_w}"
            
        block_h = H // grid_h
        block_w = W // grid_w
        
        # 2. Reshape thần thánh để tách block
        # Từ (N, 28, 28) -> (N, 7, 4, 7, 4) 
        # (N, số ô dọc, chiều cao ô, số ô ngang, chiều rộng ô)
        X_reshaped = X.reshape(N, grid_h, block_h, grid_w, block_w)
        
        # 3. Tính trung bình (mean) trên trục chiều cao ô (axis 2) và chiều rộng ô (axis 4)
        X_blocked = X_reshaped.mean(axis=(2, 4))
        
        # Lúc này X_blocked có shape (N, 7, 7)
        
        # 4. Duỗi phẳng thành vector feature (N, 49)
        return X_blocked.reshape(N, -1)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, learning_rate=0.1, epochs=100):
        print(f"Applying Block Averaging {self.grid_size}...")
        X_block = self._transform(X)
        print(f"Block Feature shape: {X_block.shape}") # Ví dụ: (60000, 49)
        
        super().fit(X_block, y, verbose=verbose, learning_rate=learning_rate, epochs=epochs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        X_block = self._transform(X)
        return super().predict(X_block, use_best=use_best)