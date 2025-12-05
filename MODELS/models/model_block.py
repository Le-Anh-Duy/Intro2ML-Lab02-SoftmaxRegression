import numpy as np
import matplotlib.pyplot as plt
from models.softmax_regression import SoftmaxRegression
import os
class BlockSoftmax(SoftmaxRegression):
    def __init__(self, num_classes, grid_size=(7, 7), *args, **kwargs):
        """
        grid_size: Kích thước lưới chia (mặc định 7x7 ô)
        """
        # Tính số lượng features mới
        self.grid_size = grid_size
        self.num_block_features = grid_size[0] * grid_size[1]
        
        super().__init__(num_features=self.num_block_features, num_classes=num_classes, *args, **kwargs)

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
        return np.asarray(X_blocked.reshape(N, -1), dtype = np.float32) / 255.0 

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        print(f"Applying Block Averaging {self.grid_size}...")
        X_block = self._transform(X)
        print(f"Block Feature shape: {X_block.shape}") # Ví dụ: (60000, 49)
        
        super().fit(X_block, y, *args, **kwargs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        X_block = self._transform(X)
        return super().predict(X_block, use_best=use_best)
    
    def predict_proba(self, X: np.ndarray, use_best=True):
        X_block = self._transform(X)
        return super().predict_proba(X_block, use_best=use_best)
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize block-averaged image.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,).
            
        Returns:
            np.ndarray: Block-averaged image upsampled back to (28, 28).
        """
        # Reshape if needed
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(1, 28, 28)
        elif sample_image.ndim == 2:
            sample_image = sample_image.reshape(1, 28, 28)
        
        # Apply block averaging preprocessing
        block_features = self._transform(sample_image).flatten()
        
        # Reshape to grid
        block_grid = block_features.reshape(self.grid_size[0], self.grid_size[1])
        
        # Upsample back to 28x28 using repeat
        block_h = 28 // self.grid_size[0]
        block_w = 28 // self.grid_size[1]
        
        vis = np.repeat(np.repeat(block_grid, block_h, axis=0), block_w, axis=1)
        
        return vis[:28, :28]
    
    def save_best_model(self, model_path: str) -> bool:
        """
        Save BlockSoftmax model including grid_size parameter.
        Saves: best_weights, grid_size
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
                grid_size=np.array(self.grid_size),
                num_classes=self.num_classes,
                num_features=self.num_features
            )
            
            print(f"BlockSoftmax model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_weight(self, weight_path: str) -> bool:
        """
        Load BlockSoftmax model including grid_size parameter.
        Loads: best_weights, grid_size
        """
        try:
            # Handle file extension
            import os
            if not os.path.exists(weight_path):
                if os.path.exists(weight_path + ".npz"):
                    weight_path += ".npz"
                elif os.path.exists(weight_path.replace('.npy', '.npz')):
                    weight_path = weight_path.replace('.npy', '.npz')
            
            # Load file
            data = np.load(weight_path)
            
            # Load all parameters
            self.best_weights = data['best_weights']
            self.weights = self.best_weights.copy()

            if 'grid_size' in data:
                self.grid_size = tuple(data['grid_size'])
            if 'num_classes' in data:
                self.num_classes = int(data['num_classes'])
            if 'num_features' in data:
                self.num_features = int(data['num_features'])
            
            print(f"BlockSoftmax model loaded successfully from {weight_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found at {weight_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False