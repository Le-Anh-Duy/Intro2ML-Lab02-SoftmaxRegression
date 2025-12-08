import numpy as np
import cv2
from models.softmax_regression import SoftmaxRegression
import os

class HOGSoftmax(SoftmaxRegression):
    def __init__(self, num_classes, bins=9, cell_grid=(4, 4), *args, **kwargs):
        """
        bins: Số lượng hướng gradient (ví dụ 9 hướng từ 0-180 độ)
        cell_grid: Chia ảnh thành lưới (4x4 cells)
        """
        # Tính toán số lượng features đầu vào
        # Features = (số cells ngang * số cells dọc) * số bins
        self.num_hog_features = cell_grid[0] * cell_grid[1] * bins
        
        super().__init__(num_features=self.num_hog_features, num_classes=num_classes, *args, **kwargs)
        
        self.bins = bins
        self.cell_grid = cell_grid

    def _compute_gradients(self, X_float: np.ndarray):
        """
        [Util] Tính Magnitude và Angle cho cả batch ảnh.
        Input: (N, H, W)
        Output: mags, angs (N, H, W)
        """
        N, H, W = X_float.shape
        mags = np.zeros((N, H, W), dtype=np.float32)
        angs = np.zeros((N, H, W), dtype=np.float32)

        for i in range(N):
            gx = cv2.Sobel(X_float[i], cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(X_float[i], cv2.CV_32F, 0, 1, ksize=1)
            m, a = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            mags[i] = m
            angs[i] = a % 180
            
        return mags, angs

    def _vectorized_binning(self, mags: np.ndarray, angs: np.ndarray):
        """
        [Util] Chia cell và tính Histogram bằng ma trận (Vectorization).
        Đây là phần phức tạp nhất được tách ra.
        """
        N, H, W = mags.shape
        gh, gw = self.cell_grid
        ch = H // gh
        cw = W // gw
        
        # 1. Cắt ảnh cho vừa khớp lưới (nếu kích thước không chia hết)
        mags = mags[:, :gh*ch, :gw*cw]
        angs = angs[:, :gh*ch, :gw*cw]
        
        # 2. Reshape & Transpose để gom pixel về các cell
        # (N, grid_h, cell_h, grid_w, cell_w) -> (N, grid_h, grid_w, cell_h, cell_w)
        mags_cells = mags.reshape(N, gh, ch, gw, cw).transpose(0, 1, 3, 2, 4)
        angs_cells = angs.reshape(N, gh, ch, gw, cw).transpose(0, 1, 3, 2, 4)
        
        # 3. Tính bin cho từng pixel
        bin_width = 180 / self.bins
        bin_indices = (angs_cells / bin_width).astype(int) % self.bins
        
        # 4. Dùng broadcasting để thay vòng lặp pixel
        hog_cells = np.zeros((N, gh, gw, self.bins), dtype=np.float32)
        
        for b in range(self.bins):
            # Mask cho bin hiện tại
            mask = (bin_indices == b)
            # Cộng dồn magnitude
            hog_cells[:, :, :, b] = np.sum(mags_cells * mask, axis=(3, 4))
            
        return hog_cells

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi batch ảnh X sang đặc trưng HOG.
        """
        # 1. Chuẩn bị dữ liệu
        if X.ndim == 2:
            X = X.reshape(-1, 28, 28)
        
        X_float = X.astype(np.float32)
        N = X.shape[0]
        
        # 2. Tính Gradient
        mags, angs = self._compute_gradients(X_float)

        # 3. Tính Histogram (Binning)
        hog_cells = self._vectorized_binning(mags, angs)
            
        # 4. Flatten và Normalize L2
        hog_features = hog_cells.reshape(N, -1)
        norm = np.linalg.norm(hog_features, axis=1, keepdims=True)
        
        return hog_features / (norm + 1e-6)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, *args, **kwargs):
        print("Extracting HOG features (Vectorized Refactored)...")
        # Transform tập Train
        X_hog = self._transform(X)
        print(f"HOG Feature shape: {X_hog.shape}")
        
        super().fit(X_hog, y, X_val=X_val, y_val=y_val, *args, **kwargs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        X_hog = self._transform(X)
        return super().predict(X_hog, use_best=use_best)
    
    def predict_proba(self, X: np.ndarray, use_best=True) -> int:
        X_hog = self._transform(X)
        return super().predict_proba(X_hog, use_best=use_best)
    
    def get_feature_visualization(self, sample_image: np.ndarray) -> np.ndarray:
        """
        Visualize HOG features as a heatmap showing gradient strength in each cell.
        
        Args:
            sample_image (np.ndarray): Input image (28, 28) or (784,).
            
        Returns:
            np.ndarray: HOG feature heatmap (28, 28).
        """
        # Reshape if needed
        if sample_image.ndim == 1:
            sample_image = sample_image.reshape(1, 28, 28)
        elif sample_image.ndim == 2:
            sample_image = sample_image.reshape(1, 28, 28)
        
        # Extract HOG features
        hog_features = self._transform(sample_image).flatten()
        
        # Reshape to (cell_grid[0], cell_grid[1], bins)
        hog_grid = hog_features.reshape(self.cell_grid[0], self.cell_grid[1], self.bins)
        
        # Sum across bins to get total gradient magnitude per cell
        cell_importance = np.sum(hog_grid, axis=2)
        
        # Normalize to [0, 1]
        if cell_importance.max() > 0:
            cell_importance = cell_importance / cell_importance.max()
        
        # Upsample to 28x28 for visualization
        cell_h = 28 // self.cell_grid[0]
        cell_w = 28 // self.cell_grid[1]
        
        vis = np.repeat(np.repeat(cell_importance, cell_h, axis=0), cell_w, axis=1)
        
        return vis[:28, :28]
    
    def save_best_model(self, model_path: str) -> bool:
        """
        Save HOGSoftmax model including HOG parameters.
        Saves: best_weights, bins, cell_grid
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
                bins=np.array([self.bins]),
                cell_grid=np.array(self.cell_grid),
                num_classes=self.num_classes,
                num_features=self.num_features
            )
            
            print(f"HOGSoftmax model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_weight(self, weight_path: str) -> bool:
        """
        Load HOGSoftmax model including HOG parameters.
        Loads: best_weights, bins, cell_grid
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
            
            # Load all parameters
            self.best_weights = data['best_weights']
            self.weights = self.best_weights.copy()
            
            if 'bins' in data:
                self.bins = int(data['bins'][0])
            if 'cell_grid' in data:
                self.cell_grid = tuple(data['cell_grid'])
            if 'num_classes' in data:
                self.num_classes = int(data['num_classes'])
            if 'num_features' in data:
                self.num_features = int(data['num_features'])
            
            print(f"HOGSoftmax model loaded successfully from {weight_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found at {weight_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False