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

    def _compute_hog_single(self, img):
        """
        Tính HOG cho 1 ảnh (28x28)
        """
        # 1. Tính Gradient theo X và Y dùng Sobel
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        
        # 2. Chuyển sang Magnitude (độ lớn) và Angle (góc)
        # angle trả về từ 0 đến 360 độ
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Chỉ quan tâm hướng vô hướng (0-180 độ), ví dụ nét lên hay nét xuống coi như nhau
        angle = angle % 180 
        
        # 3. Chia ảnh thành các cell và tính histogram
        h, w = img.shape
        cell_h = h // self.cell_grid[0]
        cell_w = w // self.cell_grid[1]
        
        hog_vector = []
        
        for i in range(self.cell_grid[0]):
            for j in range(self.cell_grid[1]):
                # Cắt vùng cell tương ứng
                cell_mag = mag[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                cell_ang = angle[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                
                # Tính histogram cho cell này
                # range=(0, 180) nghĩa là chia góc từ 0-180 thành 'bins' phần
                # weights=cell_mag nghĩa là pixel nào nét đậm thì phiếu bầu giá trị cao hơn
                hist, _ = np.histogram(cell_ang, bins=self.bins, range=(0, 180), weights=cell_mag)
                
                hog_vector.extend(hist)
                
        return np.array(hog_vector)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi cả batch ảnh X (N, 784) sang HOG features (N, features)
        """
        # Reshape lại thành ảnh (N, 28, 28) nếu đang bị flatten
        if X.ndim == 2:
            X = X.reshape(-1, 28, 28)
            
        hog_features = []
        # Lưu ý: Vòng lặp này hơi chậm, nhưng dễ hiểu và code thủ công.
        # Với N=60000 có thể mất khoảng 30s-1p để pre-process.
        for img in X:
            # Ảnh MNIST gốc là float hoặc uint8, cần đảm bảo format đúng cho cv2
            img_float = img.astype(np.float32)
            hog_features.append(self._compute_hog_single(img_float))
            
        # Chuẩn hóa L2 cho toàn bộ vector feature (giúp chống lại thay đổi độ sáng)
        hog_features = np.array(hog_features)
        norm = np.linalg.norm(hog_features, axis=1, keepdims=True)
        return hog_features / (norm + 1e-6)

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        print("Extracting HOG features...")
        X_hog = self._transform(X)
        print(f"HOG Feature shape: {X_hog.shape}") # Ví dụ: (60000, 144)
        
        super().fit(X_hog, y, *args, **kwargs)

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