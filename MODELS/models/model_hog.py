import numpy as np
import cv2
from models.softmax_regression import SoftmaxRegression

class HOGSoftmax(SoftmaxRegression):
    def __init__(self, num_classes, bins=9, cell_grid=(4, 4), **kwargs):
        """
        bins: Số lượng hướng gradient (ví dụ 9 hướng từ 0-180 độ)
        cell_grid: Chia ảnh thành lưới (4x4 cells)
        """
        # Tính toán số lượng features đầu vào
        # Features = (số cells ngang * số cells dọc) * số bins
        self.num_hog_features = cell_grid[0] * cell_grid[1] * bins
        
        super().__init__(num_features=self.num_hog_features, num_classes=num_classes, **kwargs)
        
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

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=True, learning_rate=0.1, epochs=100):
        print("Extracting HOG features...")
        X_hog = self._transform(X)
        print(f"HOG Feature shape: {X_hog.shape}") # Ví dụ: (60000, 144)
        
        super().fit(X_hog, y, verbose=verbose, learning_rate=learning_rate, epochs=epochs)

    def predict(self, X: np.ndarray, use_best=True) -> int:
        X_hog = self._transform(X)
        return super().predict(X_hog, use_best=use_best)