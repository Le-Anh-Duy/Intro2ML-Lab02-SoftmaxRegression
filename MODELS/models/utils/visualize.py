import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2


def MNIST_show(data: np.array = None, label: np.array = None):

    data = np.reshape(data, (28, 28))
    plt.imshow(data, cmap='viridis', origin='lower') # 'viridis' is a common colormap, 'origin' sets the (0,0) point
    plt.colorbar(label='Value')
    plt.title(f'Label: {label}')
    plt.show()

    
def MNIST_show(content: tuple):

    data, label = content
    data = np.reshape(data, (28, 28))
    plt.imshow(data, cmap='viridis', origin='upper') # 'viridis' is a common colormap, 'origin' sets the (0,0) point
    plt.colorbar(label='Value')
    plt.title(f'Label: {label}')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: list = None):
    """
    Visualize the confusion matrix using a Seaborn heatmap.

    Args:
        cm (np.ndarray): The confusion matrix to plot.
        class_names (list, optional): List of class names for axis tick labels. 
                                      Defaults to None.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=15)
    plt.show()

def visualize_PCA_features(model):
    """
    Hàm visualize các thành phần bên trong của model PCASoftmax
    """
    if model.components is None or model.weights is None:
        print("Model chưa được train!")
        return

    # Kích thước ảnh gốc
    img_shape = (28, 28) 

    # --------------------------------------------
    # 1. Visualize Mean Image (Ảnh trung bình)
    # --------------------------------------------
    plt.figure(figsize=(4, 4))
    plt.imshow(model.pca_mean.reshape(img_shape), cmap='gray')
    plt.title("Mean Image (Ảnh trung bình)")
    plt.axis('off')
    plt.show()

    # --------------------------------------------
    # 2. Visualize Eigen-digits (Các Features của PCA)
    # --------------------------------------------
    # model.components có shape (784, n_components)
    # Mỗi CỘT là một vector riêng (một feature)
    
    n_features_to_show = min(10, model.n_components) # Xem 10 feature đầu tiên
    
    fig, axes = plt.subplots((n_features_to_show + 4) // 5, 5, figsize=(12, 5))
    fig.suptitle(f'Top {n_features_to_show} Eigen-digits (PCA Components)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < n_features_to_show:
            # Lấy vector cột thứ i và reshape lại thành 28x28
            eigen_digit = model.components[:, i].reshape(img_shape)
            
            # Dùng cmap='jet' hoặc 'seismic' để thấy rõ vùng dương/âm
            im = ax.imshow(eigen_digit, cmap='seismic')
            ax.set_title(f"Component {i+1}")
            ax.axis('off')
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    # --------------------------------------------
    # 3. Visualize Class Weights (Trọng số của từng số)
    # --------------------------------------------
    # Trọng số W của Softmax đang ở không gian PCA: (n_components, 10)
    # Ta cần chiếu ngược về không gian Pixel: W_pixel = Components . W_pca
    
    # Bỏ dòng bias (hàng đầu tiên) của weights
    W_pca = model.weights[1:, :] 
    
    # Phép nhân ma trận để khôi phục hình ảnh trọng số
    # (784, n_com) . (n_com, 10) -> (784, 10)
    W_pixel = np.dot(model.components, W_pca)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Reconstructed Weights per Class (Mô hình tìm gì ở mỗi số?)', fontsize=16)
    
    classes = range(10) # 0-9
    for i, ax in enumerate(axes.flat):
        # Lấy cột trọng số tương ứng với class i
        weight_img = W_pixel[:, i].reshape(img_shape)
        
        # Vẽ (màu đỏ là khu vực mô hình thích, xanh là ghét)
        ax.imshow(weight_img, cmap='seismic') 
        ax.set_title(f"Digit: {classes[i]}")
        ax.axis('off')
    
    plt.show()

# --- Cách sử dụng ---
# Giả sử bạn đã có model đã train xong
# model = PCASoftmax(...)
# model.fit(X_train, y_train)

# Gọi hàm:
# visualize_model_features(model)
def visualize_block_features(model, X_sample):
    """
    So sánh ảnh gốc và ảnh sau khi qua Block Averaging
    """
    # Lấy 1 ảnh mẫu và reshape về 28x28
    img_original = X_sample.reshape(28, 28)
    
    # Tính feature (cần thêm axis để tạo batch N=1)
    # model._transform trả về (1, num_features)
    feature_vector = model._transform(img_original[np.newaxis, :, :])
    
    # Reshape lại thành lưới (ví dụ 7x7) để vẽ
    # model.grid_size là tuple (7, 7)
    img_blocked = feature_vector.reshape(model.grid_size)
    
    # Vẽ
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Ảnh gốc
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title(f"Original (28x28)")
    axes[0].axis('off')
    
    # Ảnh Features (Block)
    # Dùng interpolation='nearest' để thấy rõ các ô vuông (pixel to)
    axes[1].imshow(img_blocked, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Block Features {model.grid_size}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Cách dùng ---
# model = BlockSoftmax(num_classes=10, grid_size=(7, 7))
# model.fit(X_train, y_train)
# visualize_block_features(model, X_train[0])

def visualize_hog_features(model, X_sample):
    """
    Visualize HOG của 1 ảnh mẫu: Ảnh gốc, Gradient Magnitude và Histogram
    """
    # Lấy 1 ảnh mẫu và ép kiểu float để tính toán
    img = X_sample.reshape(28, 28).astype(np.float32)
    
    # 1. Tính toán Gradient (chỉ để vẽ minh họa)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # 2. Tính Feature Vector thực tế bằng hàm _transform của model
    # Lưu ý: _transform yêu cầu input (N, H, W) nên cần thêm trục (np.newaxis)
    # Kết quả trả về (1, num_features), cần flatten ra 1D để vẽ biểu đồ
    hog_vec = model._transform(img[np.newaxis, :, :]).flatten()
    
    # --- Vẽ đồ thị ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    
    # Ảnh gốc
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Gradient Magnitude (Độ lớn biên cạnh)
    ax2.imshow(mag, cmap='hot')
    ax2.set_title("Gradient Magnitude\n(Edges)")
    ax2.axis('off')
    
    # Histogram Feature Vector
    ax3.bar(range(len(hog_vec)), hog_vec, width=1.0)
    ax3.set_title(f"HOG Feature Vector\n({len(hog_vec)} dimensions)")
    ax3.set_xlabel("Feature Index")
    ax3.set_ylabel("Normalized Strength")
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_hog_features_order_by_angle(model, X_sample, n_bins=9, grid_size=(4,4)):
    """
    Visualize HOG chia thành n_bins hình ảnh riêng biệt.
    Mỗi hình ảnh thể hiện độ mạnh của gradient theo hướng góc đó trên lưới (grid).
    
    Args:
        model: Đối tượng model có hàm _transform
        X_sample: Input ảnh (flatten hoặc 28x28)
        n_bins: Số lượng bin góc (mặc định 9)
        grid_size: Kích thước lưới chia ảnh (Height, Width) - tương ứng cells_per_block
    """
    # 1. Chuẩn bị ảnh gốc để vẽ minh họa
    img = X_sample.reshape(28, 28).astype(np.float32)
    
    # 2. Tính Feature Vector từ model
    # Giả sử output là (1, N) -> flatten thành (N,)
    hog_vec = model._transform(img[np.newaxis, :, :]).flatten()
    
    # --- XỬ LÝ RESHAPE VECTOR ---
    h_grid, w_grid = grid_size
    expected_len = h_grid * w_grid * n_bins
    
    # Kiểm tra xem độ dài vector có khớp với cấu hình không
    if len(hog_vec) != expected_len:
        print(f"Cảnh báo: Độ dài vector ({len(hog_vec)}) không khớp với "
              f"Grid {grid_size} x Bins {n_bins} = {expected_len}.")
        return

    # Reshape lại vector. 
    # Giả định thứ tự dữ liệu là: Duyệt qua từng ô (Cell) -> Duyệt qua từng Bin
    # Shape: (Height_Grid, Width_Grid, Num_Bins)
    hog_cube = hog_vec.reshape(h_grid, w_grid, n_bins)

    # --- VẼ ĐỒ THỊ ---
    # Tạo Grid 3x3 cho 9 bins (hoặc tuỳ chỉnh nếu số bin khác 9)
    rows = int(np.ceil(np.sqrt(n_bins)))
    cols = int(np.ceil(n_bins / rows))
    
    fig = plt.figure(figsize=(12, 12))
    
    # Tiêu đề chung
    plt.suptitle(f'HOG Features Intensity by Angle\n(Grid: {grid_size}, Bins: {n_bins})', fontsize=16)

    # Vẽ từng Bin
    for i in range(n_bins):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Lấy bản đồ đặc trưng của bin thứ i (Shape: 4x4)
        feature_map = hog_cube[:, :, i]
        
        # Vẽ Heatmap
        # interpolation='nearest' giúp nhìn rõ từng ô vuông của grid
        im = ax.imshow(feature_map, cmap='hot', interpolation='nearest', vmin=0, vmax=np.max(hog_vec))
        
        # Tính góc đại diện (Giả sử 0-180 độ chia đều)
        angle_deg = i * (180 / n_bins)
        
        ax.set_title(f"Bin {i}: ~{int(angle_deg)}°")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Thêm colorbar nhỏ để tham chiếu độ lớn
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Chừa chỗ cho suptitle
    plt.show()

    # (Tuỳ chọn) Vẽ thêm ảnh gốc để dễ đối chiếu
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

# --- HƯỚNG DẪN SỬ DỤNG ---
# Giả sử bạn đã khởi tạo model và load data
# model = HOGSoftmax(...) 
# X_sample = X_train[123]

# visualize_hog_features_order_by_angle(model, X_sample, n_bins=9, grid_size=(4,4))

# Cách dùng:
# model = HOGSoftmax(num_classes=10)
# visualize_hog_features(model, X_train[0])