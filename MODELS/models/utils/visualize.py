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
    # Lấy 1 ảnh mẫu
    img_original = X_sample.reshape(28, 28)
    
    # Tính feature
    # Lưu ý: hàm _transform nhận vào batch (N, ...), nên cần bọc X_sample vào list
    # Kết quả trả về (1, 49)
    feature_vector = model._transform(X_sample[np.newaxis, :])
    
    # Reshape lại thành lưới (7x7) để vẽ
    img_blocked = feature_vector.reshape(model.grid_size)
    
    # Vẽ
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Ảnh gốc
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title(f"Original (28x28)\n784 pixels")
    axes[0].axis('off')
    
    # Ảnh Features (Block)
    # Dùng nội suy 'nearest' để nhìn rõ các ô vuông
    axes[1].imshow(img_blocked, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Block Features {model.grid_size}\n{model.num_block_features} features")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Cách dùng ---
# model = BlockSoftmax(num_classes=10, grid_size=(7, 7))
# model.fit(X_train, y_train)
# visualize_block_features(model, X_train[0])

import matplotlib.pyplot as plt

def visualize_hog_features(model, X_sample):
    """
    Visualize HOG của 1 ảnh mẫu
    """
    # Lấy 1 ảnh mẫu
    img = X_sample.reshape(28, 28).astype(np.float32)
    
    # Tính gradient lại để vẽ
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Vẽ
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Original Image")
    
    # Vẽ độ lớn gradient (Magnitude) - Sẽ thấy biên của chữ số sáng lên
    ax2.imshow(mag, cmap='hot')
    ax2.set_title("Gradient Magnitude (Edges)")
    
    # Vẽ Histogram (Feature vector)
    hog_vec = model._compute_hog_single(img)
    ax3.bar(range(len(hog_vec)), hog_vec)
    ax3.set_title("HOG Feature Vector")
    
    plt.show()

# Cách dùng:
# model = HOGSoftmax(num_classes=10)
# visualize_hog_features(model, X_train[0])