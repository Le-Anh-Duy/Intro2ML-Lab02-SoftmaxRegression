# MODELS - Thư mục chứa các model và dữ liệu

## Cấu trúc thư mục

```
MODELS/
├── data/                 # Thư mục chứa dataset MNIST
├── models/              
│   ├── src/             # Mã nguồn các model
│   └── utils/           # Các hàm tiện ích
└── trained/             # Model đã train (sẽ tạo sau khi train)
```

## Mô tả các thư mục

### 📁 `models/src/`
Chứa mã nguồn các model Softmax Regression:
- `base.py`: Class cơ sở `SoftmaxRegression` để các model khác kế thừa
- `model_pixel.py`: Model sử dụng raw pixel intensity
- `model_edge.py`: Model sử dụng edge detection (Sobel/Canny)
- `model_pca.py`: Model sử dụng PCA để giảm chiều dữ liệu

### 📁 `models/utils/`
Chứa các hàm tiện ích:
- Hàm load và preprocess MNIST dataset
- Hàm visualization
- Hàm đánh giá model

### 📁 `data/`
Chứa MNIST dataset sau khi tải về:
- `train-images-idx3-ubyte.gz`: Ảnh training (60,000 ảnh)
- `train-labels-idx1-ubyte.gz`: Label training
- `t10k-images-idx3-ubyte.gz`: Ảnh test (10,000 ảnh)
- `t10k-labels-idx1-ubyte.gz`: Label test

### 📁 `trained/`
Chứa các model đã train (file .pkl):
- `pixel_model.pkl`
- `edge_model.pkl`
- `pca_model.pkl`

## Setup và tải MNIST Dataset

### Cách 1: Tải từ Kaggle (Khuyên dùng - Nhanh nhất)

**Bước 1: Cài đặt Kaggle API**
```bash
pip install kaggle
```

**Bước 2: Cấu hình Kaggle credentials**
1. Truy cập https://www.kaggle.com/settings
2. Scroll xuống "API" section → Click "Create New Token"
3. File `kaggle.json` sẽ được tải về
4. Đặt file vào:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

**Bước 3: Chạy script tải dataset**
```bash
cd MODELS/
python download.py
```

Hoặc dùng Kaggle CLI trực tiếp:
```bash
kaggle datasets download -d hojjatk/mnist-dataset
unzip mnist-dataset.zip
```

### Cách 2: Tự động tải khi train

```bash
cd MODELS
python train.py
```

Script `train.py` sẽ tự động:
1. Tải MNIST dataset từ http://yann.lecun.com/exdb/mnist/
2. Lưu vào thư mục `data/`
3. Train cả 3 model variants
4. Lưu model vào thư mục `trained/`

### Cách 3: Tải thủ công

Tải 4 file từ trang web MNIST:
```
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Đặt vào thư mục `MODELS/data/`

### Cách 4: Dùng Python để tải

```python
from data.mnist_loader import MNISTLoader

loader = MNISTLoader(data_dir='./data')
X_train, y_train, X_test, y_test = loader.load_data()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Output: Train: (60000, 28, 28), Test: (10000, 28, 28)
```

## MNIST Dataset Info

- **Tên**: MNIST Handwritten Digits
- **Kích thước**: 60,000 ảnh train + 10,000 ảnh test
- **Định dạng**: Ảnh grayscale 28x28 pixels
- **Số classes**: 10 (chữ số 0-9)
- **Nguồn**: http://yann.lecun.com/exdb/mnist/

## Requirements

```bash
pip install -r requirements.txt
```

Các thư viện cần thiết:
- numpy: Tính toán ma trận
- opencv-python: Edge detection
- scikit-learn: PCA
- matplotlib: Visualization