# 📋 TODO LIST - MNIST Softmax Regression Project

## ✅ Đã hoàn thành
- [x] Cấu trúc thư mục project (BE, FE, MODELS)
- [x] Setup requirements.txt
- [x] Tải MNIST dataset từ Kaggle
- [x] Class base `SoftmaxRegression` (baseline đơn giản)

---

## 🎯 CẦN LÀM TIẾP THEO

### 📁 MODELS/models/utils/ (Bắt đầu từ đây)

#### 1. **dataset.py** - Xử lý dữ liệu MNIST
**Mục tiêu**: Load và chuẩn bị dữ liệu cho training

**TODO**:
- [ ] Viết class `MNISTDataset`:
  - [ ] Đọc file `.idx3-ubyte` (images) và `.idx1-ubyte` (labels)
  - [ ] Parse binary format (magic number, dimensions, data)
  - [ ] Trả về numpy arrays: `(N, 28, 28)` cho images, `(N,)` cho labels
  - [ ] Method `__len__()` và `__getitem__(idx)`

- [ ] Viết class `DataLoader`:
  - [ ] Chia data thành batches
  - [ ] Shuffle data mỗi epoch
  - [ ] Method `__iter__()` để duyệt qua batches
  - [ ] Hỗ trợ `batch_size` parameter

**Hint**: 
- MNIST binary format: magic number (4 bytes) → dimensions → pixel data
- Dùng `struct.unpack()` để đọc binary
- Images: uint8 [0-255], cần normalize về [0-1]

---

#### 2. **visualization.py** - Visualize dữ liệu và kết quả
**Mục tiêu**: Hiển thị ảnh, confusion matrix, training curves

**TODO**:
- [ ] Function `plot_samples(images, labels, predictions=None)`:
  - [ ] Hiển thị grid ảnh với matplotlib
  - [ ] Show true label và predicted label (nếu có)

- [ ] Function `plot_confusion_matrix(y_true, y_pred)`:
  - [ ] Tính confusion matrix
  - [ ] Vẽ heatmap với matplotlib/seaborn

- [ ] Function `plot_training_history(losses, accuracies)`:
  - [ ] Vẽ loss curve và accuracy curve
  - [ ] Subplot cho train và validation

---

### 📁 MODELS/models/src/ (Core Models)

#### 3. **base.py** - Implement Softmax Regression
**Mục tiêu**: Hoàn thiện class base với NumPy thuần

**TODO**:
- [ ] Method `fit(X, y)`:
  - [ ] Initialize weights với Xavier/He initialization
  - [ ] One-hot encode labels
  - [ ] Implement mini-batch gradient descent loop:
    ```
    for epoch in range(num_epochs):
        shuffle data
        for each batch:
            forward pass → compute loss → backward pass → update weights
    ```
  - [ ] Forward pass: `z = X @ W + b`, `softmax(z)`
  - [ ] Loss: Cross-entropy + L2 regularization
  - [ ] Backward: Gradient của cross-entropy wrt W, b
  - [ ] Update: `W -= learning_rate * dW`

- [ ] Method `predict(X)`:
  - [ ] Forward pass
  - [ ] Return argmax của probabilities

- [ ] Method `predict_proba(X)`:
  - [ ] Return softmax probabilities

- [ ] Method `score(X, y)`:
  - [ ] Accuracy = mean(predictions == y)

**Công thức quan trọng**:
- Softmax: `softmax(z_i) = exp(z_i) / sum(exp(z_j))`
- Cross-entropy loss: `L = -mean(sum(y_true * log(y_pred)))`
- Gradient: `dL/dW = X.T @ (y_pred - y_true) / batch_size`

---

#### 4. **model_pixel.py** - Raw Pixel Model
**Mục tiêu**: Model đơn giản nhất, dùng pixel thô

**TODO**:
- [ ] Inherit từ `SoftmaxRegression`
- [ ] Override method `preprocess_features(X)`:
  - [ ] Flatten: `(N, 28, 28) → (N, 784)`
  - [ ] Normalize: `X / 255.0`
  - [ ] Return normalized features

---

#### 5. **model_edge.py** - Edge Detection Model
**Mục tiêu**: Trích xuất features bằng edge detection

**TODO**:
- [ ] Inherit từ `SoftmaxRegression`
- [ ] Override method `preprocess_features(X)`:
  - [ ] Loop qua từng ảnh
  - [ ] Apply Sobel operator (cv2.Sobel):
    - Sobel X (vertical edges)
    - Sobel Y (horizontal edges)
    - Magnitude = sqrt(Sx² + Sy²)
  - [ ] Apply Canny edge detection (cv2.Canny)
  - [ ] Concatenate [sobel_mag, canny]
  - [ ] Flatten và normalize
  - [ ] Return edge features

**Hint**: 
- Sobel: Detect gradients → highlight edges
- Canny: Complete edge detection algorithm
- Kết hợp cả 2 để có nhiều thông tin hơn

---

#### 6. **model_pca.py** - PCA Dimensionality Reduction
**Mục tiêu**: Giảm chiều dữ liệu từ 784 xuống ~50-100 dimensions

**TODO**:
- [ ] Inherit từ `SoftmaxRegression`
- [ ] Method `fit_pca(X)`:
  - [ ] Center data: `X_centered = X - mean(X)`
  - [ ] Compute covariance matrix: `C = X_centered.T @ X_centered`
  - [ ] Eigenvalue decomposition: `eig_vals, eig_vecs = np.linalg.eigh(C)`
  - [ ] Sort eigenvectors by eigenvalues (descending)
  - [ ] Select top k components (preserve 95% variance)
  - [ ] Store `self.mean`, `self.components`

- [ ] Method `transform_pca(X)`:
  - [ ] Center: `X - self.mean`
  - [ ] Project: `X_centered @ self.components`
  - [ ] Return reduced features

- [ ] Override `preprocess_features(X)`:
  - [ ] Flatten và normalize
  - [ ] Apply PCA transform
  - [ ] Return reduced features

- [ ] Override `fit(X, y)`:
  - [ ] Fit PCA trước
  - [ ] Gọi `super().fit(X, y)`

**Công thức PCA**:
- Covariance: `C = (1/n) * X.T @ X`
- Explained variance: `eig_val / sum(eig_vals)`
- Transform: `X_new = (X - μ) @ V_k`

---

### 📁 MODELS/ (Root level)

#### 7. **train.py** - Train tất cả models
**Mục tiêu**: Script để train và save 3 models

**TODO**:
- [ ] Load MNIST dataset
- [ ] Split train/validation (nếu cần)
- [ ] Train từng model:
  ```python
  pixel_model = PixelSoftmaxRegression(lr=0.1, epochs=500)
  pixel_model.fit(X_train, y_train)
  ```
- [ ] Evaluate trên test set
- [ ] Save models với pickle: `pickle.dump(model, f)`
- [ ] Save vào `trained/pixel_model.pkl`, etc.
- [ ] Print accuracy của từng model

---

### 📁 BE/ (Backend API)

#### 8. **app.py** - Flask API
**Mục tiêu**: API endpoint để predict từ FE

**TODO**:
- [ ] Load 3 trained models khi start server
- [ ] Route `POST /predict`:
  - [ ] Nhận base64 image từ frontend
  - [ ] Decode base64 → PIL Image
  - [ ] Resize về 28x28 grayscale
  - [ ] Invert colors (canvas trắng → MNIST đen)
  - [ ] Normalize pixel values
  - [ ] Call `model.predict_proba()` cho cả 3 models
  - [ ] Return JSON:
    ```json
    {
      "pixel_model": {"prediction": 5, "probabilities": [...], "confidence": 0.95},
      "edge_model": {...},
      "pca_model": {...}
    }
    ```
- [ ] Route `GET /health`: Check models loaded
- [ ] Enable CORS

---

### 📁 FE/ (Frontend UI)

#### 9. **script.js** - Drawing Canvas Logic
**Mục tiêu**: Vẽ số và gửi đến backend

**TODO**:
- [ ] Canvas drawing:
  - [ ] Mouse events: mousedown, mousemove, mouseup
  - [ ] Touch events cho mobile
  - [ ] Draw với `ctx.lineTo()` và `ctx.stroke()`
  - [ ] Brush size phù hợp (~15px)

- [ ] Clear button: Reset canvas về trắng

- [ ] Predict button:
  - [ ] Get canvas data: `canvas.toDataURL('image/png')`
  - [ ] Fetch POST `/predict` với base64 image
  - [ ] Parse response JSON
  - [ ] Display results

- [ ] Display results:
  - [ ] Show predicted digit (lớn, bold)
  - [ ] Show confidence score
  - [ ] Show probability bars cho 10 digits
  - [ ] Repeat cho cả 3 models

---

#### 10. **style.css** - UI Styling
**TODO**:
- [ ] Canvas styling: border, cursor
- [ ] Button styles: hover effects
- [ ] Results layout: grid/flexbox
- [ ] Probability bars: height based on probability
- [ ] Responsive design cho mobile
- [ ] Color scheme: gradient background

---

#### 11. **index.html** - HTML Structure
**TODO**:
- [ ] Header: Title và instructions
- [ ] Canvas section với controls
- [ ] Results section (initially hidden)
- [ ] Link CSS và JS files

---

## 🎓 THỨ TỰ KHUYÊN DÙNG

### Phase 1: Data Pipeline ⭐ (BẮT ĐẦU TỪ ĐÂY)
1. dataset.py - Đọc và load MNIST
2. `visualization.py` - Xem data có đúng không
3. Test xem có load được ảnh + label không

### Phase 2: Core Model 🧠
4. `base.py` - Implement Softmax Regression
5. `model_pixel.py` - Test với pixel model trước
6. Train 1 epoch xem loss có giảm không

### Phase 3: Advanced Models 🚀
7. `model_edge.py` - Edge features
8. `model_pca.py` - PCA reduction
9. `train.py` - Train cả 3 models đến hội tụ

### Phase 4: Backend 🔧
10. `app.py` - Flask API
11. Test API với Postman/curl

### Phase 5: Frontend 🎨
12. `script.js` + `index.html` + `style.css`
13. Test end-to-end flow

---

## 📚 TÀI LIỆU THAM KHẢO

**Softmax Regression**:
- Cross-entropy loss derivation
- Gradient descent with softmax
- NumPy vectorization tricks

**MNIST Binary Format**:
- http://yann.lecun.com/exdb/mnist/
- Magic numbers và byte order

**Edge Detection**:
- Sobel operator
- Canny edge detection
- OpenCV documentation

**PCA**:
- Eigenvalue decomposition
- Variance explained
- Dimensionality reduction

---

## 🐛 DEBUG TIPS

- Print shapes thường xuyên: `print(X.shape)`
- Check numerical stability: softmax overflow → subtract max
- Visualize intermediate results
- Start với small dataset để test nhanh
- Use `np.set_printoptions(precision=3)` để dễ đọc

---

**Chúc bạn code vui! 🚀**