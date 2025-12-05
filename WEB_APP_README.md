# MNIST Digit Recognition - Multi-Model Web Application

Web application hoàn chỉnh để nhận diện chữ số viết tay với nhiều phương pháp trích xuất đặc trưng khác nhau.

## 🎯 Features

- ✏️ Vẽ chữ số trực tiếp trên canvas
- 🤖 So sánh 5 models với các phương pháp feature extraction khác nhau:
  - **Pixel**: Raw pixel intensity
  - **Edge**: Sobel edge detection  
  - **Block**: Block averaging (7x7 grid)
  - **HOG**: Histogram of Oriented Gradients
  - **PCA**: Principal Component Analysis (50 components)
- 📊 Hiển thị feature visualization của từng model
- 📈 Probability distribution cho tất cả các lớp (0-9)
- 🎨 UI đẹp, responsive

## 📁 Project Structure

```
├── MODELS/
│   ├── models/
│   │   ├── softmax_regression.py      # Base model
│   │   ├── model_pixel.py             # Pixel model
│   │   ├── model_edge.py              # Edge model
│   │   ├── model_block.py             # Block model
│   │   ├── model_hog.py               # HOG model
│   │   ├── model_pca.py               # PCA model
│   │   └── weights/                   # Trained model weights
│   ├── train_all_models.py            # Training script
│   └── data/
│       └── mnist_data.npz             # MNIST dataset
├── BE/
│   └── app.py                         # Flask API server
└── FE/
    ├── index.html                     # Main HTML
    ├── app.js                         # JavaScript
    └── app.css                        # Styling
```

## 🚀 Quick Start

### Step 1: Train Models

```bash
cd MODELS
python train_all_models.py
```

Sau khi train xong, weights sẽ được lưu trong `MODELS/models/weights/`:
- `pixel_best.npy`
- `edge_best.npy`
- `block_best.npy`
- `hog_best.npy`
- `pca_best.npy`

### Step 2: Start Backend Server

```bash
cd BE
pip install flask flask-cors pillow numpy opencv-python
python app.py
```

Server sẽ chạy tại: `http://localhost:5000`

### Step 3: Open Frontend

Mở file `FE/index.html` trong browser hoặc sử dụng Live Server:

```bash
cd FE
python -m http.server 8000
# Hoặc sử dụng VS Code Live Server extension
```

Truy cập: `http://localhost:8000`

## 🔌 API Endpoints

### GET /api/models
Lấy danh sách models available.

**Response:**
```json
{
  "models": [
    {"id": "pixel", "name": "Pixel Intensity"},
    {"id": "edge", "name": "Sobel Edge Detection"},
    ...
  ],
  "total": 5
}
```

### POST /api/predict
Predict digit từ canvas image.

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "model": "all"  // hoặc "pixel", "edge", etc.
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "pixel": {
      "digit": 5,
      "confidence": 0.95,
      "probabilities": [0.01, 0.02, ...],
      "visualization": "data:image/png;base64,...",
      "model_name": "Pixel Intensity"
    },
    ...
  }
}
```

### GET /api/visualize/{model}/{class}
Get feature visualization cho model và class cụ thể.

**Example:** `/api/visualize/pixel/5`

**Response:**
```json
{
  "success": true,
  "visualization": "data:image/png;base64,...",
  "class": 5,
  "model": "pixel",
  "model_name": "Pixel Intensity"
}
```

### GET /health
Health check endpoint.

## 🧠 Models Explained

### 1. Pixel Model
- **Input**: Raw 28x28 pixel values (784 features)
- **Preprocessing**: Flatten + normalize
- **Visualization**: Direct weight visualization as 28x28 image

### 2. Edge Model  
- **Input**: Sobel edge detection features (784 features)
- **Preprocessing**: Sobel gradient magnitude + normalize
- **Visualization**: Edge detection weights as 28x28 image

### 3. Block Model
- **Input**: 7x7 block-averaged features (49 features)
- **Preprocessing**: Average pixels in 4x4 blocks
- **Visualization**: Upsampled block weights to 28x28

### 4. HOG Model
- **Input**: Histogram of Oriented Gradients (144 features)
- **Preprocessing**: 4x4 cell grid, 9 orientation bins
- **Visualization**: Cell importance heatmap upsampled to 28x28

### 5. PCA Model
- **Input**: 50 principal components
- **Preprocessing**: PCA dimensionality reduction from 784 to 50
- **Visualization**: Project PCA weights back to pixel space

## 📊 Training Details

- **Dataset**: MNIST (60,000 training, 10,000 test)
- **Optimizer**: Gradient Descent
- **Learning Rate**: 0.1
- **Epochs**: 100
- **Loss**: Cross-Entropy

## 🎨 UI Features

- **Canvas**: 280x280 drawing area
- **Model Selector**: Choose single model or compare all
- **Results Grid**: Responsive grid layout showing:
  - Predicted digit + confidence
  - Feature visualization
  - Probability distribution bar chart
- **Color-coded Confidence**:
  - 🟢 Green: > 90%
  - 🟠 Orange: 70-90%
  - 🔴 Red: < 70%

## 🛠️ Development

### Adding New Models

1. Create model class inheriting from `SoftmaxRegression`
2. Implement feature extraction in `fit()` and `predict()`
3. Add `get_feature_visualization()` method
4. Add config to `train_all_models.py` and `BE/app.py`
5. Train and deploy!

### Customizing UI

- **Colors**: Edit gradient in `app.css` (`.container header`)
- **Layout**: Modify grid in `main` CSS
- **Canvas Size**: Change canvas width/height in HTML

## 📝 Requirements

```txt
# Backend
flask>=2.0.0
flask-cors>=3.0.0
numpy>=1.20.0
pillow>=8.0.0
opencv-python>=4.5.0

# Models
numpy>=1.20.0
opencv-python>=4.5.0
rich>=10.0.0
```

## 🐛 Troubleshooting

### Models not loading
- Ensure weights exist in `MODELS/models/weights/`
- Check file names match config in `BE/app.py`
- Run `train_all_models.py` first

### CORS errors
- Ensure Flask-CORS is installed
- Check API_BASE_URL in `app.js` matches server

### Canvas not drawing
- Check browser console for errors
- Ensure `app.js` is loaded correctly

## 📄 License

MIT License - Feel free to use for educational purposes!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📧 Contact

For issues or questions, please open an issue on GitHub.

---

**Made with ❤️ for Machine Learning Education**
