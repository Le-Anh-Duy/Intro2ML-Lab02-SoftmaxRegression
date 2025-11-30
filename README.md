# MNIST Digit Recognition with Softmax Regression

A web-based digit recognition system using three different feature extraction methods with Softmax Regression.

## Project Structure

```
├── BE/              # Backend Flask API
├── FE/              # Frontend web interface
├── MODELS/          # Machine learning models
│   ├── data/        # MNIST dataset
│   ├── models/      # Model implementations
│   └── trained/     # Saved trained models
└── requirements.txt # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

Navigate to the MODELS directory and train all three variants:

```bash
cd MODELS
python train.py
```

This will:
- Download MNIST dataset (if not present)
- Train three Softmax Regression models with different features:
  - **Pixel-based**: Raw normalized pixel intensities
  - **Edge-based**: Sobel/Canny edge detection features
  - **PCA-based**: Principal Component Analysis dimensionality reduction
- Save trained models to `MODELS/trained/`

### 3. Run Backend Server

```bash
cd BE
python app.py
```

The API will be available at `http://localhost:5000`

### 4. Open Frontend

Open `FE/index.html` in your web browser or serve it with:

```bash
cd FE
python -m http.server 8080
```

Then navigate to `http://localhost:8080`

## Usage

1. Draw a digit (0-9) on the canvas using your mouse
2. Click "Predict" to send the drawing to the backend
3. View predictions from all three models with confidence scores

## API Endpoints

### POST `/predict`

Receives base64-encoded image data and returns predictions from all three models.

**Request Body:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "pixel_model": {
    "prediction": 5,
    "probabilities": [0.01, 0.02, ...],
    "confidence": 0.95
  },
  "edge_model": { ... },
  "pca_model": { ... }
}
```

## Model Details

### Feature Vector Designs

1. **Pixel Model**: Uses 784-dimensional vectors (28x28 flattened) with normalized pixel intensities [0, 1]
2. **Edge Model**: Applies Sobel/Canny edge detection before flattening to enhance boundary features
3. **PCA Model**: Reduces dimensionality to 50 components while preserving 95% variance

All models use Softmax Regression (multinomial logistic regression) implemented purely in NumPy.