from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

app = Flask(__name__)
CORS(app)

# Load trained models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'trained')

def load_models():
    """Load all three trained models."""
    models = {}
    
    try:
        with open(os.path.join(MODEL_DIR, 'pixel_model.pkl'), 'rb') as f:
            models['pixel'] = pickle.load(f)
        print("Loaded pixel_model.pkl")
    except Exception as e:
        print(f"Error loading pixel_model: {e}")
        models['pixel'] = None
    
    try:
        with open(os.path.join(MODEL_DIR, 'edge_model.pkl'), 'rb') as f:
            models['edge'] = pickle.load(f)
        print("Loaded edge_model.pkl")
    except Exception as e:
        print(f"Error loading edge_model: {e}")
        models['edge'] = None
    
    try:
        with open(os.path.join(MODEL_DIR, 'pca_model.pkl'), 'rb') as f:
            models['pca'] = pickle.load(f)
        print("Loaded pca_model.pkl")
    except Exception as e:
        print(f"Error loading pca_model: {e}")
        models['pca'] = None
    
    return models

# Load models on startup
print("Loading trained models...")
MODELS = load_models()
print("Models loaded successfully!")

def preprocess_canvas_image(base64_image):
    """
    Preprocess image from canvas to 28x28 grayscale.
    
    Args:
        base64_image: Base64-encoded image string
        
    Returns:
        Preprocessed image (1, 28, 28)
    """
    # Remove data URL prefix if present
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors (canvas is white on black, MNIST is black on white)
    image = Image.eval(image, lambda x: 255 - x)
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Add batch dimension
    image_array = image_array.reshape(1, 28, 28)
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict digit from canvas image using all three models.
    
    Expected JSON body:
    {
        "image": "data:image/png;base64,..."
    }
    
    Returns:
    {
        "pixel_model": {"prediction": 5, "probabilities": [...], "confidence": 0.95},
        "edge_model": {...},
        "pca_model": {...}
    }
    """
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        image = preprocess_canvas_image(data['image'])
        
        # Get predictions from all models
        results = {}
        
        # Pixel model
        if MODELS['pixel'] is not None:
            proba = MODELS['pixel'].predict_proba(image)[0]
            prediction = int(np.argmax(proba))
            confidence = float(proba[prediction])
            
            results['pixel_model'] = {
                'prediction': prediction,
                'probabilities': proba.tolist(),
                'confidence': confidence
            }
        
        # Edge model
        if MODELS['edge'] is not None:
            proba = MODELS['edge'].predict_proba(image)[0]
            prediction = int(np.argmax(proba))
            confidence = float(proba[prediction])
            
            results['edge_model'] = {
                'prediction': prediction,
                'probabilities': proba.tolist(),
                'confidence': confidence
            }
        
        # PCA model
        if MODELS['pca'] is not None:
            proba = MODELS['pca'].predict_proba(image)[0]
            prediction = int(np.argmax(proba))
            confidence = float(proba[prediction])
            
            results['pca_model'] = {
                'prediction': prediction,
                'probabilities': proba.tolist(),
                'confidence': confidence
            }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models_loaded': {
            'pixel': MODELS['pixel'] is not None,
            'edge': MODELS['edge'] is not None,
            'pca': MODELS['pca'] is not None
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)