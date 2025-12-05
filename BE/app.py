from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
import os

# Add MODELS path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MODELS'))

from models.softmax_regression import SoftmaxRegression
from models.model_pixel import PixelSoftmax
from models.model_pca import PCASoftmax
from models.model_edge import EdgeSoftmax
from models.model_hog import HOGSoftmax
from models.model_block import BlockSoftmax

app = Flask(__name__)
CORS(app)

# Model configurations
MODELS_CONFIG = {
    'pixel': {
        'class': PixelSoftmax,
        'params': {'num_features': 784, 'num_classes': 10},
        'weight_file': 'pixel_best.npz',
        'name': 'Pixel Intensity'
    },
    'edge': {
        'class': EdgeSoftmax,
        'params': {'num_features': 784, 'num_classes': 10},
        'weight_file': 'edge_best.npz',
        'name': 'Sobel Edge Detection'
    },
    'block': {
        'class': BlockSoftmax,
        'params': {'num_classes': 10, 'grid_size': (7, 7)},
        'weight_file': 'block_best.npz',
        'name': 'Block Averaging'
    },
    'hog': {
        'class': HOGSoftmax,
        'params': {'num_classes': 10, 'bins': 9, 'cell_grid': (4, 4)},
        'weight_file': 'hog_best.npz',
        'name': 'HOG Features'
    },
    'pca': {
        'class': PCASoftmax,
        'params': {'num_features': 50, 'num_classes': 10},
        'weight_file': 'pca_best.npz',
        'name': 'PCA Reduction'
    },
}

# Load models
WEIGHT_DIR = os.path.join(os.path.dirname(__file__), '..', 'MODELS', 'models', 'weights')
MODELS = {}
def load_models():
    """Load all trained models."""
    for model_key, config in MODELS_CONFIG.items():
        try:
            # Create model instance
            model = config['class'](**config['params'])
            
            # Load weights
            weight_path = os.path.join(WEIGHT_DIR, config['weight_file'])
            model.load_weight(weight_path)
            
            MODELS[model_key] = model
            print(f"✓ Loaded {config['name']} model")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"✗ Error loading {model_key} model: {e}")
            MODELS[model_key] = None

print("Loading models...")
load_models()
print(f"Models loaded: {list(MODELS.keys())}")

def preprocess_canvas_image(base64_image):
    """Preprocess image from canvas to 28x28 grayscale."""
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
    
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    image = image.convert('L')


    # 1. Lấy bounding box của vùng vẽ (cắt bỏ viền đen thừa)
    # Đảo ngược màu để tìm vùng có mực (vì getbbox tìm vùng khác 0)
    # Giả sử vẽ trắng trên đen
    coords = image.getbbox() 
    if coords:
        image = image.crop(coords)
    
    # 2. Resize về 20x20 (để chừa lề) thay vì 28x28 ngay
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    # 3. Dán vào giữa nền đen 28x28 (Center padding)
    new_image = Image.new('L', (28, 28), 0) # 0 là màu đen
    
    # Tính toán vị trí để paste vào giữa
    paste_x = (28 - image.width) // 2
    paste_y = (28 - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    image_array = np.array(new_image, dtype=np.float32)
    return image_array.reshape(1, 28, 28)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    available_models = []
    for key, model in MODELS.items():
        if model is not None:
            available_models.append({
                'id': key,
                'name': MODELS_CONFIG[key]['name']
            })
    
    return jsonify({'models': available_models, 'total': len(available_models)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict digit from canvas image."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = preprocess_canvas_image(data['image'])
        requested_model = data.get('model', 'all')
        
        # Convert preprocessed image to base64 for frontend display
        preprocessed_img = image[0]  # Shape (28, 28)
        preprocessed_normalized = (preprocessed_img / 255.0 * 255).astype(np.uint8)
        preprocessed_pil = Image.fromarray(preprocessed_normalized, mode='L')
        buffered = BytesIO()
        preprocessed_pil.save(buffered, format="PNG")
        preprocessed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        predictions = {}
        
        if requested_model == 'all':
            models_to_use = MODELS.items()
        else:
            if requested_model not in MODELS:
                return jsonify({'error': f'Model {requested_model} not found'}), 404
            models_to_use = [(requested_model, MODELS[requested_model])]
        
        for model_key, model in models_to_use:
            if model is None:
                continue
            
            try:
                # print(model.__dict__)
                proba = model.predict_proba(image, use_best=True)
                pred = model.predict(image, use_best=True)
                
                # Get visualization of preprocessed features (pass the input image)
                visualization = model.get_feature_visualization(image[0])  # Pass 28x28 image
                
                # Convert visualization to base64
                viz_normalized = ((visualization - visualization.min()) / 
                                 (visualization.max() - visualization.min() + 1e-8) * 255).astype(np.uint8)
                viz_img = Image.fromarray(viz_normalized, mode='L')
                buffered = BytesIO()
                viz_img.save(buffered, format="PNG")
                viz_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                predictions[model_key] = {
                    'digit': int(pred[0]),
                    'confidence': float(proba[0][pred[0]]),
                    'probabilities': proba[0].tolist(),
                    'visualization': f'data:image/png;base64,{viz_base64}',
                    'model_name': MODELS_CONFIG[model_key]['name']
                }
            except Exception as e:
                print(f"Error predicting with {model_key}: {e}")
                predictions[model_key] = {'error': str(e)}
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        
        return jsonify({
            'success': True, 
            'predictions': predictions,
            'preprocessed_image': f'data:image/png;base64,{preprocessed_base64}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    models_status = {key: model is not None for key, model in MODELS.items()}
    return jsonify({'status': 'healthy', 'models': models_status})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("MNIST Digit Recognition API Server")
    print("="*50)
    print(f"Loaded {len([m for m in MODELS.values() if m is not None])}/{len(MODELS)} models")
    print("Server starting on http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
