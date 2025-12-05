"""
Training script to train all models at once and save their weights.
"""

import numpy as np
import os
from models.softmax_regression import SoftmaxRegression
from models.model_pixel import PixelSoftmax
from models.model_pca import PCASoftmax
from models.model_edge import EdgeSoftmax
from models.model_hog import HOGSoftmax
from models.model_block import BlockSoftmax
from models.utils.loader import load_dataset

# Create weights directory if not exists
os.makedirs('models/weights', exist_ok=True)

# Load data
print("Loading MNIST data...")
X_train, y_train, X_val, y_val = load_dataset('data/mnist_data.npz')

print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

# Define models to train
models_config = [
    {
        'name': 'pixel',
        'class': PixelSoftmax,
        'params': {'num_features': 784, 'num_classes': 10},
        'train_params': {'learning_rate': 0.1, 'epochs': 100}
    },
    {
        'name': 'edge',
        'class': EdgeSoftmax,
        'params': {'num_features': 784, 'num_classes': 10},
        'train_params': {'learning_rate': 0.1, 'epochs': 100}
    },
    {
        'name': 'block',
        'class': BlockSoftmax,
        'params': {'num_classes': 10, 'grid_size': (14, 14)},
        'train_params': {'learning_rate': 0.1, 'epochs': 100}
    },
    {
        'name': 'hog',
        'class': HOGSoftmax,
        'params': {'num_classes': 10, 'bins': 9, 'cell_grid': (7, 7)},
        'train_params': {'learning_rate': 0.1, 'epochs': 100}
    },
    {
        'name': 'pca',
        'class': PCASoftmax,
        'params': {'num_features': 50, 'num_classes': 10},
        'train_params': {'learning_rate': 0.1, 'epochs': 100}
    },
]

# Train each model
results = {}

for config in models_config:
    print("\n" + "="*70)
    print(f"Training {config['name'].upper()} model...")
    print("="*70)
    
    # Create model
    model = config['class'](**config['params'])
    
    # Train model
    model.fit(X_train, y_train, verbose=True, **config['train_params'])
    
    # Evaluate on validation set
    val_acc = model._accuracy(X_val, y_val, use_best=True)
    
    # Save model weights
    weight_path = f"models/weights/{config['name']}_best.npy"
    model.save_best_model(weight_path)
    
    # Store results
    results[config['name']] = {
        'val_accuracy': val_acc,
        'weight_path': weight_path
    }
    
    print(f"\n✓ {config['name'].upper()} Model:")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Weights saved to: {weight_path}")

# Print summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"{'Model':<15} {'Val Accuracy':<15} {'Weight Path'}")
print("-"*70)
for name, result in results.items():
    print(f"{name:<15} {result['val_accuracy']:<15.4f} {result['weight_path']}")

print("\n✓ All models trained successfully!")
print(f"Weights saved in: models/weights/")
