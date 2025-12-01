import sys
import os
import time
import pickle
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

# --- 1. Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'src'))
sys.path.append(os.path.join(current_dir, 'models', 'utils'))

# --- 2. Import Modules ---
try:
    from loader import load_dataset
    from dataset import dataset
    # Import 3 main models
    from model_pixel import PixelSoftmax
    from model_edge import EdgeSoftmax
    # from model_pca import PCASoftmax
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please check folder structure: models/src and models/utils")
    sys.exit(1)

console = Console()

def save_model(model, name):
    
    """Save trained model to disk"""
    
    save_dir = os.path.join(current_dir, 'trained')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"{name.lower().replace(' ', '_')}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    console.print(f"Saved model to: {filepath}")

def run_benchmark(name, model, X_train, y_train, X_test, y_test, epochs=100):
    
    """Train and evaluate a single model"""
    
    console.print(f"\n[Processing {name}")
    
    # 1. Train
    start_time = time.time()
    model.fit(X_train, y_train, verbose=True, epochs=epochs) 
    train_time = time.time() - start_time
    
    # 2. Predict
    y_pred = model.predict(X_test)
    
    # 3. Calculate Accuracy
    acc = np.mean(y_pred == y_test)
    
    return acc, train_time

def main():
    # --- 1. Load Data ---
    data_path = os.path.join(current_dir, 'data', 'mnist_data.npz')
    
    if not os.path.exists(data_path):
        console.print(f"File not found: {data_path}")
        return

    console.print("Loading dataset...")
    
    try:
        # Load raw data
        X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(data_path)
        
        train_ds = dataset("train", X_tr_raw, y_tr)
        test_ds = dataset("test", X_te_raw, y_te)
        
        console.print(f"Data loaded. Train size: {train_ds.data.shape}")

    except Exception as e:
        console.print(f"Error loading data: {e}")
        return

    # --- 2. Prepare Data ---
    X_train_norm = train_ds.data
    y_train = train_ds.label
    X_test_norm = test_ds.data
    y_test = test_ds.label
    
    # --- 3. Model Configuration ---
    model_configs = [
        ("Pixel Model", PixelSoftmax, {'learning_rate': 0.1}),
        ("Edge Model", EdgeSoftmax, {'learning_rate': 0.1}),
        # ("PCA Model", PCASoftmax, {'learning_rate': 0.1, 'n_components': 50}),
    ]

    # Result Table
    table = Table(title="Model Benchmark Results")
    table.add_column("Model Name", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Time (s)", style="green")

    # 
    num_features = X_train_norm.shape[1]
    num_classes = 10
    EPOCHS = 100

    # --- 4. Main Loop ---
    for name, ModelClass, kwargs in model_configs:

        # Init Model
        model = ModelClass(num_features=num_features, num_classes=num_classes, **kwargs)
        
        acc, duration = run_benchmark(name, model, X_train_norm, y_train, X_test_norm, y_test, epochs=EPOCHS)
        save_model(model, name)
        
        # Log Result
        table.add_row(name, f"{acc:.4f} ({acc*100:.1f}%)", f"{duration:.2f}")

    # --- 5. Final Report ---
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    main()