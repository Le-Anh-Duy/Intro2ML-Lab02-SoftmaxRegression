import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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