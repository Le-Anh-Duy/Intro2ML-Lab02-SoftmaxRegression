import matplotlib.pyplot as plt
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