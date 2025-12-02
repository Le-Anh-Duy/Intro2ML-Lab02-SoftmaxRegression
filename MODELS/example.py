from models.utils.loader import *
from models.utils.visualize import *
from models.metrics import *
from models.softmax_regression import *
from models.model_pixel import PixelSoftmax
from models.model_edge import EdgeSoftmax

X_train, y_train, X_test, y_test = load_dataset('data/mnist_data.npz')

model = PixelSoftmax(X_train[0].size, 10)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train, learning_rate=0.1, epochs=100)

y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

print(f"Accuracy: {accuracy(y_pred, y_test)}")

plt.figure(figsize=(6,6))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.axis('off')
    plt.title(f'label: {y_test[i]}, prediction: {y_pred[i]}', fontsize=8)

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, num_classes=10)

plot_confusion_matrix(cm, range(10))