from models.utils.loader import *
from models.utils.visualize import *
from models.metrics import *
from models.softmax_regression import *

X_train, y_train, X_test, y_test = load_dataset('data/mnist_data.npz')

model = SoftmaxRegression(X_train[0].size, 10)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train, epochs=20)

y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

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