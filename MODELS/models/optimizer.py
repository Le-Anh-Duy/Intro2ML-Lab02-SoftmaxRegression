import numpy as np

class Optimizer:
    def __init__(self, weight_decay: float=None):
        self.weight_decay = weight_decay

    def apply_grad(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def apply(self, weights: np.ndarray, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.weight_decay is not None:
            grad += self.weight_decay * weights
        update = self.apply_grad(grad)
        return weights - learning_rate * update
    
class Adam(Optimizer):
    def __init__(self, weight_decay=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def apply_grad(self, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.t += 1

        grad = np.clip(grad, -10.0, 10.0)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return m_hat / (np.sqrt(v_hat) + self.epsilon)
    
class GD(Optimizer):
    def __init__(self, weight_decay=None):
        super().__init__(weight_decay)

    def apply_grad(self, grad: np.ndarray) -> np.ndarray:
        return grad
    
class RMSProp(Optimizer):
    def __init__(self, weight_decay=None, beta=0.9, epsilon=1e-8):
        super().__init__(weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.v = None

    def apply_grad(self, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(grad)

        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        return grad / (np.sqrt(self.v) + self.epsilon)
    
class Adagrad(Optimizer):
    def __init__(self, weight_decay=None, epsilon=1e-8):
        super().__init__(weight_decay)
        self.epsilon = epsilon
        self.accumulated_grad = None

    def apply_grad(self, grad: np.ndarray) -> np.ndarray:
        if self.accumulated_grad is None:
            self.accumulated_grad = np.zeros_like(grad)

        self.accumulated_grad += grad ** 2
        return grad / (np.sqrt(self.accumulated_grad) + self.epsilon)