import numpy as np

class Optimizer:
    def __init__(self):
        pass

    def __call__(self, grads):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def __call__(self, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return m_hat / (np.sqrt(v_hat) + self.epsilon)
    
class SGD(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self, grad: np.ndarray) -> np.ndarray:
        return grad