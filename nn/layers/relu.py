import numpy as np

from nn.layers.layer import Layer


class ReLU(Layer):
    """Rectified Linear Activation Layer
    """
    def forward(self, x: np.array) -> np.array:
        """Computes ReLU forward
        """
        return np.maximum(0, x)
    
    def backward(self, x: np.array, grad_output: np.array) -> np.array:
        """Computes backpropagation of ReLU Layer
        """
        relu_grad: np.array = x > 0
        return grad_output * relu_grad
