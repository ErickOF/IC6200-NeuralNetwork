import numpy as np


class Layer:
    """Abstract class for Layers
    """
    def forward(self, x: np.array) -> np.array:
        """Takes input data and returns output data
        """
        pass

    def backward(self, x: np.array, grad_output: np.array) -> np.array:
        """Performs a backpropagation step through the layer
        """
        num_units: int = x.shape[1]
        dlayer_dx: np.array = np.eye(num_units)

        # Chain rule
        return np.dot(grad_output, dlayer_dx)
