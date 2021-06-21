from nn.layers.layer import layer

import numpy as np

from functools import partial
from typing import Iterator, List, Tuple


class Dense(Layer):
    """Regular densely-connected Neural Network layer.
    """
    def __init__(self, n_inputs: int, n_outputs: int,
                 learning_rate: float = 0.001) -> None:
        """Dense implements the operation: output = dot(x, weights) + bias
        weights is a weights matrix created by the layer, and bias is a bias.
        """
        # Learning rate
        self.learning_rate: float = learning_rate
        # Weight of the layer
        self.weights: np.array = np.random.normal(loc=0.0, 
                                                  scale=np.sqrt(2 / (n_inputs +\
                                                                    n_outputs)), 
                                                  size=(n_inputs, n_outputs))
        # Bias
        self.biases: np.array = np.zeros(n_outputs)

        # Previous deltas
        self.dweights: np.array = np.zeros((n_inputs, n_outputs))
        # Bias
        self.dbiases: np.array = np.zeros(n_outputs)

    def forward(self, x: np.array) -> np.array:
        """Perform an affine transformation
        """
        return np.dot(x, self.weights) + self.biases

    def backward(self, x: np.array, grad_output: np.array) -> np.array:
        """Compute backpropagation of Dense Layer
        """
        # Compute df/dx = df/ddense * ddense /dx
        # ddense/dx = weights transposed
        grad_x: np.array = np.dot(grad_output, self.weights.T)

        # Compute gradient weights and biases
        grad_weights: np.array = np.dot(x.T, grad_output)
        grad_biases: np.array = grad_output.mean(axis=0) * x.shape[0]

        # Regularized delta rule
        self.dweights = self.learning_rate * (grad_weights - self.dweights)
        self.dbiases = self.learning_rate * (grad_biases - self.dbiases)

        self.weights -= self.dweights
        self.biases -= self.dbiases
        
        return grad_x
