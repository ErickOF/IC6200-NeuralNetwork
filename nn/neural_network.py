from functools import partial
from typing import List

from nn.layers.layer import Layer
from nn.layers.dense import Dense
from nn.layers.relu import ReLU
from nn.loss.cross_entropy import *


class Network:
    """Neural Network
    """

    def __init__(self, layers: List[Layer]):
        """Constructor
        """
        self.network: List[Layer] = layers

    def forward(self, X: np.array) -> List[np.array]:
        """Compute activations of all network layers by applying them
        sequentially. Return a list of activations for each layer.
        """
        activations: List[np.array] = []
        x: np.array = X

        # Looping through each layer
        for layer in self.network:
            activations.append(layer.forward(x))
            # Updating input to last layer output
            x: np.array = activations[-1]

        return activations

    def load(self, file_name: str = '0') -> None:
        """Load network weights
        """
        # Modify allow_pickle parameter
        np_load_old = partial(np.load)
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        # Load weights
        loaded_weights: np.array = np.load(f'{file_name}_nnw.npy')
        # Index of the current weight
        i: int = 0

        for layer in self.network:
            if isinstance(layer, Dense):
                # Load weight into the layer
                layer.weights = loaded_weights[i]
                i += 1

    def predict(self, X: np.array) -> List[np.array]:
        """Compute network predictions. Returning indices of largest Logit
        probability
        """
        return self.forward(X)[-1].argmax(axis=-1)

    def save(self, file_name: str = '0') -> None:
        """Save network wieghts
        """
        np.save(f'{file_name}_nnw.npy',
                np.array([l.weights for l in self.network if isinstance(l,
                                                                        Dense)]))

    def train(self, X: np.array, y: np.array) -> np.array:
        """Train neural network
        """
        # Get the layer activations
        layer_activations: List[np.array] = self.forward(X)
        layer_inputs: List[List] = [X] + layer_activations
        logits: np.array = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss: np.array = softmax_crossentropy_with_logits(logits, y)
        loss_grad: np.array = grad_softmax_crossentropy_with_logits(logits, y)

        # Propagate gradients through the network
        for layer_index in range(len(self.network))[::-1]:
            layer: Layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

        return np.mean(loss)
