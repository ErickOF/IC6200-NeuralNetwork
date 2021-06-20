###--------------------------------IMPORTS---------------------------------###
import matplotlib.pyplot as plt
import torch


###-------------------------------PARAMETERS-------------------------------###

TRAINING_SAMPLES_NUMBER: int = 2000
TEST_SAMPLES_NUMBER: int = 2 * TRAINING_SAMPLES_NUMBER

IMAGE_DIM: tuple = (10, 10)
N_HIDDEN: int = 50
N_OUTPUTS: int = 1

LEARNING_RATE: float = 0.005
EPOCHS: int = 20


###----------------------------OTHER PARAMETERS----------------------------###

# Select device
device: torch.device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')


###------------------------------PROGRESS BAR------------------------------###

# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def updateProgressBar(i: int, total: int, prefix: str = '', suffix: str = '',
                      length: int = 100) -> None:
    percent: str = f'{100 * i / total}'
    filledLength: int = length * i // total
    bar: str = '█' * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')

    # Print new line on complete
    if i == total:
        print()


###----------------------------GENERATE DATASET----------------------------###

def generate_dataset(size=(10, 10), n_samples: int = 1000,
                     device: torch.device = torch.device('cpu')) -> tuple:
    """Generate a dataset

    Args:
        size (tuple, optional): images dimensions. Defaults to (10, 10).
        n_samples (int, optional): number of samples. Defaults to 1000.
        device (torch.device, optional): Device to be used (CPU or GPU).
                                         Defaults to torch.device('cpu').

    Returns:
        tuple: images and labels
    """
    # Generate samples
    samples: torch.Tensor = torch.randint(0, 2, (n_samples,) + size,
                                          dtype=torch.float,
                                          device=device)

    # Generate labels
    non_zeros: torch.Tensor = torch.count_nonzero(samples, dim=(1, 2))

    labels: torch.Tensor = torch.zeros(n_samples,
                                       dtype=torch.float,
                                       device=device)

    labels[non_zeros > (size[0] * size[1] // 2)] = 1

    return samples, labels


training_imgs, training_labels = generate_dataset(IMAGE_DIM,
                                                  TRAINING_SAMPLES_NUMBER)


###-------------------------SHOW SOME DATA SAMPLES-------------------------###


def showDataSamples(training_imgs: torch.Tensor,
                    training_labels: torch.Tensor) -> None:
    """Show data samples for training

    Args:
        training_imgs (torch.Tensor): images used for training
        training_labels (torch.Tensor): labels of the images used for training
    """
    rnd_samples: torch.Tensor = torch.randint(0, len(training_imgs), (15,))

    for i, rnd_sample in enumerate(rnd_samples):
        plt.subplot(3, 5, i + 1)
        plt.title(
            'Imagen Clara' if training_labels[rnd_sample] else 'Imagen Oscura')
        plt.imshow(training_imgs[rnd_sample], cmap='gray')

    plt.tight_layout(pad=1)
    plt.show()

# showDataSamples()

###-----------------------------LOSS FUNCTIONS-----------------------------###


class Loss:
    def mse(self, x: torch.Tensor) -> float:
        return x.root(2).sum().sqrt()[0]


###--------------------------ACTIVATION FUNCTIONS--------------------------###

def relu() -> None:
    return


def sigmoid() -> None:
    return


###-----------------------------NEURAL NETWORK-----------------------------###

class NeuralNetwork:
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int = 2,
                 learning_rate: float = 0.005,
                 device: torch.device = torch.device('cpu')):
        """Constructor

        Args:
            n_inputs (int): number of input neurons
            n_outputs (int): number of output neurons
            n_hidden (int, optional): number of hidden neurons. Defaults to 2.
            learning_rate (float, optional): learning rate. Defaults to 0.005.
            device (torch.device, optional): device to be used (CPU or GPU).
                                             Defaults to torch.device('cpu').
        """
        self._n_inputs: int = n_inputs
        self._n_outputs: int = n_outputs
        self._n_hidden: int = n_hidden
        self._learning_rate: float = learning_rate

        self._device: torch.device = device

        # Activation functions
        self._relu = torch.relu
        self._sigmoid = torch.sigmoid

        # Loss function
        self._loss = torch.nn.MSELoss()

        # Weights
        self._w_hidden: torch.Tensor = 2 * \
            torch.rand((n_inputs, n_hidden), device=device) - 1
        self._w_output: torch.Tensor = 2 * \
            torch.rand((n_hidden, n_outputs), device=device) - 1

        # Outputs of the layers
        self._out: list = []

        # Previous deltas
        self._delta1: torch.Tensor = torch.zeros((self._n_outputs,))
        self._delta2: torch.Tensor = torch.zeros((self._n_inputs, self._n_hidden))

    def _backpropagation(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Learn using generalized delta rule.

        Args:
            x (torch.Tensor): data
            y (torch.Tensor): label
        """
        # Hidden-Output
        delta: torch.Tensor = (self._out[-1] - y) * self._out[-1] *\
                              (1 - self._out[-1])
        self._delta1 = self._learning_rate * (delta *\
                        self._out[-3].reshape(self._w_output.shape) +\
                        self._delta1)
        self._w_output -= self._delta1
        # Input-Hidden
        delta2: torch.Tensor = delta * self._w_output
        self._delta2 = self._learning_rate * (delta2.mm(x.reshape((1, -1))).t() +\
                        self._delta2)
        self._w_hidden -= self._delta2

    def _feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Neural Network output

        Args:
            x (torch.Tensor): data

        Returns:
            torch.Tensor: prediction
        """
        self._out = [torch.matmul(x, self._w_hidden)]
        self._out.append(self._relu(self._out[-1]))
        self._out.append(torch.matmul(self._out[-1], self._w_output))
        self._out.append(self._sigmoid(self._out[-1]))

        return self._out[-1]

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1,
            save: bool = False, file_name: str = '0') -> None:
        """Train Neural Network

        Args:
            x (torch.Tensor): samples
            y (torch.Tensor): labels
            epochs (int, optional): number of epochs to train. Defaults to 1.
            save (bool, optional): True if training must be saved. Defaults to
                                   False.
            file_name (str, optional): name of the file to store the model
                                       weights. Defaults to '0'.
        """
        print('Training...')

        for epoch in range(epochs):
            print(f'Epochs: {epoch + 1}/{epochs}')

            # Show progress bar
            updateProgressBar(0, len(x), prefix='Progress',
                              suffix='Complete. Accuracy: 0. Loss: 1',
                              length=50)

            # History of loss and accuracy
            loss: list = []
            accuracy: list = []

            for i, (sample, label) in enumerate(zip(x, y)):
                # Make a guess
                prediction: torch.Tensor = self._feedforward(sample.flatten())

                # Compute loss and accuracy
                accuracy.append(1 - (label - prediction).abs())
                loss.append(self._loss(prediction, label.view(-1)))

                suffix: str = 'Complete. Accuracy: ' +\
                    f'{(sum(accuracy) / len(accuracy)).item()}. ' +\
                    f'Loss: {sum(loss) / len(loss)}'

                updateProgressBar(i + 1, len(x), prefix='Progress',
                                  suffix=suffix, length=50)

                # Learn
                self._backpropagation(sample.flatten(), label)

            # Save weights
            if save:
                torch.save([self._w_hidden, self._w_output], f'{file_name}.nnw')

            print()

        print()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Evaluate model

        Args:
            x (torch.Tensor): samples
            y (torch.Tensor): labels
        """
        print('Testing...')

        # History of loss and accuracy
        loss: list = []
        accuracy: list = []

        # Show progress bar
        updateProgressBar(0, len(x), prefix='Progress',
                          suffix='Complete. Accuracy: 0. Loss: 1',
                          length=50)

        for i, (sample, label) in enumerate(zip(x, y)):
            # Make a guess
            prediction: torch.Tensor = self._feedforward(sample.flatten())

            # Compute loss and accuracy
            accuracy.append(1 - (label - prediction).abs())
            loss.append(self._loss(prediction, label.view(-1)))

            suffix: str = 'Complete. Accuracy: ' +\
                f'{(sum(accuracy) / len(accuracy)).item()}. ' +\
                f'Loss: {sum(loss) / len(loss)}'

            updateProgressBar(i + 1, len(x), prefix='Progress', suffix=suffix,
                              length=50)

    def load(self, file_name: str = '0') -> None:
        """Load weights from file

        Args:
            file_name (str, optional): Name of the file. Defaults to '0'.
        """
        self._w_hidden, self._w_output = torch.load(f'{file_name}.nnw')


###---------------------------------TRAIN----------------------------------###

nn: NeuralNetwork = NeuralNetwork(n_inputs=IMAGE_DIM[0] * IMAGE_DIM[1],
                                  n_hidden=N_HIDDEN,
                                  n_outputs=N_OUTPUTS,
                                  learning_rate=LEARNING_RATE)

# Load previous training
nn.load()

# Train the neural network
nn.fit(training_imgs, training_labels, epochs=EPOCHS, save=True)


###----------------------------------TEST----------------------------------###

test_images, test_labels = generate_dataset(size=IMAGE_DIM,
                                            n_samples=TEST_SAMPLES_NUMBER,
                                            device=device)

nn.evaluate(test_images, test_labels)
