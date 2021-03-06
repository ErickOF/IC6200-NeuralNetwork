import numpy as np
import random
import tkinter as tk
from tqdm import trange
from typing import Iterator, List, Tuple

from nn.layers.layer import Layer
from nn.layers.dense import Dense
from nn.layers.relu import ReLU
from nn.neural_network import Network


# Defining some constants
TITLE: str = 'R&M'
# Dimensions
HEIGHT, WIDTH = 420, 530
# Background color
BG_COLOR: str = 'violet'
# Button color
BG_BTN: str = 'purple'
# Button text color
BTN_TEXT_COLOR: str = 'white'
# Text font
TEXT_FONT: str = 'Execute'
# Text size
TEXT_SIZE: int = 30
# Button text size
BTN_TEXT_SIZE: int = 12
# Board
grid: List[tk.Label] = []

# Board Color
grid_color: List[int] = []
# Gameboard dimensions 8x8
N, M = 10, 10

WHITE: int = 1
BLACK: int = 0
# Create a main window
window: tk.Tk = tk.Tk()

# Get screen dimensions
SCREEN_WIDTH: int = window.winfo_screenwidth()
SCREEN_HEIGHT: int = window.winfo_screenheight()

# Get new window position
WINDOW_POS_X: int = int((SCREEN_WIDTH / 2) - (WIDTH / 2))
WINDOW_POS_Y: int = int((SCREEN_HEIGHT / 2) - (HEIGHT / 2))

# Set title
window.title(TITLE)
# Set geometry
# WIDTHxHIGHT+X+Y
window.geometry(f'{WIDTH}x{HEIGHT}+{WINDOW_POS_X}+{WINDOW_POS_Y}')
# Set if it's resizable
window.resizable(width=False, height=False)
# Set background
window.config(bg=BG_COLOR)


def on_click(i: int, j: int, event) -> None:
    """Changes color of grid when pressed
    """
    color: str = grid[i][j].cget('bg')

    if color == 'black':
        color = 'white'
        grid_color[i][j] = WHITE
    else:
        color = 'black'
        grid_color[i][j] = BLACK

    grid[i][j].config(bg=color)
    # Create grid
    grid[i][j].grid(row=i, column=j)


# Init gameboard
grid = [[tk.Label(window, width=7, height=2, borderwidth=1, relief='solid')
         for j in range(M)] for i in range(N)]

grid_color = [[WHITE for j in range(M)] for i in range(N)]

# Bind color change to labels
for i, row in enumerate(grid):
    for j, column in enumerate(row):
        grid[i][j].bind('<Button-1>', lambda e, i=i, j=j: on_click(i, j, e))


def draw_grid(grid: List[tk.Label]) -> None:
    """This function displays gameboard.
    """
    # Color to start drawing
    color: str = 'white'

    for i in range(N):
        for j in range(M):
            # Change label bg
            grid[i][j].config(bg=color)
            # Create grid
            grid[i][j].grid(row=i, column=j)
            # Change color
            color = 'white'
        # Change color
        color = 'white'


# Draw grid
draw_grid(grid)

# Add label
lbl_result: tk.Label = tk.Label(width=12, height=2, borderwidth=0,
                                relief='solid', text='',
                                font=(TEXT_FONT, 18), bg='violet')
lbl_result.place(x=1, y=340)


def generate_dataset(size: tuple = (10, 10), n_samples: int = 1000,
                     flatten: bool = True) -> Tuple[np.array]:
    # Generate samples
    samples: np.array = np.random.randint(0, 2, (n_samples,) + size) / 1.0

    # Generate labels
    non_zeros: np.array = np.count_nonzero(samples, axis=(1, 2))
    labels: np.arry = np.zeros((n_samples,), dtype=np.uint8)
    labels[non_zeros > (size[0] * size[1] // 2)] = 1

    if flatten:
        samples = samples.reshape([samples.shape[0], -1])

    return samples, labels

X_train, y_train = generate_dataset((10, 10), 12000)

def iterate_minibatches(x: np.array, y: np.array,
                        batchsize: int) -> Iterator[Tuple[np.array]]:
    """Generate randomly mini bataches
    """
    # Randomize samples
    index: np.array = np.random.permutation(len(x))

    for start_idx in trange(0, len(x) - batchsize + 1, batchsize):
        extract: np.array = index[start_idx:start_idx + batchsize]

        yield x[extract], y[extract]

EPOCHS: int = 50

training_history: List = []
validation_history: List = []

# Create Neural Network
layers: List[Layer] = [Dense(100, 100),  # Input layer
                       ReLU(),          # ReLU layer
                       Dense(100, 50),  # Hidden layer
                       ReLU(),          # ReLU layer
                       Dense(50, 2)]    # Output layer

nn: Network = Network(layers)
nn.load('src/0')

for epoch in range(EPOCHS):
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32):
        nn.train(x_batch, y_batch)

    X_val, y_val = generate_dataset((10, 10), 250000)

    training_history.append(np.mean(nn.predict(X_train) == y_train))
    validation_history.append(np.mean(nn.predict(X_val) == y_val))

    print(f'Epoch: {epoch + 1}')
    print(f'Train accuracy: {training_history[-1]}')
    print(f'Validation accuracy: {validation_history[-1]}')

    if validation_history[-1] > 0.95:
        break

nn.save('1')


def run() -> None:
    """Executes the Neural Network
    """
    prediction: int = nn.predict(
        np.array(sum(grid_color, [])))

    if prediction == 1:
        lbl_result.config(text='Imagen Clara')
    else:
        lbl_result.config(text='Imagen Oscura')


# Button
btn_run: tk.Button = tk.Button(window, text='Predecir', bg=BG_BTN,
                               fg=BTN_TEXT_COLOR,
                               font=(TEXT_FONT, BTN_TEXT_SIZE),
                               activebackground=BTN_TEXT_COLOR,
                               activeforeground=BG_BTN,
                               command=run)  # Add command
btn_run.place(x=460, y=340)
