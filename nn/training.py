# Imports
import matplotlib.pyplot as plt
import torch


###-------------------------------PARAMETERS-------------------------------###
TRAINING_SAMPLES_NUMBER: int = 8000
TEST_SAMPLES_NUMBER: int = 2000

IMAGE_DIM: tuple = (10, 10)
N_HIDDEN: int = 50
N_OUTPUTS: int = 1

LEARNING_RATE: float = 0.05
EPOCHS: int = 10


###----------------------------OTHER PARAMETERS----------------------------###
# Select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


###----------------------------GENERATE DATASET----------------------------###
def generate_dataset(size=(10, 10), n_samples: int = 1000,
                     device=torch.device('cpu')) -> tuple:
    """
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
rnd_samples: torch.Tensor = torch.randint(0, TRAINING_SAMPLES_NUMBER, (15,))

for i, rnd_sample in enumerate(rnd_samples):
    plt.subplot(3, 5, i + 1)
    plt.title(
        'Imagen Clara' if training_labels[rnd_sample] == 1 else 'Imagen Oscura')
    plt.imshow(training_imgs[rnd_sample], cmap='gray')

plt.tight_layout(pad=1)
plt.show()
