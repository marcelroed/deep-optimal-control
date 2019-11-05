from typing import *
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms


def get_mnist(data_loader: bool = False) -> Tuple[Dataset, Dataset]:
    """
    Returns the MNIST dataset shuffled and batched.

    Args:
        batch_size_train (int): Batch size for training set
        batch_size_test (int): Batch size for test set

    Returns: (training set, test set)

    """
    train = MNIST('../../data', train=True, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         # torchvision.transforms.Normalize((0.1307,), (0.3081,))
         ]))

    test = MNIST('../../data', train=False, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         # torchvision.transforms.Normalize((0.1307,), (0.3081,))
         ]))

    return train, test


