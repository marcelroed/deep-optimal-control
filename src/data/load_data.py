from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms


def get_mnist(batch_size_train: int, batch_size_test: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the MNIST dataset shuffled and batched.

    Args:
        batch_size_train (int): Batch size for training set
        batch_size_test (int): Batch size for test set

    Returns: (training set, test set)

    """
    train = DataLoader(MNIST('../../data', train=True, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                       batch_size=batch_size_train, shuffle=True)
    test = DataLoader(MNIST('../../data', train=False, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                       batch_size=batch_size_test, shuffle=True)

    return train, test


