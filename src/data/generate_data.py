from abc import abstractmethod
from typing import *
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


def generate_trivial(n_points: int) -> np.array:
    points = torch.randn(n_points)
    labels = points < 0.3
    return points, labels


def generate_squares(n_points: int) -> np.array:
    points = torch.rand(2, n_points) * 2 - 1
    labels = (points[0] > 0) != (points[1] > 0)
    return points, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    n = 100
    points, labels = map(lambda x: getattr(x, 'numpy')(), generate_squares(n))
    plt.scatter(points[0][labels], points[1][labels], color='red')
    plt.scatter(points[0][~labels], points[1][~labels], color='blue')
    plt.show()
