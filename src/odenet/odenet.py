from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
from scipy.optimize import minimize
from typing import *


class ODENet(nn.Module):
    def __init__(self, input_shape: Tuple[int], layers: int, rk_method='euler'):
        """
        ODE inspired neural network architecture.
        Note that the shape of the input is the same as the output for every layer.

        Args:
            input_shape (Tuple[int]): Shape of the input vector
            layers (int): The amount of dense layers to use
        """
        super().__init__()
        self.delta_t = torch.full((layers, ), 1)
        self.input_shape = input_shape
        self.input_size: int = np.prod(input_shape)
        self.layers = [nn.Linear(self.input_size, self.input_size) for i in range(5)]

    def forward(self, x):
        x = torch.flatten(x)
        for delta_t, layer in zip(self.layers, self.delta_t):
            x = x + delta_t * F.sigmoid(layer(x))
        return x.reshape(self.input_shape)

    def train_network(self, x, y, epochs: int):
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)


if __name__ == '__main__':
    net: ODENet = ODENet((50,), 40)
    net.cuda(0)




