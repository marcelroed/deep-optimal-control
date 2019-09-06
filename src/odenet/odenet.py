from itertools import product
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from src.settings import *
import numpy as np
from typing import *


class ODENet(nn.Module):
    """
    ODE inspired neural network architecture.
    Note that the shape of the input is the same as the output for every layer.
    """
    def __init__(self, input_shape: Tuple[int], layers: int, rk_method: str = 'euler'):
        """
        Args:
            input_shape (Tuple[int]): Shape of the input vector
            layers (int): The amount of dense layers to use
        """
        super(ODENet, self).__init__()
        self.delta_t = nn.Parameter(torch.full((layers, ), 1))
        self.input_shape = input_shape
        self.input_size: int = np.prod(input_shape)
        self.layers = [nn.Linear(self.input_size, self.input_size) for i in range(5)] + [nn.Linear(self.input_size, 1)]

        for i, layer in enumerate(self.layers):
            setattr(self, f'{layer.__class__.__name__}{i}', layer)

    def forward(self, x: torch.Tensor, final_activation=True):
        """
        Euler's method (ResNet with stride 1) for now.

        Args:
            x (torch.Tensor): Input tensor
            final_activation (bool): Whether or not to collapse the output to the final probability.
        Returns:
            (torch.Tensor): Output tensor
        """
        x = torch.flatten(x)
        for delta_t, layer in zip(self.delta_t, self.layers):
            x = x + delta_t * torch.sigmoid(layer(x))
        x = self.layers[-1](x)
        if not final_activation:
            return x
        return torch.sigmoid(x).view([])

    def train_network(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1, lr: float = 0.001, batch_size: int = 4):
        """
        Minimize loss function over weights of the network.

        Args:
            x (torch.Tensor): Inputs
            y (torch.Tensor): Outputs
            epochs (int): Number of training iterations
            lr (float): The learning rate of the

        Returns:
            None
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device \'{device}\' for training')
        self.to(device)
        y = y.float()

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        training_results = []
        for epoch in trange(epochs):
            for i in range(x.size()[1]):
                # Unpack data
                input, label = x[:, i], y[i]

                # Reset optimizer
                optimizer.zero_grad()

                # Forward, backward then optimize
                outputs = self(input)
                loss = loss_function(outputs, label)
                loss.backward()
                optimizer.step()

                training_results.append(loss.item())
                if not (i + 1) % 100:
                    #print(training_results[-1])
                    pass




