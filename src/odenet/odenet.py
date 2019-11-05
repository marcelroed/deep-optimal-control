from typing import *
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.odenet.rk import RKLayer


from src.settings import *


class EulerNet(nn.Module):
    """
    ODE inspired neural network architecture.
    Note that the shape of the input is the same as the output for every layer.
    """
    def __init__(self, input_shape: Union[int, Tuple[int, ...]], layers: int, reduce_to: Optional[Union[int, Tuple[int, ...]]] = None,
                 rk_method: str = 'euler'):
        """
        Args:
            input_shape (Tuple[int]): Shape of the input vector
            layers (int): The amount of dense layers to use
        """
        super(EulerNet, self).__init__()
        self.delta_t = nn.Parameter(torch.full((layers, ), 1))
        self.input_shape = input_shape
        self.input_size: int = np.prod(input_shape)
        self.layers = [nn.Linear(self.input_size, self.input_size) for _ in range(layers)]
        if reduce_to is not None:
            self.layers.append(nn.Linear(self.input_size, np.prod(reduce_to)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'{layer.__class__.__name__}{i}', layer)

    def forward(self, x: torch.Tensor, final_activation=True) -> torch.Tensor:
        """
        Euler's method (ResNet with stride 1) for now.

        Args:
            x (torch.Tensor): Input tensor
            final_activation (bool): Whether or not to collapse the output to the final probability.
        Returns:
            (torch.Tensor): Output tensor
        """
        x = torch.flatten(x, 1)
        for delta_t, layer in zip(self.delta_t, self.layers[:-1]):
            x = x + delta_t * torch.relu(layer(x))
        x = self.layers[-1](x)
        return x.log_softmax(dim=1)

    def train_network(self, train_loader: DataLoader, epochs: int = 1, lr: float = 0.001, batch_size: int = 4):
        """
        Minimize loss function over weights of the network.

        Args:
            train_loader (DataLoader): DataLoader for the training data
            epochs (int): Number of training iterations
            lr (float): The learning rate of the

        Returns:
            None
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device \'{device}\' for training')
        self.to(device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        training_results = []
        for epoch in trange(epochs):
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = map(lambda x: x.to(device), batch)
                # Reset optimizer
                optimizer.zero_grad()

                # Forward, backward then optimize
                # print(x_batch.shape, y_batch.shape)
                outputs = self(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()

                training_results.append(loss.item())
        return training_results

    def evaluate(self, test_loader: DataLoader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        test_cases = len(test_loader)
        correct_predictions = 0
        with torch.no_grad():
            for x, y in test_loader:
                prediction_values = self.forward(x.to(device))
                prediction = np.argmax(prediction_values.cpu().numpy())
                if prediction == y:
                    correct_predictions += 1
            accuracy = correct_predictions / test_cases
            return accuracy


class ODENet(nn.Module):
    """
    ODE inspired neural network architecture.
    Note that the shape of the input is the same as the output for every layer.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: int, reduce_to: Optional[Union[int, Tuple[int, ...]]] = None,
                 rk_method: str = 'euler'):
        """
        Args:
            input_shape (Tuple[int]): Shape of the input vector
            layers (int): The amount of dense layers to use
        """
        super(ODENet, self).__init__()
        self.input_shape = input_shape
        self.input_size: int = np.prod(input_shape)
        self.layers = [RKLayer(self.input_size, method='euler')]
        if reduce_to is not None:
            self.layers.append(nn.Linear(self.input_size, np.prod(reduce_to)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'{layer.__class__.__name__}{i}', layer)
        self.activation = torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Uses any RK method

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            (torch.Tensor): Output tensor
        """
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.log_softmax(dim=1)

    def train_network(self, train_loader: DataLoader, epochs: int = 1, lr: float = 0.001, batch_size: int = 4):
        """
        Minimize loss function over weights of the network.

        Args:
            train_loader (DataLoader): DataLoader for the training data
            epochs (int): Number of training iterations
            lr (float): The learning rate of the

        Returns:
            None
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device \'{device}\' for training')
        self.to(device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        training_results = []
        for epoch in trange(epochs):
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = map(lambda x: x.to(device), batch)
                # Reset optimizer
                optimizer.zero_grad()

                # Forward, backward then optimize
                # print(x_batch.shape, y_batch.shape)
                outputs = self(x_batch)
                loss = loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()

                training_results.append(loss.item())
        return training_results

    def evaluate(self, test_loader: DataLoader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        test_cases = len(test_loader)
        correct_predictions = 0
        with torch.no_grad():
            for x, y in test_loader:
                prediction_values = self.forward(x.to(device))
                prediction = np.argmax(prediction_values.cpu().numpy())
                if prediction == y:
                    correct_predictions += 1
            accuracy = correct_predictions / test_cases
            return accuracy

# def lagrange_loss(loss_func, output, target):
#     loss = loss_func(output, target) +
