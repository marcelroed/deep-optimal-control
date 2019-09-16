import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

methods = {
    'euler': {
        'b': torch.tensor([1], dtype=torch.float),
        'c': torch.tensor([0], dtype=torch.float),
        'A': torch.tensor([0])
    },
    'rk4': {
        'b': torch.tensor([1/6, 1/3, 1/3, 1/6], dtype=torch.float),
        'c': torch.tensor([0, 1/2, 1/2, 1], dtype=torch.float),
        'A': torch.tensor([
            [0,   0,   0, 0],
            [1/2, 0,   0, 0],
            [0,   1/2, 0, 0],
            [0,   0,   1, 0]
        ], dtype=torch.float)
    }
}


class RKLayer(nn.Module):
    def __init__(self, in_out_size: int, activation, method='euler', interpolation='step'):
        super().__init__()
        assert method in methods.keys(), f'Method {method} is not in {tuple(methods.keys())}.'
        self.method = methods[method]
        self.activation = activation
        self.in_out_size = in_out_size
        self.transform = nn.Linear(in_out_size, in_out_size)
        self.delta_t = nn.Parameter(torch.Tensor(1))

    def forward(self, x: torch.Tensor):
        A, b, c = [self.method[key] for key in ('A', 'b', 'c')]
        # TODO: Interpolate f for x_n + c_ih
        k = torch.zeros(c.shape)
        for i in range(k.shape[0]):
            k[i] = self.delta_t * self.transform(x + k[:i] * A[i, :i])

        return self.activation(x + b * k)
