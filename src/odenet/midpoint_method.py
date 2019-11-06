from typing import Tuple, Optional, Union
from numpy import prod
import torch
import torch.nn as nn
from src.settings import *

from src.settings import config


class MidPointNetwork(nn.Module):
    def __init__(self, layers: int, in_shape: Union[int, Tuple[int]], tolerance: float = 1e-1, max_iter: int = 100):
        super().__init__()
        self.features: int = int(prod(in_shape))
        self.tol = tolerance
        self.max_iter = max_iter

        self.hs = nn.Parameter(torch.randn(layers - 1).uniform_(0.2, 1))
        print('hs:', self.hs)
        self.transforms = nn.ModuleList([nn.Linear(self.features, self.features) for _ in range(layers - 1)])

    def forward(self, x):
        # Iteratively solve for y_{n + 1}
        y_prev: torch.Tensor = x

        for n in range(len(self.transforms)):
            # Use the previous decided value as initial value
            y_lastiter: torch.Tensor = y_prev
            y_cur: torch.Tensor = y_prev

            # Iterate
            iters = 0
            while iters == 0 or (torch.norm(y_cur - y_prev) > self.tol and iters < self.max_iter):
                y_cur = y_prev + self.hs[n] * self.transforms[n]((y_prev + y_cur)/2)
                iters += 1
            print(iters)
            y_prev = y_cur

        return y_prev


if __name__ == '__main__':
    config['device'] = 'cpu'
    torch.autograd.set_detect_anomaly(True)

    net = MidPointNetwork(5, 2)
    print(list(net.parameters()))
    inp, out = torch.Tensor([1., 2.]).reshape((1, 2)), torch.Tensor([1., 2.]).reshape((1, 2))
    result = net.forward(inp)
    print(result)
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=1)
    optim.zero_grad()
    loss = loss_func(result, out)
    print(loss)
    loss.backward()
    optim.step()
    print(list(net.parameters()))

