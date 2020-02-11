import torch
import torch.nn as nn
from src.settings import *

from src.settings import config


methods = {
    'euler': {
        'b': torch.tensor([1], dtype=torch.float),
        'c': torch.tensor([0], dtype=torch.float),
        'A': torch.tensor([0], dtype=torch.float)
    },
    'rk4': {
        'b': torch.tensor([1/6, 1/3, 1/3, 1/6]),
        'c': torch.tensor([0, 1/2, 1/2, 1]),
        'A': torch.tensor([
            [0,   0,   0, 0],
            [1/2, 0,   0, 0],
            [0,   1/2, 0, 0],
            [0,   0,   1, 0]
        ])
    }
}


class RKLayer(nn.Module):
    def __init__(self, in_out_size: int, method='euler', interpolation='step'):
        super().__init__()
        assert method in methods.keys(), f'Method {method} is not in {tuple(methods.keys())}.'
        self.method = methods[method]
        self.in_out_size = in_out_size
        self.transform = nn.Linear(in_out_size, in_out_size)
        self.delta_t = nn.Parameter(torch.randn(1).uniform_(0.2, 1))

    def forward(self, x: torch.Tensor):
        A, b, c = [self.method[key].to(config['device']) for key in ('A', 'b', 'c')]
        # TODO: Interpolate f for x_n + c_ih
        ks = [None for _ in range(c.shape[0])]

        for i in range(len(ks)):
            ks[i] = self.delta_t * self.transform(x + sum([A[i, j] * ks[j] for j in range(i)]))
        return x + sum(bi*ki for bi, ki in zip(b, ks))


if __name__ == '__main__':
    config['device'] = 'cpu'
    torch.autograd.set_detect_anomaly(True)
    layer = RKLayer(2, 'rk4')
    print(list(layer.parameters()))
    inp, out = torch.Tensor([1., 2.]).reshape((1, 2)), torch.Tensor([1., 2.]).reshape((1, 2))
    result = layer.forward(inp)
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.SGD(layer.parameters(), lr=1)
    optim.zero_grad()
    loss = loss_func(result, out)
    print(loss)
    loss.backward()
    optim.step()
    print(list(layer.parameters()))

