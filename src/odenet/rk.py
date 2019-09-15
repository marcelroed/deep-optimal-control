import torch
import torch.nn as nn
import torch.nn.functional as F

methods = ('euler', )

class RKLayer(nn.Module):
    def __init__(self, in_out_size: int, activation, method='euler', interpolation='step'):
        super().__init__()
        assert method in methods, f'Method {method} is not in the list {methods}.'
        self.method = method
        self.in_out_size = in_out_size
        self.delta_t = nn.Parameter(torch.Tensor(1))

    def forward(self, x: torch.Tensor):
        # x = self.
        pass
