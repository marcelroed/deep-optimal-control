from typing import Tuple, Optional, Union
from numpy import prod
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tnrange, tqdm_notebook


# from src.settings import config


class MidPointNetwork(nn.Module):
    def __init__(self, layers: int, in_shape: Union[int, Tuple[int]], tolerance: float = 1e-6, max_iter: int = 10):
        super().__init__()
        self.features: int = int(prod(in_shape))
        self.tol = tolerance
        self.max_iter = max_iter

        # self.hs = nn.Parameter(torch.randn(layers - 1).uniform_(0.2, 1))
        self.hs = torch.ones(layers - 1)
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
            while iters == 0 or iters < self.max_iter:
                iters += 1
                y_cur = y_lastiter + self.hs[n] * torch.tanh(self.transforms[n]((y_cur + y_lastiter)/2))

                error_estimate = torch.norm(y_cur - y_prev)
                if error_estimate < self.tol:
                    break
                y_prev = y_cur

        return y_prev.log_softmax(dim=1)

    def train_network(self, train_dl: DataLoader, epochs: int = 1, lr: float = 0.005):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.NLLLoss()

        training_results = []
        for epoch in tnrange(epochs):
            for i, batch in enumerate(train_dl):
                x_batch, y_batch = batch
                y_batch = y_batch.long()

                # Reset optimizer
                optimizer.zero_grad()

                # Forward, backward then optimize
                outputs = self(x_batch)
                loss = loss_function(outputs, y_batch)

                loss.backward()
                optimizer.step()

                training_results.append(loss.item())

        return training_results

    def predict_loader(self, test_loader: DataLoader):
        results = []
        for batch in test_loader:
            results.append(self.predict(batch))
        return torch.cat(results, dim=0)

    def predict(self, batch):
        with torch.no_grad():
            return self.forward(batch)


if __name__ == '__main__':
    #config['device'] = 'cpu'
    torch.autograd.set_detect_anomaly(True)

    net = MidPointNetwork(5, 700)
    print(list(net.parameters()))
    inp, out = torch.randn(700), torch.randn(700)
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

