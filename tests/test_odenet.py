import matplotlib.pyplot as plt
import torch
from src.odenet.odenet import EulerNet
from src.odenet.rk import RKLayer
from src.data.load_data import get_mnist


def test_mnist():
    """Run MNIST through the ODE-net"""
    # Get MNIST data
    train, test = get_mnist(32, 32)

    print(train.data[0])

    net = EulerNet()


def test_rk_layers():
    l = RKLayer(2)
    print(l.parameters)


def test_delta_t_gradient():
    net = EulerNet(2, 2)
    inp = torch.Tensor([2., 2.])
    outp = torch.Tensor([2., 3.])
    print(net.delta_t)

    # Train for one step
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1)
    optimizer.zero_grad()
    outp = net.forward(inp)
    loss = loss_func(inp, outp)
    loss.backward()
    optimizer.step()

    print(net.delta_t)
