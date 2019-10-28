import matplotlib.pyplot as plt
from src.odenet.odenet import ODENet
from src.data.load_data import get_mnist


def test_mnist():
    """Run MNIST through the ODE-net"""
    # Get MNIST data
    train, test = get_mnist(32, 32)

    print(train.data[0])

    net = ODENet()

