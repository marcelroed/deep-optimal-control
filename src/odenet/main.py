import matplotlib.pyplot as plt
import numpy as np
import src.data.generate_data as gd
import src.odenet.odenet as odenet
import torch
import src.settings
from torch.utils.data import DataLoader


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    n_points, batch_size, num_workers = 100, 5, 2
    data, labels = gd.generate_squares(n_points)
    print(labels)
    net: odenet.ODENet = odenet.ODENet((2,), 5)
    net.train_network(data, labels, 1000)
