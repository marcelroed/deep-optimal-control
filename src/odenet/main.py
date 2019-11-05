import matplotlib.pyplot as plt
import numpy as np
import src.data.generate_data as gd
import src.odenet.odenet as odenet
import torch
# import src.settings
from torch.utils.data import DataLoader

if __name__ == '__main__':
    n_points, batch_size, num_workers = 100, 5, 2
    data, labels = gd.generate_squares(n_points)
    print(labels)
    net: odenet.EulerNet = odenet.EulerNet((2,), 5)
    net.train_network(data, labels, 150)
    transformed = torch.cat([net.forward(data[:, i], False) for i in range(data.size()[1])]).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    plt.scatter(transformed[labels], np.zeros(shape=transformed[labels].shape), color='red')
    plt.scatter(transformed[~labels], np.zeros(shape=transformed[~labels].shape), color='blue')
    plt.show()
