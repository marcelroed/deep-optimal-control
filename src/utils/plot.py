import matplotlib.pyplot as plt


def plot_points_labels(points, labels):
    plt.scatter(points[:, 0][labels], points[:, 1][labels], color='red')
    plt.scatter(points[:, 0][~labels], points[:, 1][~labels], color='blue')
    plt.gca().set_aspect(1)
    plt.show()
