import matplotlib.pyplot as plt

from src.data.generate_data import generate_donut, generate_squares, generate_spirals
from src.utils.plot import plot_points_labels


def test_squares():
    plot_points_labels(*generate_squares(500))


def test_donut():
    plot_points_labels(*generate_donut(500, 0.5, 20))


def test_spirals():
    plot_points_labels(*generate_spirals(500, 1, 0.03))


