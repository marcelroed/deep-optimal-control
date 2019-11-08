from typing import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from math import pi


def to_dataset(data: Tuple[torch.Tensor, torch.Tensor]):
    return TensorDataset(data[0], data[1].type(dtype=torch.long))


def _rotate_points(points: torch.Tensor, total_swirls: float):
    rs = (points[:, 0].pow(2) + points[:, 1].pow(2)).sqrt_()
    thetas = torch.atan2(points[:, 1], points[:, 0])
    print(thetas)
    d_thetas = rs * total_swirls * 2 * pi
    new_thetas = thetas + d_thetas
    return torch.stack((rs * torch.cos(new_thetas), rs * torch.sin(new_thetas))).transpose_(0, 1)


def _sample_binary(probs: torch.Tensor):
    indicators = torch.rand_like(probs)
    print(indicators)
    return indicators > probs


def generate_trivial(n_points: int):
    points = torch.randn(n_points)
    labels = points < 0.3
    return points, labels


def generate_squares(n_points: int):
    points = torch.rand(n_points, 2) * 2 - 1
    labels = (points[:, 0] > 0) != (points[:, 1] > 0)
    return points, labels


def generate_donut(n_points: int, transition_radius: float, hardness: float):
    thetas = torch.rand(n_points).mul_(2 * pi)
    rs = torch.rand(n_points).sqrt_()
    points = torch.stack((rs * torch.cos(thetas), rs * torch.sin(thetas))).transpose_(0, 1)
    labels = _sample_binary(torch.sigmoid((rs - transition_radius) * hardness))

    return points, labels


def generate_spirals(n_points: int, total_swirls: float, width_var: float):
    points = torch.rand(n_points).uniform_(-1, 1)
    labels = points > 0

    points = torch.stack((points, torch.zeros_like(points))).transpose_(0, 1)
    print(points.shape, labels.shape)
    points = _rotate_points(points, total_swirls)
    points += width_var * torch.randn_like(points)
    return points, labels


