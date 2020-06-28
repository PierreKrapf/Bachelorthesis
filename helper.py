import torchvision.transforms as transforms
import re
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy


def filenameToLabel(file_name):
    """
    Args:
        file_name (string)

    Returns:
        string | None
    """
    result = re.search("^\d+-\d+-(\w+).png$", file_name)
    return result[1] if result else None


def isPng(file_name):
    return file_name[-4:].lower() == ".png"


def printNamespace(namespace):
    for key in namespace.__dict__:
        if namespace.__dict__[key] is not None:
            print(f"{key}, {namespace.__dict__[key]}")


def rescale(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    assert(low < high), "[rescale] low={0} must be smaller than high={1}".format(
        low, high)
    old_width = torch.max(t)-torch.min(t)
    old_center = torch.min(t) + (old_width / 2.)
    new_width = float(high-low)
    new_center = low + (new_width / 2.)
    # shift everything back to zero:
    t = t - old_center
    # rescale to correct width:
    t = t * (new_width / old_width)
    # shift everything to the new center:
    t = t + new_center
    return t


def calcWhitenMatrix(data: torch.Tensor) -> torch.Tensor:
    # ensure zero centered data
    data = rescale(data, low=-1.0, high=1.0)
    # calc cov
    # cov = 1 / data.shape[0] * data @ data.t()
    cov = data.t() @ data
    # perform SVD
    U, S, V = torch.svd(cov)
    whiten_mat = (U @ S.reciprocal().diag() @ V.t())
    return whiten_mat


def calcMeanVec(data: torch.Tensor) -> torch.Tensor:
    mean = data.mean(0)
    return mean


def take(n: int, iterable):
    items = []
    for (idx,  item) in enumerate(iterable, 1):
        items.append(item)
        if n == idx:
            break
    return items


def calcMeanStdWhitenMatrixMeanVec(dataset: Dataset, transform=None):
    print("Calculating mean, std, whiten_matrix, mean_vector ... this may take a while!")
    _transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]) if transform == None else transform

    _dataset = deepcopy(dataset)
    _dataset.transform = _transform

    dl = torch.utils.data.DataLoader(
        _dataset, shuffle=True, num_workers=0, batch_size=20)
    data = torch.cat([x for (x, _) in dl], dim=0)
    mean, std = data.mean().item(), data.std().item()
    normalize = transforms.Normalize((mean,), (std,))
    _dataset.transform = transforms.Compose([_transform, normalize])
    data = torch.cat([x for (x, _) in dl], dim=0)
    data = data.view(data.shape[0], -1)
    mean_vec = calcMeanVec(data)
    whiten_matrix = calcWhitenMatrix(data)
    return (mean, std, whiten_matrix, mean_vec)


if __name__ == "__main__":
    from torchvision.datasets import ImageFolder
    ds = ImageFolder("data/rico")
    mean, std, whiten_matrix, mean_vec = calcMeanStdWhitenMatrixMeanVec(ds)
    print(mean, std, whiten_matrix.shape, mean_vec.shape)
