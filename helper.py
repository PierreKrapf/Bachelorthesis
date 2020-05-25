import re
import torch
import numpy as np


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


def whiteningMatrix(img):
    flat = torch.flatten(img)
    sigma = torch.mm(flat.t(), flat) / flat.size(0)
    u, s, _ = np.linalg.scd(sigma.numpy())
    zca_epsilon = 1e-10
    d = torch.Tensor(np.diag(1. / np.sqrt(s + zca_epsilon)))
    u = torch.Tensor(u)
    principal_components = torch.mm(torch.mm(u, d), u.t())
    return principal_components


def printNamespace(namespace):
    for key in namespace.__dict__:
        if namespace.__dict__[key] is not None:
            print(f"{key}, {namespace.__dict__[key]}")
