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
