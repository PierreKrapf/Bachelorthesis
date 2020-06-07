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
