from torch import Tensor
from torchvision.transforms import LinearTransformation
from helper import rescale


class Rescale(object):
    def __init__(self, low: float, high: float):
        assert low < high, f"low({low}) must be smaller than high({high}) in Rescale Transform"
        self.low = low
        self.high = high

    def __call__(self, tensor: Tensor):
        return rescale(tensor, self.low, self.high)
