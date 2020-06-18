import os
import torch
from torchvision.transforms import LinearTransformation
import torchvision
from helper import rescale


class Whitening(object):
    def __init__(self, data, path: str = None, mean_name: str = "mean.pt", whitening_name: str = "whitening_matrix.pt", save: bool = True):
        mean_path = os.path.join(path, mean_name)
        whitening_path = os.path.join(path, whitening_name)

        self.whitening_matrix: torch.Tensor = None
        self.mean_vector: torch.Tensor = None

        if not os.path.exists(mean_path):
            self.mean_vector = Whitening.calcMeanVec(data)
            if save:
                torch.save(self.mean_vector, mean_path)
        else:
            self.mean_vector = torch.load(mean_path)

        if not os.path.exists(whitening_path):
            self.whitening_matrix = Whitening.calcWhitening(data)
            if save:
                torch.save(self.whitening_matrix, whitening_path)
        else:
            self.whitening_matrix = torch.load(whitening_path)

        self.transform = LinearTransformation(
            self.whitening_matrix, self.mean_vector)

    def __call__(self, tensor: torch.Tensor):
        return self.transform(tensor)

    @staticmethod
    def calcMeanVec(data) -> torch.Tensor:
        # data is batched -> flatten it
        X = torch.cat([x for (x, _) in data], dim=0)
        # flatten data
        X = X.view(X.shape[0], -1)
        # calc mean vector
        X = X.mean(0)
        return X

    @staticmethod
    def calcWhitening(data) -> torch.Tensor:
        # data is batched -> flatten it
        X = torch.cat([x for (x, _) in data], dim=0)
        # flatten data
        X = X.view(X.shape[0], -1)
        # zero center data
        X = rescale(X, -1.0, 1.0)
        # calc covariance
        cov = X.t() @ X
        # calc singular value decomposition
        U, S, V = torch.svd(cov)
        return (U @ S.reciprocal().diag() @ V.t())
