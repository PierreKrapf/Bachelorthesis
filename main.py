import os
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm
from net import Net
from dataset import SimpleDataset
from training import Training
from args import ArgParser
from helper import printNamespace


def main():
    config = ArgParser().parse_args()
    printNamespace(config)
    config = config.__dict__
    tr = Training(savepoint_dir=config["savepoint_dir"], lr=config["learning_rate"], momentum=config["momentum"],
                  weight_decay=config["weight_decay"], no_cuda=config["no_cuda"], batch_size=config["batch_size"], print_per=config["print_per"])
    tr.run(epochs=config["epochs"])


if __name__ == "__main__":
    main()
