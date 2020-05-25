import argparse
import os
import sys
import math


class Path(object):
    def __call__(self, raw):
        path = os.path.normpath(raw)
        if not os.path.isdir(path):
            print(
                "Savepoint directory does not exist yet. It will be created if necessary!")
        return path


class FloatRange(object):
    def __init__(self, start=-float("inf"), end=float("inf")):
        self.start = start
        self.end = end

    def __contains__(self, other):
        return self.start <= other <= self.end

    def __call__(self, other):
        try:
            x = float(other)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%r not a floating-point literal" % (x,))

        if x not in self:
            raise argparse.ArgumentTypeError(
                "%r not in range [%f, %f]" % (x, self.start, self.end))
        return x


class IntRange(object):
    def __init__(self, start=-math.inf, end=math.inf):
        self.start = start
        self.end = end

    def __contains__(self, other):
        return self.start <= other <= self.end

    def __call__(self, other):
        try:
            x = int(other)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a integer literal" % (x,))

        if x not in self:
            raise argparse.ArgumentTypeError(
                "%r not in range [%d, %d]" % (x, self.start, self.end))
        return x


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument("-lr", dest="learning_rate", default=0.0001,
                          type=FloatRange(0.0, 1.0),  help="Defines learning rate for training! Valid values: 0 to 1")
        self.add_argument("-mm", dest="momentum", default=0.0,
                          type=FloatRange(0.0, 1.0),  help="Defines momentum for training! Valid values: 0 to 1")
        self.add_argument("-wd", dest="weight_decay", default=0.0,
                          type=FloatRange(0.0, 1.0),  help="Defines weight decay for training! Valid values: 0 to 1")
        self.add_argument("-pp", dest="print_per", default=2000,
                          type=int,  help="Print lost after x batches!")
        self.add_argument("-b", dest="batch_size", default=10,
                          type=int,  help="Defines batch_size!")
        self.add_argument("--no-cuda", dest="no_cuda", default=False, const=True, nargs='?',
                          help="Uses CPU if set otherwise defaults to GPU usage!")
        self.add_argument("-sp", dest="savepoint_dir",
                          default=os.path.join("./data", "savepoints"), type=Path(), help="Defines which directory savepoints are saved to!")
        self.add_argument("-e", dest="epochs", default=10,
                          type=IntRange(1, 100), help="Defines the number of epochs to run! Valid values: 1 to 100")
