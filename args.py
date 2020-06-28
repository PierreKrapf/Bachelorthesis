import argparse
import os
import sys
import math
from constants import DEFAULTS


class Path(object):
    def __call__(self, raw):
        path = os.path.normpath(raw)
        if not os.path.isdir(path):
            print(
                f"{path} directory does not exist yet. It will be created if necessary!")
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


# Handles parsing and validation of args
class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        # hyper parameter
        self.add_argument("-lr", dest="learning_rate", default=DEFAULTS["LEARNING_RATE"],
                          type=FloatRange(0.0, 1.0),  help="Defines learning rate for training! Valid values: 0 to 1")
        self.add_argument("-mm", dest="momentum", default=DEFAULTS["MOMENTUM"],
                          type=FloatRange(0.0, 1.0),  help="Defines momentum for training! Valid values: 0 to 1")
        self.add_argument("-wd", dest="weight_decay", default=DEFAULTS["WEIGHT_DECAY"],
                          type=FloatRange(0.0, 1.0),  help="Defines weight decay for training! Valid values: 0 to 1")

        # other net config
        self.add_argument("-b", dest="batch_size", default=DEFAULTS["BATCH_SIZE"],
                          type=int,  help="Defines batch_size!")
        self.add_argument("--no-cuda", dest="no_cuda", default=DEFAULTS["NO_CUDA"], const=True, nargs='?',
                          help="Uses CPU if set otherwise defaults to GPU usage!")
        self.add_argument("-eps", dest="epochs", default=DEFAULTS["EPOCHS"],
                          type=IntRange(1, 100), help="Defines the number of epochs to run! Valid values: 1 to 100")
        self.add_argument("--train", dest="train", default=DEFAULTS["TRAIN"], const=True,
                          nargs='?', help="If flag is set, only training will be run! If both --train and --evaluate are omitted both are assumed to be true.")
        self.add_argument("--eval", dest="evaluate", default=DEFAULTS["EVALUATE"], const=True,
                          nargs='?', help="If flag is set, only evaluation will be run! If both --train and --evaluate are omitted both are assumed to be true.")

        # logging
        self.add_argument("-pp", dest="print_after_batch", nargs="?", default=DEFAULTS["PRINT_AFTER_BATCH"],
                          type=IntRange(0, math.inf),  help="Print lost after x batches!")

        # saving & loading
        self.add_argument("--root-dir", dest="root_dir", default=DEFAULTS["ROOT_DIR"],
                          type=Path(), help="Defines root dir. All other paths will be children of this.")
        self.add_argument("--no-save", dest="no_save", default=DEFAULTS["NO_SAVE"], const=True,
                          type=bool, nargs='?', help="If set program will not save anything!")

        # ...savepoints
        self.add_argument("--savepoint-dir", dest="savepoint_dir",
                          default=DEFAULTS["SAVEPOINT_DIR"], type=Path(), help="Defines which directory savepoints are saved to!")
        self.add_argument("--no-save-sp", dest="no_save_savepoint", default=DEFAULTS["NO_SAVE_SAVEPOINT"],
                          const=True, nargs="?", help="If set no savepoints will be created during training!")
        self.add_argument("--max-savepoints", dest="max_savepoints", default=DEFAULTS["MAX_SAVEPOINTS"], type=IntRange(1, 100),
                          help="Maxmimum number of savepoints to keep!(1-100) Older savepoints will be deleted in favor of newer ones!")

        # ...prepared values (whitening)
        self.add_argument("--prep-dir", dest="prep_dir", default=DEFAULTS["PREP_DIR"], type=Path(),
                          help="Defines directory where data for whitening will be saved!")
        self.add_argument("--no-save-prep", dest="no_save_prep", type=bool, default=DEFAULTS["NO_SAVE_PREP"],
                          const=True, nargs="?", help="If set calculated values for whitening will not be saved!")

        # ...data
        self.add_argument("-data", "--data-dir", dest="data_dir", default=DEFAULTS["DATA_DIR"],
                          type=Path(), help="Defines where to load data from!")
