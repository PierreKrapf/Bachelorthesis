from args import ArgParser
from inspect import getmembers, ismethod
import os
from constants import DEFAULTS


class Config(object):

    def __init__(self):
        super().__init__()
        # initial hyperparameters
        self.learning_rate: float = DEFAULTS["MOMENTUM"]
        self.momentum: float = DEFAULTS["LEARNING_RATE"]
        self.weight_decay: float = DEFAULTS["WEIGHT_DECAY"]

        self.print_after_batch: int = DEFAULTS["PRINT_AFTER_BATCH"]
        self.batch_size: int = DEFAULTS["BATCH_SIZE"]
        self.epochs: int = DEFAULTS["EPOCHS"]
        self.no_cuda: bool = DEFAULTS["NO_CUDA"]

        # Global save
        self.no_save: bool = DEFAULTS["NO_SAVE"]
        self.root_dir: str = DEFAULTS["ROOT_DIR"]

        self.no_save_savepoint: bool = DEFAULTS["NO_SAVE_SAVEPOINT"]
        self.savepoint_dir: str = DEFAULTS["SAVEPOINT_DIR"]
        # Maximum savepoints at any given time
        # Oldest savepoint will be deleted if limit is reached
        self.max_savepoints: int = DEFAULTS["MAX_SAVEPOINTS"]

        # Should whitening matrix, mean_vector not be saved for whitening?
        self.no_save_prep: bool = DEFAULTS["NO_SAVE_PREP"]
        self.prep_dir: str = DEFAULTS["PREP_DIR"]

        # Where should data be loaded from?
        self.data_dir: str = DEFAULTS["DATA_DIR"]

        # Should the net train?
        self.train: bool = DEFAULTS["TRAIN"]
        # Should the net perform and log evaluations?
        self.evaluate: bool = DEFAULTS["EVALUATE"]

        parsed = ArgParser().parse_args().__dict__.items()
        self_keys = self.__dict__.keys()
        for (key, value) in parsed:
            assert key in self_keys, f"Key {key} from parser not found in Config object!{self_keys}"
            self[key] = value if value else self[key]

        # Having both values False makes no sense (this is the default too)
        # It is assumed that values have been omitted and both are set to True
        if not self.train and not self.evaluate:
            self.train = True
            self.evaluate = True

        # join paths with root path
        self.data_dir = os.path.join(self.root_dir, self.data_dir)
        self.prep_dir = os.path.join(self.root_dir, self.prep_dir)
        self.savepoint_dir = os.path.join(self.root_dir, self.savepoint_dir)

        # apply global save option if neccessary
        if self.no_save:
            self.no_save_prep = True
            self.no_save_savepoint = True

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        text = "Config: {\n"
        for (key, value) in self.__dict__.items():
            text += f"\t{key} := {value}\n"
        text += "}"
        return text


# FOR DEBUGGING
if __name__ == "__main__":
    config = Config()
    print(config)
