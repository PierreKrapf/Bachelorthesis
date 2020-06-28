import os

DEFAULTS = {
    "LEARNING_RATE": 0.1,
    "MOMENTUM": 0.0,
    "WEIGHT_DECAY": 0.0,
    "PRINT_AFTER_BATCH": 100,
    "BATCH_SIZE": 20,
    "EPOCHS": 10,
    "NO_CUDA": False,
    "NO_SAVE": False,
    "ROOT_DIR": os.path.curdir,
    "NO_SAVE_SAVEPOINT": False,
    "SAVEPOINT_DIR": "savepoints",
    "MAX_SAVEPOINTS": 10,
    "NO_SAVE_PREP": False,
    "PREP_DIR": "cache",
    "DATA_DIR": "data",
    "TRAIN": False,
    "EVALUATE": False
}
