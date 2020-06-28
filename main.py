from config import Config
from training import Training


def main():

    config = Config()
    print(config)

    tr = Training(
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        savepoint_dir=config.savepoint_dir,
        data_dir=config.data_dir,
        prep_dir=config.prep_dir,
        no_cuda=config.no_cuda,
        batch_size=config.batch_size,
        max_savepoints=config.max_savepoints,
        no_save_prep=config.no_save_prep,
        no_save_savepoints=config.no_save_savepoint
    )
    if config.train:
        tr.train(epochs=config.epochs, evaluate=config.evaluate,
                 print_per_batches=config.print_after_batch)
    if config.evaluate:
        tr.evaluate()


if __name__ == "__main__":
    main()
