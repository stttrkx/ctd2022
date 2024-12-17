#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import argparse
import logging
import pprint
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from LightningModules.DNN import EdgeClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
pp = pprint.PrettyPrinter(indent=2)


def headline(message):
    buffer_len = (80 - len(message)) // 2 if len(message) < 80 else 0
    return "-" * buffer_len + " " + message + " " + "-" * buffer_len


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 3: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)

    common_configs = all_configs["common_configs"]
    dnn_configs = all_configs["model_configs"]

    logging.info(headline("a) Initialising model"))

    model = EdgeClassifier(dnn_configs)
    print(model)

    logging.info(headline("b) Running training"))

    # Experiment Tracking
    save_directory = common_configs["artifact_directory"]

    # From TrainTrack
    logger = None
    logger_choice = "wandb"

    if logger_choice == "wandb":
        logger = WandbLogger(
            save_dir=save_directory,
            project=common_configs["experiment_name"],
            id=common_configs["resume_id"],
        )

    elif logger_choice == "tb":
        logger = TensorBoardLogger(
            save_dir=save_directory,
            name=common_configs["experiment_name"],
            # version=model_config["resume_id"],
            version=None,
        )

    elif logger_choice is None:
        logger = None

    # Trainer
    trainer = pl.Trainer(
        gpus=common_configs["gpus"],
        max_epochs=1,  # dnn_configs["max_epochs"],
        logger=logger,
    )

    # Training
    trainer.fit(model)

    logging.info(headline("c) Saving model"))

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(
        os.path.join(save_directory, common_configs["experiment_name"] + ".ckpt")
    )

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    train(args.config)
