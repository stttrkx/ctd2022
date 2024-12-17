#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import math
import argparse
import logging
import pprint
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from LightningModules.DNN.Models.tune_network import EdgeClassifier_Tune

device = "cuda" if torch.cuda.is_available() else "cpu"
pp = pprint.PrettyPrinter(indent=2)


def headline(message):
    buffer_len = (80 - len(message)) // 2 if len(message) < 80 else 0
    return "-" * buffer_len + " " + message + " " + "-" * buffer_len


# Argument Parser
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("TuneASHA.py")
    add_arg = parser.add_argument
    parser.add_argument("--smoke-test", action="store_true", help="Quick Testing")
    parser.add_argument("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


# Trainable (Functional API)
def trainer_dense(combo_config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    """PyTorch-Lighting Trainer (RayTune Function API)"""

    # Model Config
    common_configs = combo_config["common_configs"]
    dnn_configs = combo_config["model_configs"]

    # Build Model
    logging.info(headline("a) Initialising model"))
    model = EdgeClassifier_Tune(dnn_configs)

    logging.info(headline("b) Running training"))

    # Loggers, Callbacks, etc
    save_directory = common_configs["artifact_directory"]
    experiment_dir = common_configs["experiment_name"]
    os.makedirs(experiment_dir, exist_ok=True)

    logger = TensorBoardLogger(save_directory, name=experiment_dir, version=None)

    metrics = {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"}
    callbacks = [
        TuneReportCallback(metrics, on="validation_end"),
        TuneReportCheckpointCallback(
            metrics, filename="checkpoint", on="validation_end"
        ),
    ]

    # Build Trainer
    kwargs = {
        "gpus": math.ceil(num_gpus),
        "max_epochs": num_epochs,
        "logger": logger,
        "enable_progress_bar": False,
        "callbacks": callbacks,
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    trainer = Trainer(**kwargs)

    # Run Trainer
    trainer.fit(model)


# Tuner :: ASHA Scheduler
def tuner_pbt(
    config_file="pipeline_config.yaml",
    num_samples=10,
    num_epochs=10,
    cpus_per_trial=1,
    gpus_per_trial=1,
):

    # (0) Model Config
    with open(config_file) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    # (1) Trainable
    trainable_w_param = tune.with_parameters(
        trainer_dense, num_epochs=num_epochs, num_gpus=gpus_per_trial
    )
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    # (2) Param Space
    tune_config = {
        "l1_size": tune.choice([128, 256, 512, 1024]),
        "l2_size": tune.choice([128, 256, 512, 1024]),
        "l3_size": tune.choice([128, 256, 512, 1024]),
        "l4_size": tune.choice([128, 256, 512, 1024]),
        # "l5_size": tune.choice([128, 256, 512, 1024]),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Combo Config
    # combine using unpacking operator '**', similar to update() but return a dict
    # combo_config = {**model_config,**tune_config}
    # combine using dictionary comprehension
    combo_config = {k: v for d in (model_config, tune_config) for k, v in d.items()}

    # (3) Tune Config
    scheduler = PopulationBasedTraining(  # PBT Trial Schedular
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": [32, 64, 128],
        },
    )

    # (4) Run Config
    reporter = CLIReporter(  # CLIReporter Callback
        parameter_columns=[
            "l1_size",
            "l2_size",
            "l3_size",
            "l4_size",
            # "l5_size",
            "batch_size",
        ],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    # (5) Init Tuner
    tuner = tune.Tuner(
        trainable=tune.with_resources(trainable_w_param, resources=resources_per_trial),
        param_space=combo_config,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            # search_alg=BayesOptSearch(),  # Optimization Algorithm, Default: Random Search
            scheduler=scheduler,  # Trial Schedular
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            local_dir=combo_config["common_configs"]["artifact_directory"],
            name="PBTResult",
            progress_reporter=reporter,
            # log_to_file=True,
            log_to_file=("my_stdout.log", "my_stderr.log"),
        ),
    )

    # (6) Fit Tuner
    result_grid = tuner.fit()

    # (7) Iterate over results
    for i, result in enumerate(result_grid):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a mean accuracy metric of:",
            result.metrics["mean_accuracy"],
        )

    # (8) Print Best Hyperparameters
    best_result = result_grid.get_best_result()
    print("\nBest Result: ", result_grid.get_best_result(metric="loss", mode="min"))
    print(
        "\nWorst Result: ",
        result_grid.get_best_result(metric="mean_accuracy", mode="min"),
    )
    print("\nBest Result Hyperprameters: ", best_result.config)
    print("\nBest Result Directory : ", best_result.log_dir)
    print("\nBest Result Meterics : ", best_result.metrics)


if __name__ == "__main__":
    args = parse_args()
    config = args.config

    if args.smoke_test:
        print("Running Smoke Test...")
        tuner_pbt(
            config, num_samples=1, num_epochs=1, cpus_per_trial=1, gpus_per_trial=0
        )
    else:
        tuner_pbt(
            config, num_samples=10, num_epochs=10, cpus_per_trial=32, gpus_per_trial=1
        )
