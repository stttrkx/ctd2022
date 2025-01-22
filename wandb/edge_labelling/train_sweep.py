import os
import yaml
import wandb
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from your_model import YourLightningModule  # Import your model here


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config=None):
    """Main training function that will be called by W&B sweep agent."""
    with wandb.init(config=config, resume="allow"):  # Added resume="allow"
        # Get hyperparameters from W&B
        config = wandb.config

        # Initialize the model with sweep config
        model = YourLightningModule(
            learning_rate=config.learning_rate,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            # Add other hyperparameters as needed
        )

        # Initialize WandbLogger with resume support
        wandb_logger = WandbLogger(
            project=config.project_name,
            log_model=True,
            resume="allow",  # Added resume support
        )

        # Define callbacks with support for resuming
        checkpoint_dir = os.path.join("checkpoints", wandb.run.id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        # Check for existing checkpoints
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
            if checkpoints:
                latest_checkpoint = max(
                    [os.path.join(checkpoint_dir, ckpt) for ckpt in checkpoints],
                    key=os.path.getmtime,
                )
            else:
                latest_checkpoint = None
        else:
            latest_checkpoint = None

        # Initialize trainer with resume support
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            accelerator="auto",
            devices=1,
            deterministic=True,
            enable_checkpointing=True,
        )

        # Start or resume training
        if latest_checkpoint:
            trainer.fit(model, ckpt_path=latest_checkpoint)
        else:
            trainer.fit(model)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model configuration YAML file",
    )
    parser.add_argument(
        "--sweep_config",
        type=str,
        required=True,
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID to resume")
    parser.add_argument(
        "--count", type=int, default=None, help="Number of runs to execute"
    )
    args = parser.parse_args()

    # Load configurations
    model_config = load_config(args.model_config)
    sweep_config = load_config(args.sweep_config)

    if args.sweep_id:
        # Resume existing sweep
        sweep_id = args.sweep_id
        print(f"Resuming sweep: {sweep_id}")
    else:
        # Initialize new sweep
        sweep_id = wandb.sweep(sweep_config, project=model_config["project_name"])
        print(f"Created new sweep: {sweep_id}")

    # Start or resume the sweep agent
    wandb.agent(sweep_id, function=train, count=args.count)


if __name__ == "__main__":
    main()
