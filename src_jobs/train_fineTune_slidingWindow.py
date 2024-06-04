import pickle
import warnings
from datetime import datetime
from itertools import repeat
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from artifact import Saw_centered
from data import CachedArtifactDataset, CenteredArtifactDataset
from mask_detector import WindowTransformerDetector
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from sliding_window_detector import FineTunedSlidingWindowDetector
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from utilities import parameters_k

# stop warnings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

width = 512
loss = "label"
loss_boost_fp = 0.5
artifact = Saw_centered()
batch_size = 32  # 'values': [32, 64, 128]
warmup = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_datasets = [
    "australian_electricity_demand_dataset",
    "electricity_hourly_dataset",
    "electricity_load_diagrams",
    "HouseholdPowerConsumption1",
    # "HouseholdPowerConsumption2",
    # "london_smart_meters_dataset_without_missing_values",
    "solar_10_minutes_dataset",
    "wind_farms_minutely_dataset_without_missing_values",
]


def load_series(names: list[str], split: str, path: str):
    series: list[np.ndarray] = list()
    counts: list[float] = list()
    for name in names:
        with open(f"{path}/{name}_{split}.pickle", "rb") as f:
            raw = [a for a in pickle.load(f) if len(a) > width]
            series.extend(np.array(a).astype(np.float32) for a in raw)
            counts.extend(repeat(1 / len(raw), len(raw)))
    counts = np.array(counts)
    return series, np.divide(counts, np.sum(counts))


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")
    ]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


# Run it:
def main(
    input_path: Path = typer.Option(default=...),
    val_path: Path = typer.Option(default=...),
    model_path: Path = typer.Option(default=...),
    output_path: Path = typer.Option(default=...),
):
    """
    Args:
        input_path (Path): directory containing images (patches / dataset)
        val_path (Path): directory containig validation file, in case it was already created
        model_path (WindowTras): directory containing pretrained model
        output_path (Path)
    """
    # Check GPU connection:
    print("GPU: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Check input arguments are right:
    assert (
        input_path.resolve().is_dir()
    ), f"Provided 'input_path' directory ({input_path}) doesn't exist!"

    # model mask_pool4_transformer4: epoch=0-step=28500-v1.ckpt"
    # model Full train GPU mask all datasets: epoch=0-step=50000.ckpt"
    pretrainedModel = WindowTransformerDetector.load_from_checkpoint(
        # f"{model_path}/epoch=0-step=28500-v1.ckpt"
        f"{model_path}/epoch=0-step=12500.ckpt"
    ).cpu()

    # model
    model = FineTunedSlidingWindowDetector(
        pretrainedModel=pretrainedModel,
        pooling="avg",
        act_fct=Sigmoid(),
        warmup=warmup,
    )

    model_name = f"{model.__class__.__name__}_{parameters_k(model)}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    print(model_name)

    # validation
    val_file = Path(f"{val_path}")
    val = CachedArtifactDataset(file=val_file)
    val_loader = DataLoader(val, batch_size=batch_size)

    # train
    train_data, train_weights = load_series(train_datasets, "TRAIN", str(input_path))
    train_dataset = CenteredArtifactDataset(
        train_data,
        width=width,
        padding=64,
        artifact=artifact,
        weight=train_weights,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    print("sanity Check: ")
    # sanity check
    batch = next(iter(train_loader))
    batch["data"]

    # initialize callbacks
    checkpointcallback = ModelCheckpoint(
        monitor="validation",
        mode="min",
        save_top_k=1,
        dirpath=output_path,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # initialize logger
    logger = MLFlowLogger(
        log_model="all",
        run_name=model_name,
        experiment_name="transformer_fineTune_slidingWindow",
        tracking_uri=mlflow.get_tracking_uri(),
    )
    early_stop_callback = EarlyStopping(
        monitor="val_fbeta", min_delta=0.0, patience=20, verbose=True, mode="max"
    )

    # initialize trainer
    trainer = Trainer(
        logger=logger,
        max_steps=50000,
        val_check_interval=500,
        callbacks=[checkpointcallback, lr_monitor, early_stop_callback],
        accelerator="gpu",
        devices=1,
    )

    # Set mlflow_experiment:
    mlflow.set_experiment("transformer_fineTune_slidingWindow")
    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_every_n_step=500)

    # Train the model.
    with mlflow.start_run(
        # nested=False,
        run_name=model_name,
        experiment_id="transformer_fineTune_slidingWindow",
    ) as run:
        print("Starting training.")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training completed.")

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    # # Save model:
    # mlflow.pytorch.save_model(pytorch_model=model, path=[output_path + "_model"])

    # model.save()

    # End mlflow run:
    mlflow.end_run()

    print("Job completed successfully.")


if __name__ == "__main__":
    typer.run(main)
