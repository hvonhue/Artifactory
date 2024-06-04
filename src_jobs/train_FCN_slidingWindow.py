"""
python train_CNN_slidingWindow.py --input-path /workspaces/AICoE_Ramping_Artefacts/artifactory-master/data/processed --val-path /workspaces/AICoE_Ramping_Artefacts/artifactory-master/data/validation_slidingWindow_noLondon512.pkl --output-path /workspaces/AICoE_Ramping_Artefacts/artifactory-master/data/output
"""
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
from mlflow import MlflowClient
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from sliding_window_detector import ConvolutionalSlidingWindowDetector
from torch.utils.data import DataLoader
from utilities import parameters_k

# stop warnings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# # width of window
width = 512
convolution_features = [128, 256, 128]
convolution_width = [8, 5, 3]
convolution_dropout = 0.0
batch_normalization = True
loss = "label"  # "mask" for mask detector, "label for sliding window"
loss_boost_fp = 0
artifact = Saw_centered()
batch_size = 32  # 'values': [32, 64, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = ConvolutionalSlidingWindowDetector(
    window=width,
    convolution_features=convolution_features,
    convolution_width=convolution_width,
    convolution_dropout=0.0,
    batch_normalization=batch_normalization,
    loss=loss,
)
model_name = f"{model.__class__.__name__}_{loss}_{parameters_k(model)}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
run_name = model_name

val_datasets = [
    "australian_electricity_demand_dataset",
    "electricity_hourly_dataset",
    "electricity_load_diagrams",
    "HouseholdPowerConsumption1",
    # "HouseholdPowerConsumption2",
    # "london_smart_meters_dataset_without_missing_values",
    "solar_10_minutes_dataset",
    "wind_farms_minutely_dataset_without_missing_values",
]
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
print(model_name)


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
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


# Run it:
def main(
    input_path: Path = typer.Option(default=...),
    val_path: Path = typer.Option(default=...),
    output_path: Path = typer.Option(default=...),
):
    """
    Args:
        input_path (Path): directory containing images (patches / dataset)
        val_path (Path): directory containig validation file, in case it was already created
    """

    # Check input arguments are right:
    assert (
        input_path.resolve().is_dir()
    ), f"Provided 'input_path' directory ({input_path}) doesn't exist!"

    # TODO sinnvolles seeding, Artefakte dürfen natürlich keinen seed haben...
    # seed_everything(42, workers=True)

    # initialize logger
    logger = MLFlowLogger(
        log_model="all",
        run_name=model_name,
        experiment_name="artifactory_FCN_detector",
        tracking_uri=mlflow.get_tracking_uri(),
    )

    # validation
    val_file = Path(f"{val_path}")

    if not val_file.exists():
        f"'val_file' at provided 'val_path' directory ({val_path}) doesn't exist! Creating validation file..."
        val_data, val_weights = load_series(val_datasets, "VAL", str(input_path))
        val_gen = CenteredArtifactDataset(
            val_data,
            width=width,
            padding=64,
            artifact=artifact,
            weight=val_weights,
        )
        val = CachedArtifactDataset.generate(val_gen, n=2048, to=val_file)
    else:
        val = CachedArtifactDataset(file=val_file)
    val_loader = DataLoader(val, batch_size=batch_size)

    # train
    train_data, train_weights = load_series(train_datasets, "TRAIN", str(input_path))
    print("Dataset")
    train_dataset = CenteredArtifactDataset(
        train_data,
        width=width,
        padding=64,
        artifact=artifact,
        weight=train_weights,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # initialize callbacks
    checkpointcallback = ModelCheckpoint(
        dirpath=f"{output_path}/{model_name}",
        monitor="val_fbeta",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="validation", min_delta=0.0, patience=10, verbose=True, mode="min"
    )
    # initialize trainer
    trainer = Trainer(
        logger=logger,
        max_steps=50000,
        val_check_interval=500,
        callbacks=[checkpointcallback, lr_monitor, early_stop_callback],
    )
    print("Initialized trainer.")

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_every_n_step=500)

    print("Fit")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training completed.")

    # Fetch the auto logged parameters and metrics.
    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    print("Job completed successfully.")


if __name__ == "__main__":
    typer.run(main)
