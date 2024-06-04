"""
python train_VAE.py --input-path /workspaces/AICoE_Ramping_Artefacts/artifactory-master/data/processed --output-path /workspaces/AICoE_Ramping_Artefacts/artifactory-master/data/output
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
from data import CenteredArtifactDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from unsupervised_detector_alpercanberk import VAE
from utilities import parameters_k

# stop warnings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

run = "embsz16_latsz16_bsz128_lay64-128-256-128-64_ep100_cosineWR_v1"
latent_dim = 16
layer_sizes = "64,128,256,128,64"
batch_norm = True
stdev = 0.1
kld_beta = 0.05
lr = 0.001
weight_decay = 1e-5
batch_size = 128
epochs = 60
# # width of window
width = 512
num_features = 32
latent_dim = 16
loss = "label"  # "mask" for mask detector, "label for sliding window"
loss_boost_fp = 0.5
artifact = Saw_centered()
batch_size = 32  # 'values': [32, 64, 128]
warmup = 15000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = VAE(
    latent_dim=128,
    input_height=1,
    input_width=width,
    input_channels=1,
    lr=1e-2,
    batch_size=batch_size,
)

model_name = f"{model.__class__.__name__}_{loss}_{parameters_k(model)}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
run_name = model_name

train_datasets = [
    "australian_electricity_demand_dataset",
    "electricity_hourly_dataset",
    "electricity_load_diagrams",
    "HouseholdPowerConsumption1",
    # "london_smart_meters_dataset_without_missing_values",
    "solar_10_minutes_dataset",
    "wind_farms_minutely_dataset_without_missing_values",
    # 'ACSF1',
    # 'CinCECGTorso',
    # 'HouseTwenty',
    # 'Mallat',
    # 'MixedShapesRegularTrain',
    # 'Phoneme',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'Rock',
    # 'SemgHandGenderCh2',
    # 'mitbih',
    # 'ptbdb',
    # 'ETTh',
    # 'ETTm'
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


# Run it:
def main(
    input_path: Path = typer.Option(default=...),
    output_path: Path = typer.Option(default=...),
):
    """
    Args:
        input_path (Path): directory containing images (patches / dataset)
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
        experiment_name="artifactory_CNN_Pool_detector",
        tracking_uri=mlflow.get_tracking_uri(),
    )

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

    # sanity check
    batch = next(iter(train_loader))
    print(batch["data"])

    # initialize callbacks
    checkpointcallback = ModelCheckpoint(
        dirpath=output_path, monitor="train_loss", mode="min", save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=0.0, patience=20, verbose=True, mode="min"
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

    print("Starting training.")
    trainer.fit(model, train_dataloaders=train_loader)

    print("Training completed.")

    print("Job completed successfully.")


if __name__ == "__main__":
    typer.run(main)
