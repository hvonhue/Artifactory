"""
python train_rejectionSampling_slidingWindow.py --input-path ../data/processed --val-path ../data/val_files/val_SW_noCiECGT512.pkl --model-path ../models/SW_adaFCN_Trans.ckpt  --output-path ../data/output
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
from data import CachedArtifactDataset, RejectionSamplingCenteredDataset
from modeling import DelayedEarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sliding_window_detector import SlidingWindowTransformerDetector
from torch.utils.data import DataLoader
from utilities import parameters_k

# stop warnings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

width = 512
loss = "label"
loss_boost_fp = 0.2
artifact = Saw_centered()
batch_size = 32  # 'values': [32, 64, 128]
warmup = 30
rejection = 0.3  # highest fraction that will not be resampled

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_datasets = [
    # 'CinCECGTorso', # do not train on this dataset for validation purposes
    'ETTm', # 1
    'ETTh', # 2
    "electricity_load_diagrams",  # 3
    "australian_electricity_demand_dataset",  # 4
    'Phoneme', # 5
    "electricity_hourly_dataset",  # 6
    "HouseholdPowerConsumption1",  # 7
    "london_smart_meters_dataset_without_missing_values",  # 8
    'SemgHandGenderCh2', # 9
    'PigCVP', # 10
    'HouseTwenty', # 11
    "wind_farms_minutely_dataset_without_missing_values",  # 12
    'ptbdb', # 13
    'mitbih', # 14
    'PigArtPressure', # 15
    "solar_10_minutes_dataset", # 16
    'Mallat', # 17
    'MixedShapesRegularTrain', # 18
    'Rock', # 19
    'ACSF1', # 20
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

    pretrainedModel = SlidingWindowTransformerDetector.load_from_checkpoint(
        f"{model_path}"
        # f"{model_path}"
    )

    # model
    model = pretrainedModel.to(device)

    model_name = f"{model.__class__.__name__}_{parameters_k(model)}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    print(model_name)

    # validation
    val_file = Path(f"{val_path}")
    val = CachedArtifactDataset(file=val_file)
    val_loader = DataLoader(val, batch_size=batch_size)

    # train
    train_data, train_weights = load_series(train_datasets, "TRAIN", str(input_path))
    train_dataset = RejectionSamplingCenteredDataset(
        train_data,
        model=model,
        width=width,
        padding=64,
        artifact=artifact,
        weight=train_weights,
        rejection=rejection,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    print("sanity Check: ")
    # sanity check
    batch = next(iter(train_loader))
    batch["data"]

    # initialize logger
    logger = MLFlowLogger(
        log_model="all",
        run_name=model_name,
        experiment_name="transformer_rejection_slidingWindow",
        tracking_uri=mlflow.get_tracking_uri(),
    )

    # initialize callbacks
    checkpointcallback = ModelCheckpoint(
        # every_n_train_steps = 1000,
        # save_top_k = -1,
        monitor="val_fbeta",
        mode="max",
        save_top_k=1,
        dirpath=f"{output_path}/{model_name}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = DelayedEarlyStopping(
        monitor="validation", min_delta=0.005, patience=10, warmupES=5000
    )

    # initialize trainer
    trainer = Trainer(
        logger=logger,
        max_steps=20000,
        val_check_interval=500,
        callbacks=[checkpointcallback, lr_monitor, early_stop_callback],
    )
    print("Initialized trainer.")

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_every_n_step=500)

    print("Starting training.")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Training completed.")

    print("Job completed successfully.")


if __name__ == "__main__":
    typer.run(main)
