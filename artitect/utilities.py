from math import floor, log
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class TrainerCallback(Callback):
    """
    Callback for modules that train other modules.

    If the trained LightningModule has a `trains` attribute,
    the callback will save the trained modules in the
    log directory.

    Methods
    -------
    on_save_checkpoint(trainer: Trainer, pl_module: LightningModule)
        Saves the trained modules' parameters and state dict to the log directory.
    """
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Saves the trained modules' parameters and state dict to the log directory.

        Parameters
        ----------
        trainer : Trainer
            The Trainer instance.
        pl_module : LightningModule
            The LightningModule being trained.
        """
        if not hasattr(pl_module, "trains"):
            return
        folder = Path(trainer.log_dir) / "models"
        folder.mkdir(exist_ok=True)
        for train in pl_module.trains:
            model = getattr(pl_module, train)
            param = getattr(model, "hparams")
            torch.save(
                {"hparams": param, "state": model.state_dict()}, folder / f"{train}.pt"
            )


def plot(artifact: dict, extra=None, extra_label=None) -> None:
    """
    Plot the data, artifact, and mask from the given dictionary.

    Parameters
    ----------
    artifact : dict
        Dictionary containing 'data', 'artifact', and 'mask' keys with their corresponding values.
    extra : list, optional
        List of additional data series to plot.
    extra_label : list, optional
        List of labels for the additional data series.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(artifact["data"] + artifact["artifact"], label="data")
    plt.plot(artifact["artifact"], label="artifact")
    plt.plot(artifact["mask"], label="mask")
    if extra is not None:
        for e, el in zip(extra, extra_label):
            plt.plot(e, label=el)
    plt.legend()
    plt.show()


def parameters(model):
    """
    Calculate the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to calculate parameters for.

    Returns
    -------
    int
        The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_k(model):
    """
    Calculate the number of trainable parameters in a model, formatted with SI prefixes.

    Parameters
    ----------
    model : torch.nn.Module
        The model to calculate parameters for.

    Returns
    -------
    str
        The number of trainable parameters formatted with SI prefixes (K, M, G, etc.).
    """
    number = parameters(model)
    units = ["", "K", "M", "G", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return f"{number / k**magnitude:.2f}{units[magnitude]}"
