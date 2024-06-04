from math import floor, log
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class TrainerCallback(Callback):
    """Callback for modules that train other modules.

    If the trained LightningModule has a trains attribute,
    the callback will save the trained modules in the
    log directory.

    """

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> None:
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_k(model):
    number = parameters(model)
    units = ["", "K", "M", "G", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return f"{number / k**magnitude:.2f}{units[magnitude]}"
