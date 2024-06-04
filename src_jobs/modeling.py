from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Embedding,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from torch.optim.lr_scheduler import _LRScheduler

activations = {
    "relu": ReLU(inplace=True),
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
}


def _convolutions(
    convolution_features: list[int],
    convolution_width: int | list[int],
    convolution_dilation: int | list[int] = 1,
    convolution_dropout: float = 0.0,
    batch_normalization: bool = False,
    activation: str = "sigmoid",
    last: bool = True,
    pad: bool = False,
) -> Sequential:
    """Create a sequence of convolutional layers.

    Args:
        convolution_features: Number of features in each layer.
        convolution_width: Width of each layer. If an integer,
            all layers will have the same width. If a list,
            each layer will have the corresponding width.
        last: If False, the last layer will not have an activation.

    """
    if isinstance(convolution_dilation, int):
        convolution_dilation = [convolution_dilation] * (len(convolution_features) - 1)
    if isinstance(convolution_width, int):
        convolution_width = [convolution_width] * (len(convolution_features) - 1)
    layers = Sequential()
    for i in range(len(convolution_features) - 1):
        layers.append(
            Conv1d(
                in_channels=convolution_features[i],
                out_channels=convolution_features[i + 1],
                kernel_size=convolution_width[i],
                dilation=convolution_dilation[i],
                padding="same" if pad else 0,
            )
        )
        if i < len(convolution_features) - 2 or last:
            if batch_normalization:
                layers.append(BatchNorm1d(num_features=convolution_features[i + 1]))
            layers.append(activations[activation])
        if convolution_dropout > 0:
            layers.append(Dropout(convolution_dropout))
    return layers


def _linear(
    features: list[int],
    activation: Module = Sigmoid(),
    last: bool = True,
    batch_normalization: bool = False,
) -> Sequential:
    """Create a sequence of linear layers."""
    layers = Sequential()
    for i in range(len(features) - 1):
        layers.append(Linear(features[i], features[i + 1]))
        if batch_normalization:
            layers.append(BatchNorm1d(num_features=features[i + 1]))
        if i < len(features) - 2 or last:
            layers.append(activation)
    return layers


def _size(s: int, layers: Sequential) -> int:
    """Compute the size of the output."""
    for layer in layers:
        if isinstance(layer, Conv1d):
            s = s - layer.kernel_size[0] + 1
    return s


class SinusoidalPositionEmbedding(Module):
    """Classic sinusoidal positions."""

    def __init__(self, dimension: int, length: int):
        """

        Args:
            dimension: Embedding dimension.
            length: Sequence length.

        """
        super().__init__()
        self.dimension = dimension
        self.length = length

        # initialise position encoding
        idx = torch.arange(self.length).unsqueeze(1)
        den = torch.exp(torch.arange(0, dimension, 2) * (-np.log(10000.0) / dimension))
        pe = torch.zeros(1, self.length, self.dimension)
        pe[0, :, 0::2] = torch.sin(idx * den)
        pe[0, :, 1::2] = torch.cos(idx * den)
        self.register_buffer("position", pe)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.position


class LearnedPositionEmbedding(Module):
    """Learn position embedding."""

    def __init__(self, dimension: int, length: int):
        super().__init__()
        self.dimension = dimension
        self.length = length
        self.embedding = Embedding(length, dimension)
        self.register_buffer("ids", torch.arange(length).unsqueeze(0))

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.embedding(self.ids)


# def Fbeta_loss(pred: torch.Tensor, label: float, beta: float = 0.5) -> float:
#     print(
#         ("loss: ", -label * np.log(pred) + (1 - label) * np.log(beta**2 + pred)).sum()
#     )
#     return (-label * np.log(pred) + (1 - label) * np.log(beta**2 + pred)).sum()


def macro_soft_f1(preds: torch.Tensor, labels: torch.Tensor, beta: float = 0.5):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    soft_f1 = (
        (1 + beta**2) * tp / ((1 + beta**2) * tp + (beta**2) * fn + fp + 1e-16)
    )
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1

    return cost


"""Warm up learning rate scheduler module."""


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_step: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.last_step = last_step

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_step + 1
        return [
            lr
            * self.warmup_steps**-0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]

    def step(self, step=None):
        self.last_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
            print(lr)


class DelayedEarlyStopping(pl.Callback):
    def __init__(
        self,
        patience: int = 5,
        monitor: str = "val_loss",
        warmupES: int = 10,
        min_delta=0.01,
    ):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.warmupES = warmupES
        self.min_delta = min_delta

    def on_validation_end(self, trainer, pl_module):
        print(trainer.global_step)
        if trainer.global_step > self.warmupES:
            current_score = trainer.callback_metrics.get(self.monitor)
            if self.best_score is None:
                self.best_score = current_score

            elif self.min_delta > self.best_score - current_score:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

            else:
                self.best_score = current_score
                self.counter = 0

            if self.early_stop:
                trainer.should_stop = True
