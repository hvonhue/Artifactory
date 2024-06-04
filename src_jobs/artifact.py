from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class Artifact(ABC):
    """Base class for artifacts."""

    def __init__(self, max_width: float = 59) -> None:
        self.max_width = max_width
        self.generator = np.random.default_rng()

    @abstractmethod
    def generate(
        self, max_width: int, min_width: int, min_rate: float, max_rate: float
    ) -> np.ndarray:
        pass


class Saw(Artifact):
    """Sawtooth artifact."""

    def generate(
        self, max_width: int = 59, min_width: int = 1, min_rate=0.00023, max_rate=0.387
    ) -> np.ndarray:
        self.max_width = max_width
        width = self.generator.integers(min_width, max_width, endpoint=True)
        if width > 2:
            activation = self.generator.integers(width)
            rate = self.generator.uniform(min_rate, max_rate) * np.sign(
                self.generator.uniform(-1, 1)
            )
            time = np.arange(width)
            ramp = rate * time
            return ramp - (rate * (width - 1) * (time >= activation))
        else:
            return [0]


class Saw_centered(Artifact):
    """Sawtooth artifact."""

    def generate(
        self, max_width: int = 59, min_width: int = 3, min_rate=0.00023, max_rate=0.387
    ) -> Tuple[Any, int]:
        self.max_width = max_width
        width = self.generator.integers(min_width, max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation)), activation


class Saw_centered_Francois(Artifact):
    """Sawtooth artifact."""

    def generate(
        self, max_width: int = 20, min_width: int = 2, min_rate=0.025, max_rate=0.45
    ) -> Tuple[Any, int]:
        width = self.generator.integers(min_width, max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation)), activation
