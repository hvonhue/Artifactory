from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
"""
Artifact module
"""

class Artifact(ABC):
    """Base class for artifacts."""

    def __init__(self, max_width: float = 59) -> None:
        """
        Initialising default values
        :param max_width: Maximum width of the artifact
        :type max_width: float
        """
        self.max_width = max_width
        self.generator = np.random.default_rng()

    @abstractmethod
    def generate(
        self, max_width: int, min_width: int, min_rate: float, max_rate: float
    ) -> np.ndarray:
        pass


class Saw(Artifact):
    """Sawtooth artifact - Ramping artifact in the energy context."""

    def generate(
        self, max_width: int = 59, min_width: int = 1, min_rate=0.00023, max_rate=0.387
    ) -> np.ndarray:
        """
        Generate isolated ramping artifact. If width is 1 or 2, the artifact is empty.

        :param max_width: Maximum artifact width
        :type max_width: int
        :param min_width: Minimum artifact width
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp
        :param min_rate: Minimum absoluteslope of the ramp
        :return: An array with the artifact
        """
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
    """Centered Sawtooth artifact. """

    def generate(
        self, max_width: int = 59, min_width: int = 3, min_rate=0.00023, max_rate=0.387
    ) -> Tuple[Any, int]:
        """
        Generate isolated centered ramping artifact.
        Point of activation is returned to place in the center of a window. No empty artifacts.

        :param max_width: Maximum artifact width
        :type max_width: int
        :param min_width: Minimum artifact width
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp
        :param min_rate: Minimum absoluteslope of the ramp
        :return: Tuple containing an array with the artifact and the point of activation
        """
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
    """Sawtooth artifact. Ramping artifact in system imbalance data with adapted parameters."""

    def generate(
        self, max_width: int = 20, min_width: int = 2, min_rate=0.025, max_rate=0.45
    ) -> Tuple[Any, int]:
        """
        Generate isolated centered ramping artifact. No empty artifacts.

        :param max_width: Maximum artifact width
        :param min_width: Minimum artifact width
        :param max_rate: Maximum absolute slope of the ramp
        :param min_rate: Minimum absoluteslope of the ramp
        :return: Tuple containing an array with the artifact and the point of activation
        """
        width = self.generator.integers(min_width, max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation)), activation
