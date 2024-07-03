"""
Artifact module
===============

This module defines abstract and concrete classes for generating different types of artifacts.
The primary purpose of this module is to simulate ramping artifacts in the energy context.

Classes:
    Artifact: An abstract base class for artifact generation.
    Saw: A class for generating sawtooth artifacts.
    Saw_centered: A class for generating centered sawtooth artifacts.
    Saw_centered_Francois: A class for generating sawtooth artifacts with specific parameters for system imbalance data.

Usage:
    Instantiate a subclass of `Artifact` and call its `generate` method to produce an artifact.

Example:
    saw = Saw()
    artifact = saw.generate()
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class Artifact(ABC):
    """
    Abstract base class for generating artifacts.

    Attributes:
        max_width (float): Maximum width of the artifact.
        generator (Generator): Random number generator.
    """

    def __init__(self, max_width: float = 59) -> None:
        """
        Initialize the Artifact with a maximum width and a random number generator.

        :param max_width: Maximum width of the artifact, defaults to 59.
        :type max_width: float
        """
        self.max_width = max_width
        self.generator = np.random.default_rng()

    @abstractmethod
    def generate(
        self, max_width: int, min_width: int, min_rate: float, max_rate: float
    ) -> np.ndarray:
        """
        Abstract method to generate an artifact.

        :param max_width: Maximum artifact width.
        :type max_width: int
        :param min_width: Minimum artifact width.
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp.
        :param min_rate: Minimum absolute slope of the ramp.
        :return: An array with the artifact.
        :rtype: np.ndarray
        """
        pass


class Saw(Artifact):
    """Sawtooth artifact - Ramping artifact in the energy context."""

    def generate(
        self, max_width: int = 59, min_width: int = 1, min_rate=0.00023, max_rate=0.387
    ) -> np.ndarray:
        """
        Generate isolated ramping artifact. If width is 1 or 2, the artifact is empty.

        :param max_width: Maximum artifact width, defaults to 59.
        :type max_width: int
        :param min_width: Minimum artifact width, defaults to 1.
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp, defaults to 0.387.
        :type max_rate: float
        :param min_rate: Minimum absolute slope of the ramp, defaults to 0.00023.
        :type min_rate: float
        :return: An array with the artifact.
        :rtype: np.ndarray
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

        :param max_width: Maximum artifact width, defaults to 59.
        :type max_width: int
        :param min_width: Minimum artifact width, defaults to 3.
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp, defaults to 0.387.
        :type max_rate: float
        :param min_rate: Minimum absolute slope of the ramp, defaults to 0.00023.
        :type min_rate: float
        :return: Tuple containing an array with the artifact and the point of activation.
        :rtype: Tuple[Any, int]
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

        :param max_width: Maximum artifact width, defaults to 20.
        :type max_width: int
        :param min_width: Minimum artifact width, defaults to 2.
        :type min_width: int
        :param max_rate: Maximum absolute slope of the ramp, defaults to 0.45.
        :type max_rate: float
        :param min_rate: Minimum absolute slope of the ramp, defaults to 0.025.
        :type min_rate: float
        :return: Tuple containing an array with the artifact and the point of activation.
        :rtype: Tuple[Any, int]
        """
        width = self.generator.integers(min_width, max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation)), activation
