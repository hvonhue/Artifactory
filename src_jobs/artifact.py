from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

speed_weights = [
    0.06837607,
    0.1025641,
    0.17948718,
    0.03418803,
    0.07692308,
    0.04273504,
    0.01709402,
    0.05982906,
    0.04273504,
    0.03418803,
    0.07692308,
    0.05128205,
    0.05982906,
    0.08547009,
    0.04273504,
    0,
    0.01709402,
    0,
    0.00854701,
]
speed_bins = [
    0.01,
    0.03,
    0.05,
    0.07,
    0.09,
    0.11,
    0.13,
    0.15,
    0.17,
    0.19,
    0.21,
    0.23,
    0.25,
    0.27,
    0.29,
    0.31,
    0.33,
    0.35,
    0.37,
    0.39,
]
width_weights = [
    0.08403361,
    0.07563025,
    0.10084034,
    0.05042017,
    0.05882353,
    0.05882353,
    0.03361345,
    0.03361345,
    0.00840336,
    0.15966387,
    0.04201681,
    0.05042017,
    0.01680672,
    0.05042017,
    0.08403361,
    0.03361345,
    0.05042017,
    0.00840336,
    0,
]
width_bins = [
    1.5,
    4.5,
    7.5,
    10.5,
    13.5,
    16.5,
    19.5,
    22.5,
    25.5,
    28.5,
    31.5,
    34.5,
    37.5,
    40.5,
    43.5,
    46.5,
    49.5,
    52.5,
    55.5,
    58.5,
]
position_weights = [
    0,
    0.02197802,
    0.06593407,
    0.10989011,
    0.03296703,
    0.01098901,
    0.03296703,
    0.13186813,
    0.12087912,
    0.0989011,
    0.04395604,
    0.02197802,
    0.02197802,
    0.05494505,
    0.06593407,
    0.06593407,
    0.01098901,
    0.03296703,
    0.05494505,
]
position_bins = [
    0.025,
    0.075,
    0.125,
    0.175,
    0.225,
    0.275,
    0.325,
    0.375,
    0.425,
    0.475,
    0.525,
    0.575,
    0.625,
    0.675,
    0.725,
    0.775,
    0.825,
    0.875,
    0.925,
    0.975,
]


class Artifact(ABC):
    """Base class for artifacts."""

    def __init__(self, max_width: int = 59) -> None:
        self.max_width = max_width
        self.generator = np.random.default_rng()

    @abstractmethod
    def generate(self, min_width: int, min_rate: float, max_rate: float) -> np.ndarray:
        pass


class Saw(Artifact):
    """Sawtooth artifact."""

    def generate(
        self, min_width: int = 1, min_rate=0.00023, max_rate=0.387
    ) -> np.ndarray:
        width = self.generator.integers(min_width, self.max_width, endpoint=True)
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
        self, min_width: int = 3, min_rate=0.00023, max_rate=0.387
    ) -> Tuple[Any, int]:
        width = self.generator.integers(min_width, self.max_width, endpoint=True)
        activation = self.generator.integers(width)
        rate = self.generator.uniform(min_rate, max_rate) * np.sign(
            self.generator.uniform(-1, 1)
        )
        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation)), activation


class Saw_normal(Artifact):
    """Sawtooth artifact."""

    def generate(self, min_width=1, min_rate=0.00023, max_rate=0.387) -> np.ndarray:
        # normal distribution of width, position and rate obtained statistically from real data
        width = self.generator.normal(loc=25.31, scale=15.7)
        activation = self.generator.normal(loc=0.394, scale=0.305) * width
        rate = self.generator.normal(loc=0.146, scale=0.097) * np.sign(
            self.generator.uniform(-1, 1)
        )

        # use this if artifact should not be significantly higher or lower than the min/max rate in a wondow
        # while True:
        #     rate = self.generator.normal(loc=0.0244, scale=0.336) * np.sign(
        #         self.generator.uniform(-1, 1)
        #     )
        #     if min_rate < rate and rate < max_rate:
        #         break

        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation))


class Saw_weighted_uniform(Artifact):
    """Sawtooth artifact."""

    def generate(self, min_width=1, min_rate=0.00023, max_rate=0.387) -> np.ndarray:
        # normal distribution of width, position and rate obtained statistically from real data
        speed_bin = self.generator.choice(len(speed_bins), p=speed_weights)
        speed_bin_width = np.diff(speed_bins)[0]
        rate = self.generator.uniform(
            max(min_rate, speed_bins[speed_bin] - speed_bin_width / 2),
            min(max_rate, speed_bins[speed_bin] + speed_bin_width / 2),
            endpoint=False,
        ) * np.sign(self.generator.uniform(-1, 1))

        width_bin = self.generator.choice(len(width_bins), p=width_weights)
        width_bin_width = np.diff(width_bins)[0]
        width = self.generator.integers(
            max(min_width, width_bins[width_bin] - width_bin_width / 2),
            min(self.max_width, width_bins[width_bin] + width_bin_width / 2),
            endpoint=False,
        )

        position_bin = self.generator.choice(len(position_bins), p=position_weights)
        position_bin_width = np.diff(position_bins)[0]
        activation = (
            self.generator.uniform(
                max(0, position_bins[position_bin] - position_bin_width / 2),
                min(1, position_bins[position_bin] + position_bin_width / 2),
                endpoint=False,
            )
            * width
        )

        time = np.arange(width)
        ramp = rate * time
        return ramp - (rate * (width - 1) * (time >= activation))
