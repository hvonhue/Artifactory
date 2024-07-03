"""
This module contains classes for creating and handling datasets of artifacts for 
machine learning models. The main classes included are:

- ArtifactDataset: Generates artifacts and sequences of data with optional padding and weighting.
- RealisticArtifactDataset: Ensures more realistic artifact generation by considering activity in windows.
- CenteredArtifactDataset: Centers artifacts within the data windows and includes options for the presence of artifacts.
- CenteredArtifactDataOnlyDataset: Similar to CenteredArtifactDataset but focuses only on the data with artifacts included.
- RejectionSamplingCenteredDataset: Uses rejection sampling to generate data sequences with artifacts.
- TestArtifactDataset: A test dataset containing data sequences and labels, supporting both data and file input.
- CachedArtifactDataset: Caches generated artifact data for reuse and quick access.
- load_files: Utility function to load data from multiple files.
- load_file: Utility function to load data from a single file.

These classes are designed to facilitate the creation and manipulation of datasets for training 
and evaluating machine learning models, particularly in the context of detecting artifacts within data sequences.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from artifact import Artifact
from sliding_window_detector import SlidingWindowTransformerDetector
from torch.utils.data import Dataset, IterableDataset



class ArtifactDataset(IterableDataset):
    """
    Artifact dataset.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.max_rates = [(s.max() - s.min()) / 10 for s in self.data]
        self.min_rates = [(s.max() - s.min()) / 20 for s in self.data]
        self.artifact = artifact
        self.width = width
        # fixed position
        self._position = width // 2 - artifact.max_width // 2
        self._position_mask = np.zeros(width, dtype=np.float32)
        self._position_mask[self._position] = 1
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """
        Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            dict: A dictionary containing the window, the artifact, the mask, and the position.
        """
        # pick a sequence
        # i = self.rng.integers(0, len(self.data) - 1)
        i = self.rng.choice(len(self.data), p=self.weight)
        sequence = self.data[i]
        # generate a window
        while True:
            length = self.rng.integers(0, len(sequence) - self.width)
            window = sequence[length : length + self.width]
            if window.sum() > 0.01:
                break
        # generate artifact
        artifact = self.artifact.generate(
            max_rate=self.max_rates[i], min_rate=self.min_rates[i]
        )
        length = len(artifact)
        # generate position
        if self.padding == "center":
            position = self._position
            position_mask = self._position_mask
        else:
            position = self.rng.integers(
                self.padding, self.width - (self.artifact.max_width + self.padding)
            )
            position_mask = np.zeros_like(window, dtype=np.float32)
            position_mask[position : position + length] = 1
        # generate delta
        delta = np.zeros_like(window, dtype=np.float32)
        delta[position : position + length] = artifact
        # mask
        m = np.zeros_like(window, dtype=np.float32)
        m[position : position + length] = 1
        # return
        return {
            "data": window,
            "artifact": delta,
            "mask": m,
            "position": position,
            "position_mask": position_mask,
        }


class RealisticArtifactDataset(IterableDataset):
    """
    Artifact dataset mimicing ramping artifacts for a gas plant.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
        p_has_artifact (float, optional): Probability of having an artifact. Defaults to 0.5.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.artifact = artifact
        self.width = width
        # fixed position
        self._position = width // 2 - artifact.max_width // 2
        self._position_mask = np.zeros(width, dtype=np.float32)
        self._position_mask[self._position] = 1
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """
        Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            dict: A dictionary containing the window, the artifact, the mask, and the position.
        """
        # pick a sequence
        # i = self.rng.integers(0, len(self.data) - 1)
        i = self.rng.choice(len(self.data), p=self.weight)
        sequence = self.data[i]
        # generate a window
        windows_tried = 0
        while True:
            # only check 3 different windows for activity
            # if all had no activity, choose another sequence
            if windows_tried > 3:
                i = self.rng.choice(len(self.data), p=self.weight)
                sequence = self.data[i]
                windows_tried = 0
            seq_start = self.rng.integers(0, len(sequence) - self.width)
            window = sequence[seq_start : seq_start + self.width]
            windows_tried = windows_tried + 1
            # check whether there is activity in the chosen window
            if window.sum() > 0.01:
                break
        # generate artifact
        artifact = self.artifact.generate()
        length = len(artifact)
        # generate position
        if self.padding == "center":
            position = self._position
            position_mask = self._position_mask
        else:
            position = self.rng.integers(
                self.padding, self.width - (self.artifact.max_width + self.padding)
            )
            position_mask = np.zeros_like(window, dtype=np.float32)
            position_mask[position : position + length] = 1
        # generate delta
        delta = np.zeros_like(window, dtype=np.float32)
        delta[position : position + length] = artifact
        # mask
        m = np.zeros_like(window, dtype=np.float32)
        # if width of artifact is 1, sequence will have no artifact
        if length > 1:
            m[position : position + length] = 1
        # return
        return {
            "data": window,
            "artifact": delta,
            "mask": m,
            "position": position,
            "position_mask": position_mask,
        }


class CenteredArtifactDataset(IterableDataset):
    """
    Artifact dataset mimicing ramping artifacts for a gas plant, placing artifacts in the center of a window.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
        p_has_artifact (float, optional): Probability of having an artifact. Defaults to 0.5.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
        p_has_artifact: float = 0.5,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.artifact = artifact
        self.width = width
        # fixed position
        self._position = width // 2
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight
        self.p_has_artifact = p_has_artifact

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            A dictionary containing the window, the artifact and the label.

        """
        # pick a sequence
        # i = self.rng.integers(0, len(self.data) - 1)
        i = self.rng.choice(len(self.data), p=self.weight)
        sequence = self.data[i]
        # generate a window
        windows_tried = 0
        while True:
            # only check 3 different windows for activity
            # if all had no activity, choose another sequence
            if windows_tried > 3:
                i = self.rng.choice(len(self.data), p=self.weight)
                sequence = self.data[i]
                windows_tried = 0
            seq_start = self.rng.integers(0, len(sequence) - self.width)
            window = sequence[seq_start : seq_start + self.width]
            windows_tried = windows_tried + 1
            # check whether there is activity in the chosen window
            if window.sum() > 0.01:
                break
        # determine whether sanple will have an artifact
        has_artifact = self.rng.random() <= self.p_has_artifact
        if has_artifact:
            # generate artifact
            artifact, activation = self.artifact.generate()
            length = len(artifact)
            # generate position
            position_start = self._position - activation
            # generate delta
            delta = np.zeros_like(window, dtype=np.float32)
            delta[position_start : position_start + length] = artifact
            label = 1
        else:
            delta = np.zeros_like(window, dtype=np.float32)
            label = 0

        # return
        return {
            "data": window,
            "artifact": delta,
            "label": label,
        }


class CenteredArtifactDataOnlyDataset(IterableDataset):
    """
    Artifact dataset mimicing ramping artifacts for a gas plant, placing the artifact in the center.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
        p_has_artifact (float, optional): Probability of having an artifact. Defaults to 0.5.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
        p_has_artifact: float = 0.5,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.artifact = artifact
        self.width = width
        # fixed position
        self._position = width // 2
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight
        self.p_has_artifact = p_has_artifact

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            A dictionary containing the window and the label.

        """
        # pick a sequence
        # i = self.rng.integers(0, len(self.data) - 1)
        i = self.rng.choice(len(self.data), p=self.weight)
        sequence = self.data[i]
        print(i)
        # generate a window
        windows_tried = 0
        while True:
            # only check 3 different windows for activity
            # if all had no activity, choose another sequence
            if windows_tried > 3:
                i = self.rng.choice(len(self.data), p=self.weight)
                sequence = self.data[i]
                print(i)
                windows_tried = 0
                # break
            seq_start = self.rng.integers(0, len(sequence) - self.width)
            window = sequence[seq_start : seq_start + self.width]
            windows_tried = windows_tried + 1
            # check whether there is activity in the chosen window
            if window.sum() > 0.01:
                break
        # determine whether sanple will have an artifact
        has_artifact = self.rng.random() <= self.p_has_artifact
        if has_artifact:
            # generate artifact
            artifact, activation = self.artifact.generate()
            length = len(artifact)
            # generate position
            position_start = self._position - activation
            # generate delta
            delta = np.zeros_like(window, dtype=np.float32)
            delta[position_start : position_start + length] = artifact
            label = 1
        else:
            delta = np.zeros_like(window, dtype=np.float32)
            label = 0

        # return
        return {"data": (window + delta), "label": label}


class RejectionSamplingCenteredDataset(IterableDataset):
    """
    Artifact dataset mimicing ramping artifacts for a gas plant, placing the artifact in the center.
    Uses a previously trained model to perform rejection sampling.
    Samples with a lower confidence score or misclassified samples have a higher probability to
    be sampled for training.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        model (SlidingWindowTransformerDetector): The previous trained model.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
        p_has_artifact (float, optional): Probability of having an artifact. Defaults to 0.5.
        rejection (float): The proportion of highest confidence values not to be resampled.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        artifact: Artifact,
        model: SlidingWindowTransformerDetector,
        width: int,
        padding: str | int = "center",
        weight: Optional[list[float]] = None,
        p_has_artifact: float = 0.5,
        rejection: float = 0.1,
    ) -> None:
        """"""
        # properties
        self.data = data
        self.artifact = artifact
        self.model = model
        self.width = width
        # fixed position
        self._position = width // 2
        # padding
        self.padding = padding
        # random generator and weight for sampling
        self.rng = np.random.default_rng()
        self.weight = weight
        self.p_has_artifact = p_has_artifact
        # rejection parameter: Leave samples with high confidence out of sampling
        # (confidence > 1 - rejection)
        self.rejection = rejection

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        """Generate an artifact.

        Ensures that each window has some activity.

        Returns:
            A dictionary containing the window, the artifact, the mask and the position.

        """
        while True:
            # pick a sequence
            # i = self.rng.integers(0, len(self.data) - 1)
            i = self.rng.choice(len(self.data), p=self.weight)
            sequence = self.data[i]
            # generate a window
            windows_tried = 0
            while True:
                # only check 3 different windows for activity
                # if all had no activity, choose another sequence
                if windows_tried > 3:
                    i = self.rng.choice(len(self.data), p=self.weight)
                    sequence = self.data[i]
                    windows_tried = 0
                seq_start = self.rng.integers(0, len(sequence) - self.width)
                window = sequence[seq_start : seq_start + self.width]
                windows_tried = windows_tried + 1
                # check whether there is activity in the chosen window
                if window.sum() > 0.01:
                    break
            # determine whether sample will have an artifact
            has_artifact = self.rng.random() < self.p_has_artifact
            if has_artifact:
                # generate artifact
                artifact, activation = self.artifact.generate()
                length = len(artifact)
                # generate position
                position_start = self._position - activation
                # generate delta
                delta = np.zeros_like(window, dtype=np.float32)
                delta[position_start : position_start + length] = artifact
                label = 1
            else:
                delta = np.zeros_like(window, dtype=np.float32)
                label = 0

            # TODO: Konzept überarbeiten - Auch wenn confidence 0,
            # wird manchmal nicht genommen
            prediction = self.model(torch.tensor(window + delta).unsqueeze(0))
            confidence = 1 - np.abs(label - prediction.detach().numpy())
            # sample more frequently if confidence is low
            if 1 - confidence - self.rejection > np.random.rand():
                break

        # return
        return {
            "data": window,
            "artifact": delta,
            "position": self._position,
            "label": label,
        }


class TestArtifactDataset(Dataset):
    """
    Test artifact dataset mimicking ramping artifacts for a gas plant.

    Args:
        data (list[np.ndarray]): List of numpy arrays representing the data.
        artifact (Artifact): The artifact generator instance.
        width (int): The width of the data window.
        padding (str | int, optional): The padding type or size. Defaults to "center".
        weight (Optional[list[float]], optional): Weights for sampling. Defaults to None.
        p_has_artifact (float, optional): Probability of having an artifact. Defaults to 0.5.
    """

    def __init__(
        self, data: Optional[list] = None, file: Union[str, Path, None] = None
    ) -> None:
        """
        Initialize the TestArtifactDataset with either data or a file path.

        Args:
            data (Optional[list]): List of numpy arrays representing the data.
            file (Union[str, Path, None]): Path to a file containing the data.
        """
        if data is not None:
            self.data = data
        elif file is not None:
            self.data = pickle.load(open(file, "rb"))

    def __len__(self) -> int:
        """
        Return the number of window in the dataset.

        Returns:
            int: Number of window.
        """
        return len(self.data)

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Retrieve the window at the given index.

        Args:
            i (int): Index of the data point.

        Returns:
            np.ndarray: window.
        """
        return self.data[i]

    @classmethod
    def generate(
        cls,
        labels: list[int],
        width: int,
        n: int,
        data: list[int],
        to: Union[str, Path, None] = None,
        start_index: int = 0,
    ):
        """
        Generate a dataset with specified parameters.

        Args:
            labels (list[int]): List of artifact labels.
            width (int): Width of the data window.
            n (int): Number of samples to generate.
            data (list[int]): List of data points.
            to (Union[str, Path, None], optional): Path to save the generated data. Defaults to None.
            start_index (int, optional): Starting index for the data. Defaults to 0.

        Returns:
            TestArtifactDataset: Generated dataset.
        """
        data = data[start_index:]
        labels = labels[start_index:]
        data_dicts = []
        for i in range(n):
            sequence = data[i * width : (width * i) + width]
            artifact_labels = labels[i * width : (width * i) + width]
            data_dicts.append(
                {
                    "data": sequence,
                    "mask": artifact_labels,
                    "start_index": start_index + i * width,
                    "dataset": "real",
                }
            )

        if to is not None:
            pickle.dump(data_dicts, open(to, "wb"))
        return cls(data=data_dicts)


class CachedArtifactDataset(Dataset):
    """
    Cached artifact dataset.

    Args:
        data (Optional[list[np.ndarray]]): List of numpy arrays representing the data.
        file (Union[str, Path, None]): Path to a file containing the data.
    """

    def __init__(
        self, data: Optional[list] = None, file: Union[str, Path, None] = None
    ) -> None:
        """
        Initialize the CachedArtifactDataset with either data or a file path.

        Args:
            data (Optional[list[np.ndarray]]): List of numpy arrays representing the data.
            file (Union[str, Path, None]): Path to a file containing the data.
        """

        if data is not None:
            self.data = data
        elif file is not None:
            self.data = pickle.load(open(file, "rb"))

    def __len__(self) -> int:
        """
        Return the number of windows in the dataset.

        Returns:
            int: Number of windows.
        """
        return len(self.data)

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Retrieve the window at the given index.

        Args:
            i (int): Index of the window.

        Returns:
            np.ndarray: window.
        """
        return self.data[i]

    @classmethod
    def generate(
        cls, dataset: ArtifactDataset, n: int, to: Union[str, Path, None] = None
    ):
        """
        Generate a dataset from an ArtifactDataset instance.

        Args:
            dataset (ArtifactDataset): Instance of ArtifactDataset to generate data from.
            n (int): Number of samples to generate.
            to (Union[str, Path, None], optional): Path to save the generated data. Defaults to None.

        Returns:
            CachedArtifactDataset: Generated dataset.
        """
        data = [next(dataset) for _ in range(n)]
        if to is not None:
            pickle.dump(data, open(to, "wb"))
        return cls(data=data)


def load_files(files: np.ndarray | str | Path | list[str | Path]) -> np.ndarray:
    """
    Load data from multiple files.

    Args:
        files (Union[np.ndarray, str, Path, list[Union[str, Path]]]): List or array of file paths.

    Returns:
        np.ndarray: Loaded data.
    """
    if isinstance(files, Path) or isinstance(files, str):
        files = [files]
    data = list()
    for file in files:
        data.extend(load_file(file))
    return data


def load_file(file: str | Path) -> list[np.ndarray]:
    """
    Load data from a single file.

    Args:
        file (Union[str, Path]): Path to the file.

    Returns:
        list[np.ndarray]: Loaded data.
    """
    with open(file, "rb") as f:
        return pickle.load(f)
