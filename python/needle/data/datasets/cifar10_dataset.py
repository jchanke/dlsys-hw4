import os
import pickle
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sized,
    Union,
)

import numpy as np

from ..data_basic import Dataset

# There are data_batch_{1,2,3,4,5} in cifar-10-batches-py
N_DATA_BATCH = 5


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None,
    ):
        """
        Parameters:
          base_folder (str): cifar-10-batches-py folder filepath
          train (bool): if True load training dataset, else load test dataset

        Divide pixel values by 255. so that images are in 0-1 range.

        Attributes:
          X (np.ndarray): NumPy array of images
          y (np.ndarray): NumPy array of labels

        ======================================================================
        For more documentation about the CIFAR-10 dataset, see:
        https://www.cs.toronto.edu/~kriz/cifar.html.
        """
        ### BEGIN YOUR SOLUTION
        # Add transforms to self.transforms (data_basic.py)
        super().__init__(transforms)

        # Unpickle data — see instructions on doc's website
        X = np.empty((0, 1024 * 3))
        y = []
        if train:
            for i in range(1, N_DATA_BATCH + 1):
                d = unpickle(os.path.join(base_folder, f"data_batch_{i}"))
                X = np.concatenate((X, d[b"data"]), axis=0)
                y += d[b"labels"]
        else:
            d = unpickle(os.path.join(base_folder, "test_batch"))
            X = d[b"data"]
            y = d[b"labels"]

        # Reshape each image from (N, C*H*W) -> (N, C, H, W),
        # then normalize pixel values
        X = X.reshape((-1, 3, 32, 32)) / 255

        # Convert y from List[int] to np.ndarray
        y = np.array(y)

        assert len(X) == len(y), "data and labels have different lengths"

        self.X = X
        self.y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


def unpickle(
    file: str,
) -> Dict[
    Literal[
        # For {data, test}_batch's
        b"batch_label",
        b"labels",
        b"data",
        b"filenames",
        # For batches.meta
        b"label_names",
    ],
    np.ndarray,
]:
    """
    Unpickles the CIFAR-10 pickle'd objects to Python dictionaries.
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
