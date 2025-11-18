import os
from typing import List, Tuple

import numpy as np

from needle import Tensor
from needle import backend_ndarray as nd


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """

    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, "train.txt"), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, "test.txt"), max_lines)

    def tokenize(self, path, max_lines=None) -> List[int]:
        """
        Args:
          path (str): Path to text file.
          max_lines (int): Maximum number of lines to read in.

        Returns:
          ids (np.ndarray): List of ids

        Tokenizes a text file, first adding each word in the file to the
        dictionary, and then tokenizing the text file to a list of IDs. When
        adding words to the dictionary (and tokenizing the file content) '<eos>'
        should be appended to the end of each line in order to properly account
        for the end of the sentence.
        """
        ### BEGIN YOUR SOLUTION
        ids = []

        # Iterate over lines, adding words to dict & accumulating id's
        with open(path, mode="r") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i == max_lines:
                    break
                for word in line.strip().split():
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
                self.dictionary.add_word("<eos>")
                ids.append(self.dictionary.word2idx["<eos>"])

        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype) -> np.ndarray:
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get

    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.

    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.

    If the data cannot be evenly divided by the batch size, trim off the
    remainder.

    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    data = np.array(data[: nbatch * batch_size])
    return np.reshape(data, (batch_size, nbatch)).swapaxes(0, 1)
    ### END YOUR SOLUTION


def get_batch(
    batches: np.ndarray,
    i: int,
    bptt: int,
    device=None,
    dtype=None,
) -> Tuple[Tensor, Tensor]:
    """
    `get_batch` subdivides the source data into chunks of length `bptt`.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:

      ┌ a g m s ┐ ┌ b h n t ┐
      └ b h n t ┘ └ c i o u ┘

    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.

    Args:
      batches (np.ndarray): Array returned from batchify function.
      i (int): index
      bptt (int): Sequence length.

    Returns:
      data (ndl.Tensor):
        Tensor of shape (bptt, bs) with cached data as NDArray
      target (ndl.Tensor):
        Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    data = Tensor(batches[i : i + bptt], device=device, dtype=dtype)

    # To get the target: un-batchify the sequence, cyclic shift, then re-batchify
    _, bs = batches.shape
    unbatched = np.swapaxes(batches, 0, 1).reshape((-1,))
    shifted = np.roll(unbatched, -1)
    batches_shifted = batchify(shifted, bs, device=device, dtype=dtype)
    target = Tensor(
        batches_shifted[i : i + bptt].reshape((-1,)),
        device=device,
        dtype=dtype,
    )

    # --- Actual solution from course staff ---
    # seq_len = min(bptt, len(batches) - i - 1)
    # data = Tensor(batches[i : i + seq_len], device=device, dtype=dtype)
    # target = Tensor(
    #     batches[i + 1 : i + seq_len + 1].reshape((-1,)),
    #     device=device,
    #     dtype=dtype,
    # )

    return data, target
    ### END YOUR SOLUTION
