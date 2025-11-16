import sys

sys.path.append("./python")
import math

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(
        self,
        a: int,  # in-channels
        b: int,  # out-channels
        k: int,  # kernel size
        s: int,  # stride
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv(
                in_channels=a,
                out_channels=b,
                kernel_size=k,
                stride=s,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(dim=b, device=device, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(a=3, b=16, k=7, s=4, device=device, dtype=dtype),
            ConvBN(a=16, b=32, k=3, s=2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(a=32, b=32, k=3, s=1, device=device, dtype=dtype),
                    ConvBN(a=32, b=32, k=3, s=1, device=device, dtype=dtype),
                )
            ),
            nn.Sequential(
                ConvBN(a=32, b=64, k=3, s=2, device=device, dtype=dtype),
                ConvBN(a=64, b=128, k=3, s=2, device=device, dtype=dtype),
            ),
            nn.Residual(
                nn.Sequential(
                    ConvBN(a=128, b=128, k=3, s=1, device=device, dtype=dtype),
                    ConvBN(a=128, b=128, k=3, s=1, device=device, dtype=dtype),
                )
            ),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10, device=device, dtype=dtype),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        seq_len=40,
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(cifar10_train_dataset[1][0].shape)
