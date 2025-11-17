"""The module."""

import math
from typing import List, Literal

import numpy as np

import needle.init as init
from needle import ops
from needle.autograd import Tensor

from .nn_basic import Module, Parameter


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity:

          h' = tanh(x @ W_ih + b_ih + h @ W_hh + b_hh)

        Parameters:
          input_size (int):
            The number of expected features in the input X
          hidden_size (int):
            The number of features in the hidden state h
          bias (bool):
            If False, then the layer does not use bias weights
          nonlinearity (Literal["tanh", "relu"]):
            The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
          W_ih:
            The learnable input-hidden weights of shape
            (input_size, hidden_size).
          W_hh:
            The learnable hidden-hidden weights of shape
            (hidden_size, hidden_size).
          bias_ih:
            The learnable input-hidden bias of shape (hidden_size,).
          bias_hh:
            The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k =
        1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Initialize weights & biases
        self.has_bias = bias
        I, H = input_size, hidden_size
        bound = 1 / math.sqrt(hidden_size)

        W_ih = init.rand(I, H, low=-bound, high=bound, device=device, dtype=dtype)
        W_hh = init.rand(H, H, low=-bound, high=bound, device=device, dtype=dtype)
        if bias:
            bias_ih = init.rand(H, low=-bound, high=bound, device=device, dtype=dtype)
            bias_hh = init.rand(H, low=-bound, high=bound, device=device, dtype=dtype)

        self.W_ih = Parameter(W_ih)
        self.W_hh = Parameter(W_hh)
        self.bias_ih = Parameter(bias_ih) if bias else None
        self.bias_hh = Parameter(bias_hh) if bias else None

        self.nonlinearity = ops.Tanh() if nonlinearity == "tanh" else ops.ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        W_ih, W_hh = self.W_ih, self.W_hh
        B, _ = X.cached_data.shape
        H, _ = W_hh.cached_data.shape
        if h is None:
            h = init.zeros(B, H, device=X.device, dtype=X.dtype, requires_grad=False)

        Z = X @ W_ih + h @ W_hh
        if self.has_bias:
            bias = self.bias_ih + self.bias_hh
            Z += ops.broadcast_to(ops.reshape(bias, shape=(1, H)), shape=(B, H))
        return self.nonlinearity(Z)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN (with tanh or ReLU non-linearity) to an input
        sequence.

        Args:
          input_size (int):
            The number of expected features in the input x.
          hidden_size (int):
            The number of features in the hidden state h.
          num_layers (int):
            Number of recurrent layers.
          nonlinearity (Literal["tanh", "relu"]):
            The non-linearity to use. Can be either 'tanh' or 'relu'.
          bias (bool):
            If False, then the layer does not use bias weights.

        Variables:
          rnn_cells[k].W_ih:
            The learnable input-hidden weights of the k-th layer.
              k = 0: shape is (input_size, hidden_size).
              k > 0: shape is (hidden_size, hidden_size).

          rnn_cells[k].W_hh:
            The learnable hidden-hidden weights of the k-th layer, of shape
            (hidden_size, hidden_size).

          rnn_cells[k].bias_ih:
            The learnable input-hidden bias of the k-th layer, of shape
            (hidden_size,).

          rnn_cells[k].bias_hh:
            The learnable hidden-hidden bias of the k-th layer, of shape
            (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size=input_size if k == 0 else hidden_size,
                hidden_size=hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for k in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:

          X of shape (seq_len, bs, input_size) containing the features of the
          input sequence.

          h0 of shape (num_layers, bs, hidden_size) containing the initial hidden
          state for each element in the batch. Defaults to zeros if not provided.

        Outputs:

          output of shape (seq_len, bs, hidden_size) containing the output
          features (h_t) from the last layer of the RNN, for each t.

          h_n of shape (num_layers, bs, hidden_size) containing the final hidden
          state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, _, _ = X.cached_data.shape
        Xs = ops.split(X, axis=0)
        h0s = ops.split(h0, axis=0) if h0 is not None else (None,) * self.num_layers

        # Run the sequence `Xs` through the RNN for each layer
        output = [None for _ in range(seq_len)]
        h_n = []
        for layer_i, rnn in enumerate(self.rnn_cells):
            input_seq = Xs if layer_i == 0 else output
            for j in range(seq_len):
                h_prev = h0s[layer_i] if j == 0 else output[j - 1]
                output[j] = rnn(input_seq[j], h_prev)

            # Save the last hidden state for layer `i`
            h_n.append(output[-1])

        output = ops.stack(tuple(output), axis=0)
        h_n = ops.stack(tuple(h_n), axis=0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
