import sys
from typing import Literal, Tuple

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
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 1,
        seq_model: Literal["rnn", "lstm", "transformer"] = "rnn",
        seq_len: int = 40,
        device=None,
        dtype="float32",
        # --- Transformer-only parameters ---
        num_head: int | None = None,
        dim_head: int | None = None,
        dropout: float | None = None,
        causal: bool | None = None,
    ):
        """
        Consists of
          an embedding layer,
          a sequence model (either RNN, LSTM or Transformer), and
          a linear layer ("LM head").

        Parameters:
          output_size (int): Size of the dictionary, or number of tokens.
          embedding_size (int): Embedding dimension.
          hidden_size (int): Number of features in the hidden state of LSTM/RNN.
          seq_model (Literal['rnn', 'lstm', 'transformer']): 'rnn' or 'lstm'
          num_layers (int): Number of layers in LanguageModel.
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.seq_len = seq_len

        if seq_model == "rnn":
            SeqModel = nn.RNN
        elif seq_model == "lstm":
            SeqModel = nn.LSTM
        elif seq_model == "transformer":
            SeqModel = nn.Transformer

        # --- Token embedding ---
        self.embedding = nn.Embedding(
            output_size,
            embedding_size,
            device=device,
            dtype=dtype,
        )
        # --- Language Model ---
        if seq_model in ["rnn", "lstm"]:
            self.seq_model = SeqModel(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        elif seq_model == "transformer":
            """
            Explanatory notes:
             - In the GPT-2 paper, the MLP's hidden size H (MLP = D --> H --> D)
               is given as 4 * D.
             - LanguageModel expects inputs of the shape (T, B), so
               Transformer's forward layer should expect input of shape 
               (T, B, D)
            """
            assert num_head and dim_head and dropout and causal

            self.seq_model = nn.Transformer(
                embedding_size=embedding_size,
                hidden_size=4 * embedding_size,
                num_layers=num_layers,
                num_head=num_head,
                dim_head=dim_head,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
                batch_first=False,
                sequence_len=seq_len,
            )

        # --- Projection back to probabilities (output_size/num_tokens)
        if seq_model == "transformer":
            self.linear = nn.Linear(
                embedding_size, output_size, device=device, dtype=dtype
            )
        else:
            self.linear = nn.Linear(
                hidden_size, output_size, device=device, dtype=dtype
            )
        ### END YOUR SOLUTION

    def forward(
        self,
        x: ndl.Tensor,
        h: ndl.Tensor | Tuple[ndl.Tensor] | None = None,
    ):
        """
        Given sequence (and the previous hidden state if given), returns
        probabilities of next word (along with the last hidden state from the
        sequence model).

        Args:
          x of shape (seq_len, bs)
          h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs,
            hidden_size)

        Returns (out, h)
          logits of shape (seq_len*bs, output_size)
          h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs,
            hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        _, O = self.linear.weight.cached_data.shape

        # --- Embedding: x_embed has shape (T, B, D) ---
        x_embed = self.embedding(x)
        z, h = self.seq_model(x_embed, h)

        # `z` has shape (seq_len, bs, hidden_size); reshape before/after matmul
        seq_len, bs, hidden_size = z.cached_data.shape
        z = ndl.ops.reshape(z, (seq_len * bs, hidden_size))

        logits = self.linear(z)
        # Reassemble batches into sequence
        logits = ndl.ops.reshape(logits, (-1, O))

        return logits, h
        ### END YOUR SOLUTION

    def multinomial(
        probs: np.ndarray,
    ):
        """
        Input:
          probs (np.ndarray), of shape (B,O,)

        Returns:
          out (np.ndarray), or shape (B,) where each entry is a randomly
            sampled index from 0 to O-1.
        """
        probs = probs / probs.sum(axis=1, keepdims=True)
        cumprobs = np.cumsum(probs, axis=1)
        random_vals = np.random.rand(len(probs))
        return (cumprobs < random_vals[:, None]).sum(axis=1, keepdims=True)

    def generate(
        self,
        idx: ndl.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        corpus: ndl.data.Corpus = None,
        h0: ndl.Tensor | Tuple[ndl.Tensor] | None = None,
        # top_k=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time.

        Most likely you'll want to make sure to be in model.eval() mode of
        operation for this.

        Args:
          idx (Tensor): of shape (T, B)

        -----------------------------------------------------------------------
        Modified from:
        https://github.com/karpathy/nanoGPT/blob/master/model.py
        """
        h = h0
        device, dtype = idx.device, idx.dtype
        is_transformer = isinstance(self.seq_model, nn.Transformer)

        # FOR DEMO:
        i2w = corpus.dictionary.idx2word

        idx = idx.cached_data.numpy()

        for i in idx:
            i = i.astype("int32").item()
            print(i2w[i], end=" ")

        for _ in range(max_new_tokens):
            T, B = idx.shape
            if self.seq_len:
                idx_cond = idx if T <= self.seq_len else idx[-self.seq_len :, :]
            else:
                idx_cond = idx

            idx_cond = ndl.Tensor(idx_cond, device=device, dtype=dtype)

            if is_transformer:
                logits, _ = self(idx_cond, None)
            else:
                logits, h = self(idx_cond, h)

            # logits: (T*B,O,) --> (B,T,O,)
            _, O = logits.shape
            logits = logits.reshape((min(T, self.seq_len), B, O)).transpose((0, 1))

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits.numpy()
            logits = logits[:, -1, :] / temperature
            logits = logits.reshape((B, O))

            # # Optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert logits to (normalized) probabilities
            probs = np.exp(logits)
            probs = probs / np.sum(probs, axis=(-1,), keepdims=True)

            # Sample from the distribution
            # idx_next = self.multinomial(probs)
            # idx_next = self.multinomial(probs).T

            probs = probs / probs.sum(axis=1, keepdims=True)
            cumprobs = np.cumsum(probs, axis=1)
            random_vals = np.random.rand(len(probs))

            idx_next = (cumprobs < random_vals[:, None]).sum(axis=1, keepdims=True).T

            # idx_next: has shape (B,)
            # append sampled index to the running sequence and continue
            idx = np.concatenate((idx, idx_next), axis=0)

            # FOR DEMO: print each new thing as it comes

            word = i2w[idx_next.item()]
            print(word, end=" ")

        return idx


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
