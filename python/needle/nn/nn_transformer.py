from dataclasses import dataclass
from typing import List

import numpy as np

import needle.backend_ndarray.ndarray as ndarray
import needle.init as init
from needle import ops
from needle.autograd import Tensor

from .nn_basic import Dropout, LayerNorm1d, Linear, Module, Parameter, ReLU, Sequential
from .nn_sequence import Embedding


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # 50k bpe + 256 bytes token + 1 end token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """

    def __init__(
        self,
        *,
        dropout=0.0,
        causal: bool = False,
        device=None,
        dtype="float32",
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1
        )

        return ndarray.array(mask, device=device)

    def matmul(
        self,
        a: Tensor,
        b_transpose: Tensor,
    ):
        """
        Batched matrix multiplication.

        Args:
          a (ndarray):
            B x N x M matrix.
          b_transpose (Tensor):
            B x P x M matrix.

        Returns:
          out (ndarray):
            A x B, a B x N x P matrix.
        """
        # Reshape BNM -> BN1M
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        # Reshape BPM -> B1PM
        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        # Broadcast a: BN1M -> BNPM
        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        # Broadcast b_transpose: B1PM -> BNPM
        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        # Matrix multiply: BNP
        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function.
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=(3,)),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False,
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ):
        """
        The forward function of the MultiHeadAttention activation function.

        Input: three states q, k, v, with shape (batch_size, num_head, seq_len,
        dim_head)

        Output: the activation output `result` and attention softmax probability
        `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim
        assert queries_len == keys_values_len

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # assert q.cached_data.is_compact()
        # assert k.cached_data.is_compact()
        # assert v.cached_data.is_compact()

        B, H, T_q, T_kv = batch_size, num_head, queries_len, keys_values_len

        # probs: (B, H, T_q, T_kv)
        QK_T = self.matmul(q, k) / q_dim**0.5
        if self.causal:
            mask = self.create_causal_mask(T_q, T_kv, device=q.device)
            mask = mask.broadcast_to(QK_T.shape)
            QK_T += mask

        probs = self.dropout(self.softmax(QK_T))

        assert probs.cached_data.is_compact()
        assert probs.shape == (B, H, T_q, T_kv)

        result = self.matmul(probs, ops.transpose(v, axes=(-1, -2)))
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):
    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head

        self.q_projection = Linear(
            q_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.k_projection = Linear(
            k_features, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.v_projection = Linear(
            v_features, inner_dim, bias=False, device=device, dtype=dtype
        )

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal, device=device, dtype=dtype
        )

        self.out_projection = Linear(
            inner_dim, out_features, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        q,
        k=None,
        v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        B, H, D = batch_size, self.num_head, self.dim_head

        # --- Reshape Q, K, V to 2d matrices for LayerNorm and matmul ---
        Q, K, V = (u.reshape((-1, u.shape[-1])) for u in [q, k, v])

        # --- Apply pre-norm, and projection (W_{q,k,v}) ---
        Q = self.q_projection(self.prenorm_q(Q))
        K = self.k_projection(self.prenorm_k(K))
        V = self.v_projection(self.prenorm_v(V))

        # (B*T,HD) --> (B,T,H,D) --> (B,H,T,D)
        Q = Q.reshape((B, queries_len, H, D)).transpose((1, 2))
        K = K.reshape((B, keys_values_len, H, D)).transpose((1, 2))
        V = V.reshape((B, keys_values_len, H, D)).transpose((1, 2))

        # --- Compute multi-head attention activation ---
        X, _ = self.attn(Q, K, V)

        # --- Reshape X: (B,H,T_q,D) --> (B,T_q,H,D) --> (B,T_q,HD) ---
        X = X.transpose((1, 2)).reshape((B, queries_len, H * D))

        # --- Project back to the input space of the layer ---
        # Convert to/from 2d for matmul
        X = X.reshape((B * queries_len, H * D))
        result = self.out_projection(X)
        result = result.reshape((B, queries_len, self.out_features))
        return result


class TransformerLayer(Module):
    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout=0.0,
        causal=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attn = Sequential(
            AttentionLayer(
                q_features=q_features,
                num_head=num_head,
                dim_head=dim_head,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
            ),
            Dropout(p=dropout),
        )
        self.mlp = Sequential(
            LayerNorm1d(q_features, device=device, dtype=dtype),
            Linear(q_features, hidden_size, bias=True, device=device, dtype=dtype),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_size, q_features, bias=True, device=device, dtype=dtype),
            Dropout(p=dropout),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        """
        The forward function of a Transformer Layer.

        Input:
          x (Tensor): the hidden states from previous layers, with shape
            (B, T, D)

        Ouput:
          out (Tensor): the hidden states after the Transformer Layer, with shape
            (B, T, D)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        B, T, D = batch_size, seq_len, x_dim

        x = x + self.attn(x)

        # (B,T,D) --> (B*T,D)
        mlp_residual = x.reshape((B * T, D))
        mlp_residual = self.mlp(mlp_residual).reshape((B, T, D))

        x = x + mlp_residual
        ### END YOUR SOLUTION

        return x


class Transformer(Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        causal=True,
        device=None,
        dtype="float32",
        batch_first=False,
        sequence_len=2048,
    ):
        """
        Inputs:
          embedding_size (int): D, the embedding dim. of each token
          hidden_size (int): H, the MLP maps from D --> H --> D
          num_layers (int): number of TransformerLayer layers
          num_head (int): number of attention heads
          dim_head (int): dimension of each attention head
          dropout (float): probability of dropout in both attention & MLP
            sub-layers
          causal (bool): whether to do causal masking
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.pos_embedding = Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype,
        )

        self.transformer_layers = Sequential(
            *[
                TransformerLayer(
                    q_features=embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        ### END YOUR SOLUTION

    def forward(
        self,
        x: Tensor,
        h: Tensor = None,  # for compatibility with other LanguageModel's
    ):
        """
        Adds positional embedding to x, then runs through transformer layers.

        Inputs:
          x (Tensor): Input embedded sequence, with shape (T, B, D) where
            D is the embedding dimension.
          h (Tensor): Dummy 'hidden state' — not used. To match the function
            signature of RNN's and LSTM's.

        Outputs:
          x (Tensor): Transformed output, with shape (T, B, D).
          h (Tensor): Dummy 'hidden state' — not used (see above).
        """
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        B, T, D = x.shape
        assert D == self.pos_embedding.embedding_dim

        # --- Positionally embed timestep id's ---
        # (T,) --> (T,B)
        timestep_ids = np.arange(T, dtype=np.int32)
        timestep_ids = np.broadcast_to(np.reshape(timestep_ids, (T, 1)), (T, B))
        timestep_tensor = Tensor(
            timestep_ids,
            device=x.device,
            dtype="float32",
            requires_grad=False,
        )

        # (T, B) --> (T, B, D) --> (B, T, D)
        pos_emb = self.pos_embedding(timestep_tensor).transpose((0, 1))

        # --- Add positional embedding ---
        x = x + pos_emb

        # --- Run through Transformer layers ---
        x = self.transformer_layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
