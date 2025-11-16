"""The module."""

from typing import Any

import numpy as np

import needle.init as init
from needle import ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        W = init.kaiming_uniform(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
        )
        self.weight = Parameter(W)
        if bias:
            # Round-about initialization because kaiming_uniform depends on the
            # `fan_in` argument to compute the {upper,lower} bounds
            b = init.kaiming_uniform(
                fan_in=out_features,
                fan_out=1,
                device=device,
                dtype=dtype,
            )
            self.bias = Parameter(ops.reshape(b, (1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        XW = X @ self.weight
        return XW + ops.broadcast_to(self.bias, XW.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, *other_dims = X.shape
        dim = 1
        for d in other_dims:
            dim *= d
        return ops.reshape(X, (batch, dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = x
        for M in self.modules:
            result = M(result)
        return result
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, num_labels = logits.shape
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        z_y = ops.summation(init.one_hot(num_labels, y) * logits, axes=(1,))
        return ops.summation(log_sum_exp - z_y) / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def _broadcast_along_batch(self, x: Tensor, batch) -> Tensor:
        """Transforms x from shape (n,) -> (1, n) -> (batch, n)."""
        return ops.broadcast_to(ops.reshape(x, (1, self.dim)), (batch, self.dim))

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, _ = x.shape
        w = self._broadcast_along_batch(self.weight, batch)
        b = self._broadcast_along_batch(self.bias, batch)

        # Training — use batch stats, update running stats
        if self.training:
            m = self.momentum
            mu = ops.summation(x, axes=(0,)) / batch
            self.running_mean.data = (1 - m) * self.running_mean + m * mu

            mu = self._broadcast_along_batch(mu, batch)
            sigma_2 = ops.summation((x - mu) ** 2, axes=(0,)) / batch
            self.running_var.data = (1 - m) * self.running_var + m * sigma_2
            stdev = self._broadcast_along_batch((sigma_2 + self.eps) ** 0.5, batch)
            return w * ((x - mu) / stdev) + b

        # Eval — use running stats
        mean = self._broadcast_along_batch(self.running_mean, batch)
        var = self._broadcast_along_batch(self.running_var, batch)
        stdev = (var + self.eps) ** 0.5

        return w * ((x - mean) / stdev) + b
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim  # number of features per example
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        w = ops.broadcast_to(ops.reshape(self.weights, (1, self.dim)), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        batch, num_features = x.shape
        mean = ops.summation(x, axes=(1,)) / num_features
        mean = ops.broadcast_to(ops.reshape(mean, (batch, 1)), (batch, num_features))
        var = ops.summation((x - mean) ** 2, axes=(1,)) / num_features
        var = ops.broadcast_to(ops.reshape(var, (batch, 1)), (batch, num_features))
        stdev = (var + self.eps) ** 0.5
        return w * ((x - mean) / stdev) + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(*x.shape, p=(1 - self.p))
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
