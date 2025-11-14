"""Operator implementations."""

from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
from ..backend_selection import BACKEND, array_api
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (x,) = node.inputs
        n = self.scalar
        return out_grad * n * power_scalar(x, n - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs
        return (
            out_grad / y,
            -out_grad * x / (y * y),
        )
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        i, j = self.axes if self.axes else (-1, -2)
        return array_api.swapaxes(a, axis1=i, axis2=j)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, shape=self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


def _sum_along_axes(small: Tensor, big: Tensor) -> tuple[int]:
    """Given two tensors, return the axes along which we would have to sum `big`
    so that it matches the shape of `small`.

    To broadcast `small` to `big`, NumPy goes along the shape of `small`
    right-to-left, checking if the axis has length 1 and if the corresponding
    axis of `big` has length n > 1 (in which case it makes n copies). If `big`
    has a larger dimension (more axes) than `small`, then `small`'s shape is
    effectively padded with 1's.

    To reverse this operation, we just need to go from right-to-left again,
    checking for the same conditions.
    """
    small_dim = len(small.shape)
    big_dim = len(big.shape)
    return tuple(
        -j
        for j in range(1, big_dim + 1)
        if j > small_dim or small.shape[-j] != big.shape[-j]
    )


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, shape=self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return reshape(summation(out_grad, axes=_sum_along_axes(a, out_grad)), a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if axes is not None and not isinstance(axes, tuple):
            self.axes = (axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        shape_list = list(a.shape)
        axes = self.axes if self.axes else range(len(shape_list))
        for i in axes:
            shape_list[i] = 1
        return broadcast_to(reshape(out_grad, tuple(shape_list)), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        X_adjoint = matmul(out_grad, transpose(W))
        W_adjoint = matmul(transpose(X), out_grad)
        X_sum_over_axes = tuple(range(-len(X_adjoint.shape), -len(X.shape)))
        W_sum_over_axes = tuple(range(-len(W_adjoint.shape), -len(W.shape)))

        return (
            reshape(summation(X_adjoint, axes=X_sum_over_axes), X.shape),
            reshape(summation(W_adjoint, axes=W_sum_over_axes), W.shape),
        )
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        return out_grad * Tensor(a.cached_data > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        (a,) = node.inputs
        cosh = (exp(a) + exp(-a)) / 2
        return out_grad / cosh**2
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.

        Parameters:
          axis (int): dimension to concatenate along

        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: Tuple[NDArray]) -> Tensor:
        ### BEGIN YOUR SOLUTION
        first = args[0]

        shape = first.shape[: self.axis] + (len(args),) + first.shape[self.axis :]
        ndim = len(shape)

        out = array_api.empty(shape=shape, device=first.device)
        for i, x in enumerate(args):
            slices = tuple(i if j == self.axis else slice(None) for j in range(ndim))
            out[slices] = x
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors. (The "inverse" of
        Stack.)

        Parameters:
          axis (int): dimension to split
        """
        self.axis = axis

    def compute(self, A) -> TensorTuple:
        ### BEGIN YOUR SOLUTION
        n = self.axis
        shape = A.shape[:n] + A.shape[n + 1 :]
        tensors = []
        for i in range(A.shape[n]):
            indices = tuple(i if j == n else slice(None) for j in range(len(A.shape)))
            t = array_api.reshape(A[indices].compact(), shape=shape)
            tensors.append(t)
        return tuple(tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: int | Tuple[int] | None = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: int | tuple[int], dilation: int):
        # Coerce `axes` to a tuple
        assert axes is not None, "undefined behavior when axes=None"
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ds = tuple(self.dilation + 1 if i in self.axes else 1 for i in range(a.ndim))
        new_shape = tuple(s * d for s, d in zip(a.shape, ds))
        out = array_api.full(
            shape=new_shape,
            fill_value=0,
            dtype=a.dtype,
            device=a.device,
        )

        idxs = tuple(slice(None, None, d) for d in ds)
        out[idxs] = a

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        # Coerce `axes` to a tuple
        assert axes is not None, "undefined behavior when axes=None"
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ds = tuple(self.dilation + 1 if i in self.axes else 1 for i in range(a.ndim))
        idxs = tuple(slice(None, None, d) for d in ds)
        return a[idxs]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
