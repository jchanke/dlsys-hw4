from typing import Any, Optional, Union

from ..autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value
from ..backend_selection import BACKEND, array_api
from .ops_mathematic import *


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # Assuming Z is 2-dim; & we softmax over axis=1
        self.axis = (1,)

        # Take max across self.axes
        M = array_api.max(Z, axis=self.axis, keepdims=True)

        # self.Z := Z - max(Z)
        self.Z = Z - array_api.broadcast_to(M, Z.shape)

        # return (Z - max(Z)) - log_sum_exp(Z - max(Z))
        log_sum_exp = array_api.log(
            array_api.sum(
                array_api.exp(self.Z),
                self.axis,
                keepdims=True,
            )
        )
        return self.Z - array_api.broadcast_to(log_sum_exp, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        batch, num_labels = node.shape
        softmax = exp(node)
        softmax = reshape(softmax, (batch, num_labels, 1))
        softmax = broadcast_to(softmax, (batch, num_labels, num_labels))

        # make m copies of the identity matrix
        I = array_api.eye(num_labels)
        I = array_api.reshape(I, (1, num_labels, num_labels))
        I = array_api.broadcast_to(I, (batch, num_labels, num_labels))
        I = Tensor(I, requires_grad=False)

        # make n copies of the out_grad matrix
        out_grad_stack = reshape(out_grad, (batch, 1, num_labels))
        out_grad_stack = broadcast_to(out_grad_stack, (batch, num_labels, num_labels))

        return summation((I - softmax) * out_grad_stack, axes=(2,))
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes
        if axes is not None and not isinstance(axes, tuple):
            self.axes = (axes,)

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # Take max across self.axes
        M = array_api.max(Z, axis=self.axes, keepdims=True)

        # self.Z := Z - max(Z)
        self.Z = Z - array_api.broadcast_to(M, Z.shape)

        # Compute LogSumExp; remembering to re-add M
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(self.Z), self.axes))
        return log_sum_exp + array_api.reshape(M, log_sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_Z = exp(Tensor(self.Z, device=node.device))
        sum_exp_Z = summation(exp_Z.detach(), axes=self.axes)

        # Reshape & broadcast: go from (smaller) output to (bigger) input shape
        shape = list(self.Z.shape)
        axes = self.axes if self.axes else range(len(shape))
        for i in axes:
            shape[i] = 1
        return (
            broadcast_to(reshape(out_grad / sum_exp_Z, tuple(shape)), self.Z.shape)
            * exp_Z
        )
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
