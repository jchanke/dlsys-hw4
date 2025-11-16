"""The module."""

import math
from typing import Any, Callable, List

import numpy as np

import needle.init as init
from needle import ops
from needle.autograd import Tensor

from .nn_basic import Module, Parameter


class Conv(Module):
    """
    Multi-channel 2D convolutional layer.

    IMPORTANT:
      Accepts inputs in NCHW format, outputs also in NCHW format.
      Only supports padding=same
      No grouped convolution or dilation.
      Only supports square kernels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        K, I, O = kernel_size, in_channels, out_channels
        W = init.kaiming_uniform(
            fan_in=K * K * I,
            fan_out=K * K * O,
            shape=(K, K, I, O),
            device=device,
            dtype=dtype,
        )
        self.weight = Parameter(W)

        self.bias = None
        if bias:
            bound = 1 / math.sqrt(K * K * I)
            b = init.rand(
                O,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
            self.bias = Parameter(b)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_NHWC = ops.permute(x, axes=(0, 2, 3, 1))
        K, O = self.kernel_size, self.out_channels
        padding = int((K - 1) / 2)
        y_NHWC = ops.conv(x_NHWC, self.weight, stride=self.stride, padding=padding)

        if self.bias:
            N, H, W, _ = y_NHWC.cached_data.shape
            bias = ops.broadcast_to(ops.reshape(self.bias, (1, 1, 1, O)), (N, H, W, O))
            y_NHWC += bias

        return ops.permute(y_NHWC, axes=(0, 3, 1, 2))
        ### END YOUR SOLUTION
