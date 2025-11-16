"""Optimization module"""

import numpy as np

import needle as ndl


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(
        self,
        params: list[ndl.nn.Parameter],
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        lr = self.lr
        wd = self.weight_decay
        beta = self.momentum
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # Compute the gradient of the (loss + regulariztion term)
            grad = p.grad.data + wd * p.data
            u = beta * self.u.get(i, 0) + (1 - beta) * grad
            self.u[i] = u.data
            p.data = p - lr * self.u[i]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        eps, lr, wd = self.eps, self.lr, self.weight_decay

        # Compute the gradient & update momentum for each parameter
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Grad is the gradient of the (loss + regularization term)
            grad = p.grad.data + wd * p.data
            u = b1 * self.m.get(i, 0) + (1 - b1) * grad
            v = b2 * self.v.get(i, 0) + (1 - b2) * grad**2

            self.m[i] = u.data
            self.v[i] = v.data

            # Unbiased momentum for u, v
            u_hat = u / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)

            # Update parameter
            p.data = p - lr * (u_hat / (v_hat**0.5 + eps))
        ### END YOUR SOLUTION
