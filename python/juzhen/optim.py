"""Optimizers. The update itself runs in C++ (ml/optimizer.hpp), one call per step."""

import math

import numpy as np

import juzhen as jz

__all__ = ["AdamW"]


def _is_number(value):
    """A Python or numpy number (np.float32(0.01) is not a float, but is as good)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


class AdamW:
    """AdamW: Adam with decoupled weight decay (biases and norms are not decayed).

    params comes from net.parameters(). Adam's moment estimates live in the
    layers, so a checkpoint of the network also saves them.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if isinstance(params, jz.nn.Sequential):
            raise TypeError("AdamW takes the network's parameters: optim.AdamW(net.parameters(), lr=...)")
        if type(params).__name__ != "ParameterList":
            kind = "a matrix" if isinstance(params, jz.Matrix) else type(params).__name__
            raise TypeError(f"AdamW takes net.parameters(), got {kind}. It updates all the parameters of one "
                            "network; a matrix of your own can be trained as the table of "
                            "nn.Sequential(nn.Embedding(rows, cols))")
        for name, value, positive in (("lr", lr, False), ("weight_decay", weight_decay, False), ("eps", eps, True)):
            rule = "more than 0" if positive else "0 or more"
            if not _is_number(value):
                raise TypeError(f"AdamW: {name} must be a number ({rule}), got {type(value).__name__}: {value!r}")
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"AdamW: {name} must be {rule}, got {value}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2 or not all(_is_number(b) for b in betas):
            raise TypeError(f"AdamW: betas must be two numbers such as (0.9, 0.999), got {betas!r}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"AdamW: both betas must be at least 0 and below 1, got {tuple(betas)}")
        core = jz._device(params.device)
        self.params = params
        self._lr = float(lr)
        self._impl = core.AdamW(params, self._lr, float(betas[0]), float(betas[1]),
                                float(eps), float(weight_decay))

    def zero_grad(self):
        """Clear the gradients; call before each backward()."""
        self._impl.zero_grad()

    def step(self):
        """Update every parameter that has a gradient; returns how many were updated."""
        updated = self._impl.step()
        if not updated:
            raise RuntimeError("step(): no parameter has a gradient, so nothing was updated; call "
                               "net.backward(loss.grad) between zero_grad() and step()")
        return updated

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._impl.set_lr(float(value))
        self._lr = float(value)

    def clip_grad_norm(self, max_norm):
        """Scale the gradients so their total norm is at most max_norm; returns the norm before."""
        return self._impl.clip_grad_norm(float(max_norm))
