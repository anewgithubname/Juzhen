"""Optimizers. The update itself runs in C++ (ml/optimizer.hpp), one call per step."""

import juzhen as jz

__all__ = ["AdamW"]


class AdamW:
    """AdamW: Adam with decoupled weight decay (biases and norms are not decayed).

    params comes from net.parameters(). Adam's moment estimates live in the
    layers, so a checkpoint of the network also saves them.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        core = jz._device(params.device)
        self.params = params
        self._lr = float(lr)
        self._impl = core.AdamW(params, self._lr, float(betas[0]), float(betas[1]),
                                float(eps), float(weight_decay))

    def zero_grad(self):
        """Clear the gradients; call before each backward()."""
        self._impl.zero_grad()

    def step(self):
        """Update every parameter that has a gradient."""
        self._impl.step()

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
