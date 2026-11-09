"""Layers, losses, AdamW and checkpoints against numpy references."""

import numpy as np
import pytest
from conftest import close

import juzhen as jz
from juzhen import nn, optim


def randomize(net, rng, scale=0.5):
    """Give every parameter N(0, scale^2) values (the default init is tiny)."""
    for p in net.parameters():
        p.assign(jz.array(scale * rng.standard_normal(p.shape), device=net.device))


def numeric_input_grad(net, x, r, eps=1e-2):
    """d/dx sum(net(x) * r) by central differences, in float64 from float32 runs."""
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(*x.shape):
        out = []
        for sign in (1, -1):
            xp = x.copy()
            xp[idx] += sign * eps
            out.append(np.sum(net.predict(jz.array(xp, device=net.device)).numpy().astype(np.float64) * r))
        grad[idx] = (out[0] - out[1]) / (2 * eps)
    return grad


class TestLayers:
    def test_linear_forward_backward(self, device, rng):
        net = nn.Sequential(nn.Linear(3, 2), device=device)
        randomize(net, rng)
        weight, bias = net.parameters()[0], net.parameters()[1]
        assert (weight.name, weight.shape, bias.name, bias.shape) == ("layer0.weight", (2, 3), "layer0.bias", (2, 1))
        W, b = weight.value.numpy(), bias.value.numpy()

        x = rng.standard_normal((5, 3)).astype(np.float32)
        y = net(jz.array(x, device=device))
        close(y, x @ W.T + b.T)

        g = rng.standard_normal((5, 2)).astype(np.float32)
        close(net.backward(jz.array(g, device=device)), g @ W)
        close(weight.grad, g.T @ x)
        close(bias.grad, g.sum(0, keepdims=True).T)

    @pytest.mark.parametrize("activation, f", [("tanh", np.tanh), ("relu", lambda z: np.maximum(z, 0))])
    def test_dense(self, device, rng, activation, f):
        net = nn.Sequential(nn.Dense(4, 3, activation=activation), device=device)
        randomize(net, rng)
        W, b = (p.value.numpy() for p in net.parameters())
        x = rng.standard_normal((6, 4)).astype(np.float32)
        close(net(jz.array(x, device=device)), f(x @ W.T + b.T))

    def test_gradcheck_deep_network(self, device, rng):
        net = nn.Sequential(nn.Dense(4, 5), nn.Dense(5, 3, activation="relu"), nn.Linear(3, 2), device=device)
        randomize(net, rng)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        r = rng.standard_normal((3, 2)).astype(np.float32)
        net(jz.array(x, device=device))
        analytic = net.backward(jz.array(r, device=device)).numpy()
        np.testing.assert_allclose(analytic, numeric_input_grad(net, x, r), rtol=2e-2, atol=2e-3)

    def test_jvp(self, device, rng):
        net = nn.Sequential(nn.Dense(4, 5), nn.Linear(5, 2), device=device)
        randomize(net, rng)
        x = rng.standard_normal((3, 4)).astype(np.float32)
        t = rng.standard_normal((3, 4)).astype(np.float32)
        eps = 1e-2
        f = lambda z: net.predict(jz.array(z, device=device)).numpy().astype(np.float64)
        numeric = (f(x + eps * t) - f(x - eps * t)) / (2 * eps)
        jvp = net.jvp(jz.array(x, device=device), jz.array(t, device=device))
        np.testing.assert_allclose(jvp.numpy(), numeric, rtol=2e-2, atol=2e-3)

    def test_layernorm(self, device, rng):
        net = nn.Sequential(nn.LayerNorm(6, batch_size=4), device=device)
        x = rng.standard_normal((4, 6)).astype(np.float32) * 3 + 1
        y = net(jz.array(x, device=device)).numpy()
        expected = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-3)
        assert [p.name for p in net.parameters()] == ["layer0.gamma", "layer0.beta"]

    @staticmethod
    def conv_net(device, relu):
        conv = nn.Conv2d(2, 3, kernel_size=3, image_size=5, batch_size=2, padding=1, relu=relu)
        try:
            return nn.Sequential(conv, device=device)
        except ValueError as e:  # no convolution for this device in this build
            pytest.skip(str(e))

    def test_conv2d_gradcheck(self, device, rng):
        # without ReLU: finite differences that cross a ReLU kink are wrong
        net = self.conv_net(device, relu=False)
        assert net.out_features == 3 * 5 * 5
        randomize(net, rng, scale=0.3)
        x = rng.standard_normal((2, 2 * 5 * 5)).astype(np.float32)
        r = rng.standard_normal((2, 3 * 5 * 5)).astype(np.float32)
        net(jz.array(x, device=device))
        analytic = net.backward(jz.array(r, device=device)).numpy()
        np.testing.assert_allclose(analytic, numeric_input_grad(net, x, r, eps=1e-3), rtol=1e-2, atol=2e-3)

    def test_conv2d_relu(self, device, rng):
        # with ReLU = without ReLU, then max(., 0), with the gradient masked
        a, b = self.conv_net(device, relu=True), self.conv_net(device, relu=False)
        for p, q in zip(a.parameters(), b.parameters()):
            value = jz.array(0.3 * rng.standard_normal(p.shape), device=device)
            p.assign(value)
            q.assign(value)
        x = jz.array(rng.standard_normal((2, 50)), device=device)
        y = b(x).numpy()
        close(a(x), np.maximum(y, 0))
        r = rng.standard_normal(y.shape).astype(np.float32)
        close(a.backward(jz.array(r, device=device)), b.backward(jz.array(r * (y > 0), device=device)).numpy())

    def test_lazy_build_and_errors(self, device):
        net = nn.Sequential(nn.Dense(8, activation="relu"), nn.Linear(2), device=device)
        assert not net.built
        with pytest.raises(RuntimeError, match="not built"):
            net.parameters()
        assert net(jz.ones(3, 5, device=device)).shape == (3, 2)
        assert net.built and net.in_features == 5 and len(net.parameters()) == 4
        with pytest.raises(ValueError, match="takes 5 features"):
            net(jz.ones(3, 4, device=device))
        with pytest.raises(ValueError, match="takes 3 features"):
            nn.Sequential(nn.Linear(4, 2), nn.Linear(3, 1))
        other = "gpu" if device == "cpu" else "cpu"
        if jz.has_gpu():
            with pytest.raises(ValueError, match="use x.to"):
                net(jz.ones(3, 5, device=other))

    def test_predict_does_not_allow_backward(self, device):
        net = nn.Sequential(nn.Linear(3, 2), device=device)
        with pytest.raises(RuntimeError, match="before backward"):
            net.backward(jz.ones(1, 2, device=device))
        net.predict(jz.ones(1, 3, device=device))
        with pytest.raises(RuntimeError, match="before backward"):
            net.backward(jz.ones(1, 2, device=device))

    def test_forward_result_is_a_copy(self, device, rng):
        net = nn.Sequential(nn.Linear(3, 2), device=device)
        x = jz.array(rng.standard_normal((4, 3)), device=device)
        y = net(x)
        saved = y.numpy()
        net(jz.ones(4, 3, device=device))  # overwrites the layer's own output
        close(y, saved)


class TestLosses:
    def test_softmax_cross_entropy(self, device, rng):
        z = rng.standard_normal((5, 4)).astype(np.float32)
        y = np.eye(4, dtype=np.float32)[rng.integers(0, 4, 5)]
        loss = nn.softmax_cross_entropy(jz.array(z, device=device), jz.array(y, device=device))
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        assert loss.item() == pytest.approx(-np.mean(np.sum(y * np.log(p), 1)), rel=1e-5)
        close(loss.grad, (p - y) / 5)

    def test_mse_and_error_rate(self, device, rng):
        a = rng.standard_normal((5, 3)).astype(np.float32)
        b = rng.standard_normal((5, 3)).astype(np.float32)
        loss = nn.mse(jz.array(a, device=device), jz.array(b, device=device))
        assert loss.item() == pytest.approx(np.sum((a - b) ** 2) / 5, rel=1e-5)
        close(loss.grad, 2 * (a - b) / 5)
        labels = nn.one_hot([0, 1, 2, 0, 1], 3, device=device)
        scores = jz.array(np.eye(3)[[0, 1, 1, 2, 1]], device=device)
        assert nn.error_rate(scores, labels).item() == pytest.approx(0.4)

    def test_one_hot(self):
        close(nn.one_hot(np.array([2, 0]), 3), [[0, 0, 1], [1, 0, 0]])
        with pytest.raises(ValueError):
            nn.one_hot([3], 3)


class TestAdamW:
    def test_one_step_matches_numpy(self, device, rng):
        net = nn.Sequential(nn.Linear(3, 2), device=device)
        randomize(net, rng)
        lr, b1, b2, eps, wd = 0.01, 0.8, 0.9, 1e-6, 0.1
        opt = optim.AdamW(net.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
        before = [p.value.numpy() for p in net.parameters()]
        x = jz.array(rng.standard_normal((4, 3)), device=device)
        opt.zero_grad()
        net.backward(nn.mse(net(x), jz.zeros(4, 2, device=device)).grad)
        grads = [p.grad.numpy() for p in net.parameters()]
        opt.step()
        for p, w, g, decay in zip(net.parameters(), before, grads, (wd, 0.0)):  # biases are not decayed
            m, v = (1 - b1) * g / (1 - b1), (1 - b2) * g * g / (1 - b2)
            close(p.value, w * (1 - lr * decay) - lr * m / (np.sqrt(v) + eps), tol=1e-4)

    def test_training_reduces_loss(self, device, rng):
        jz.manual_seed(0)
        x = rng.standard_normal((64, 3)).astype(np.float32)
        y = np.sin(x @ np.array([[1.0], [-2.0], [0.5]], np.float32))
        X, Y = jz.array(x, device=device), jz.array(y, device=device)
        net = nn.Sequential(nn.Dense(3, 16), nn.Linear(16, 1), device=device)
        opt = optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0)
        losses = []
        for _ in range(300):
            opt.zero_grad()
            loss = nn.mse(net(X), Y)
            net.backward(loss.grad)
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.1 * losses[0]
        opt.lr = 1e-3
        assert opt.lr == 1e-3
        assert opt.clip_grad_norm(1e6) > 0


class TestCheckpoint:
    def test_roundtrip(self, device, rng, tmp_path):
        def make():
            net = nn.Sequential(nn.Dense(3, 4), nn.Linear(4, 2), device=device)
            return net, optim.AdamW(net.parameters(), lr=1e-2)

        net, opt = make()
        randomize(net, rng)
        x = jz.array(rng.standard_normal((5, 3)), device=device)
        opt.zero_grad()
        net.backward(nn.mse(net(x), jz.zeros(5, 2, device=device)).grad)
        opt.step()
        path = tmp_path / "run.ckpt"
        jz.save_checkpoint(net, path, epoch=3, step=40)

        other, _ = make()
        assert not np.allclose(other.predict(x).numpy(), net.predict(x).numpy())
        assert jz.load_checkpoint(other, path) == {"epoch": 3, "step": 40, "data_position": 0}
        close(other.predict(x), net.predict(x).numpy())

    def test_wrong_architecture(self, device, tmp_path):
        net = nn.Sequential(nn.Linear(3, 2), device=device)
        jz.save_checkpoint(net, tmp_path / "a.ckpt")
        with pytest.raises(RuntimeError, match="cannot load"):
            jz.load_checkpoint(nn.Sequential(nn.Linear(4, 2), device=device), tmp_path / "a.ckpt")


class TestFit:
    def test_fit_and_evaluate(self, device, rng):
        jz.manual_seed(0)
        centers = np.array([[0, 6], [6, 0], [-6, -6]])  # well separated
        labels = rng.integers(0, 3, 600)
        X = (centers[labels] + rng.standard_normal((600, 2))).astype(np.float32)
        Y = np.eye(3, dtype=np.float32)[labels]
        model = jz.Sequential(jz.Dense(16, activation="relu"), jz.Linear(3), device=device)
        history = model.fit(X, Y, epochs=5, batch_size=20, lr=1e-2, verbose=False)
        assert history[-1] < history[0]
        result = model.evaluate(X, Y)
        assert result["accuracy"] > 0.9 and result["loss"] < history[0]
