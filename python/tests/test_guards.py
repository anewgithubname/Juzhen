"""Mistakes that used to give a wrong result without an error.

Each test is one of them: what the user writes, and that it now either
works or stops with a message that names the way out.
"""

import pathlib
import re

import numpy as np
import pytest
from conftest import close

import juzhen as jz
from juzhen import nn, optim


class TestComparisons:
    def test_elementwise_against_numpy(self, device, rng):
        a = rng.integers(0, 3, (4, 5)).astype(np.float32)
        b = rng.integers(0, 3, (4, 5)).astype(np.float32)
        A, B = jz.array(a, device=device), jz.array(b, device=device)
        for op in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"):
            close(getattr(A, op)(B), getattr(a, op)(b))            # two matrices
            close(getattr(A, op)(1), getattr(a, op)(1))            # a matrix and a number
            close(getattr(A.T, op)(B.T), getattr(a.T, op)(b.T))    # views
            close(getattr(A, op)(B[0:1]), getattr(a, op)(b[0:1]))  # broadcasting a row
            close(getattr(A, op)(B[:, 0:1]), getattr(a, op)(b[:, 0:1]))
        close(1 < A, 1 < a)                    # the number on the left
        close(np.float32(1) >= A, 1 >= a)      # a numpy number on either side
        close(A == np.int64(2), a == 2)
        close((A > 1) * A, (a > 1) * a)        # the result is a matrix of 1s and 0s
        assert jz.mean(A == B).item() == pytest.approx(np.mean(a == b))

    def test_nan_and_inf(self, device):
        A = jz.array([[np.nan, np.inf, 1.0]], device=device)
        close(A == A, [[0, 0, 1]])  # nan as in numpy; equal infinities differ (the difference is nan)
        close(A == np.inf, [[0, 1, 0]])
        close(A != A, [[1, 1, 0]])

    def test_other_types(self, device):
        A = jz.ones(2, 2, device=device)
        assert (A == None) is False and (A != None) is True and (A == "x") is False  # noqa: E711
        assert A in [A] and hash(A) == hash(A)
        with pytest.raises(TypeError, match="not supported"):
            A < "x"
        with pytest.raises(ValueError, match="cannot be broadcast together for =="):
            A == jz.ones(3, 3, device=device)
        if jz.has_gpu():
            with pytest.raises(ValueError, match="different devices"):
                A == A.to("gpu" if device == "cpu" else "cpu")

    def test_truth_value(self, device):
        A, B = jz.ones(2, 2, device=device), jz.ones(2, 2, device=device)
        with pytest.raises(ValueError, match=r"\(A == B\)\.numpy\(\)\.all\(\)"):
            bool(A == B)  # was: False, because == compared the objects
        with pytest.raises(ValueError):
            if A:  # was: always True
                pass
        assert jz.sum(A) == 4 and not jz.sum(A) == 5  # a 1 x 1 matrix is one True or False
        assert (A == B).numpy().all()

    def test_a_matrix_against_a_numpy_array_is_refused(self, device):
        # an n x 1 matrix of class numbers against (n,) labels used to be compared as an n x n table
        scores = jz.array(np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1, 2]], device=device)
        y = np.array([0, 1, 2, 0, 1, 0])  # 5 of 6 right
        predicted = jz.array(scores.numpy().argmax(1), device=device)  # n x 1
        for compare in (lambda: predicted == y, lambda: y == predicted, lambda: predicted != y,
                        lambda: y < predicted, lambda: predicted == list(y)):
            with pytest.raises(TypeError, match=r"jz\.array\(a, device=.*nn\.error_rate"):
                compare()
        assert jz.mean(predicted == jz.array(y, device=device)) == pytest.approx(5 / 6)
        assert np.mean(jz.argmax(scores) == y) == pytest.approx(5 / 6)  # positions are numpy integers
        assert 1 - nn.error_rate(scores, y).item() == pytest.approx(5 / 6)

    def test_no_masks(self, device):
        A = jz.array([[1.0, 2.0], [3.0, 4.0]], device=device)
        with pytest.raises(TypeError, match="masks are not supported"):
            A[A > 2]

    def test_loss_equals_number(self, device):
        P = jz.ones(2, 2, device=device)
        zero = nn.mse(P, P)
        assert zero == 0 and not zero != 0 and zero == nn.mse(P, P) and zero != 1 and (zero == "x") is False


class TestEmptyMatrices:
    @pytest.mark.parametrize("make", ["zeros", "ones", "randn", "rand"])
    def test_refused(self, device, make):
        for rows, cols in [(0, 3), (3, 0), (-1, 3), (2.5, 3)]:
            with pytest.raises(ValueError, match="positive whole number"):
                getattr(jz, make)(rows, cols, device=device)
        assert getattr(jz, make)(np.int64(2), 3.0, device=device).shape == (2, 3)

    def test_shape_as_one_argument(self):
        with pytest.raises(TypeError, match=r"write jz\.zeros\(2, 3\)"):
            jz.zeros((2, 3))
        with pytest.raises(TypeError, match=r"jz\.ones\(3, 1\)"):
            jz.ones(3)


class TestFitChecksItsData:
    def data(self, rng):
        X = rng.standard_normal((64, 5)).astype(np.float32)
        return X, X[:, :1].copy()

    def test_nan_in_x_with_relu(self, device, rng):
        # relu(nan) is 0: this used to train to the end with finite losses and nan weights
        X, y = self.data(rng)
        X[3, 2] = np.nan
        net = nn.Sequential(nn.Dense(16, "relu"), nn.Linear(1), device=device)
        with pytest.raises(ValueError, match="fit: X holds nan at row 3, column 2"):
            net.fit(X, y, loss="mse", verbose=False)
        with pytest.raises(ValueError, match="fit: X holds nan"):
            net.fit(jz.array(X, device=device), y, loss="mse", verbose=False)

    def test_inf_in_y_and_evaluate(self, device, rng):
        X, y = self.data(rng)
        net = nn.Sequential(nn.Dense(16, "relu"), nn.Linear(1), device=device)
        net.fit(X, y, loss="mse", verbose=False)
        y[5, 0] = np.inf
        y[7, 0] = np.nan
        with pytest.raises(ValueError, match=r"evaluate: Y holds inf at row 5, column 0 \(2 such values"):
            net.evaluate(X, y, loss="mse")

    def test_large_finite_values_pass(self):
        # the check sums the data: in float64, so that large float32 values do not overflow it
        X = np.full((64, 4), 3e38, np.float32)
        assert nn._finite(X, "X", "fit") is X

    def test_regression_targets_need_mse(self, device, rng):
        X, _ = self.data(rng)
        Y = rng.standard_normal((64, 3)).astype(np.float32)
        for targets in (Y, jz.array(Y, device=device)):
            net = nn.Sequential(nn.Dense(8), nn.Linear(3), device=device)
            with pytest.raises(ValueError, match="row 0 is .*For a regression pass loss='mse' to fit"):
                net.fit(X, targets, verbose=False)  # was: trained with cross-entropy
            with pytest.raises(ValueError, match="pass loss='mse' to evaluate"):
                net.evaluate(X, targets)
            assert len(net.fit(X, targets, loss="mse", verbose=False)) == 1
        with pytest.raises(TypeError, match="class numbers use Y.astype.int.; for a regression pass loss='mse'"):
            nn.Sequential(nn.Dense(8), nn.Linear(1), device=device).fit(X, Y[:, :1], verbose=False)

    def test_one_hot_and_probabilities_pass(self, device, rng):
        X, _ = self.data(rng)
        one_hot = np.eye(3, dtype=np.float32)[rng.integers(0, 3, 64)]
        smooth = one_hot * 0.9 + 0.1 / 3
        one_hot[5] = 0  # a row that counts for nothing
        for Y in (one_hot, smooth, jz.array(smooth, device=device), rng.integers(0, 3, 64)):
            net = nn.Sequential(nn.Dense(8), nn.Linear(3), device=device)
            assert len(net.fit(X, Y, verbose=False)) == 1
            assert "accuracy" in net.evaluate(X, Y)


class TestBackwardBelongsToTheLastCall:
    def make(self, device, rng):
        net = nn.Sequential(nn.Dense(4, 8, "relu"), nn.Linear(1), device=device)
        X1, X2 = (jz.array(rng.standard_normal((16, 4)), device=device) for _ in range(2))
        Y = jz.array(rng.standard_normal((16, 1)), device=device)
        return net, optim.AdamW(net.parameters(), lr=1e-3), X1, X2, Y

    def test_a_call_in_between_is_refused(self, device, rng):
        # the gradient used to be computed from what the second call kept: wrong, and no error
        net, opt, X1, X2, Y = self.make(device, rng)
        loss = nn.mse(net(X1), Y)
        net(X2)
        with pytest.raises(RuntimeError, match="earlier output.*net.predict.*jz.vstack"):
            net.backward(loss.grad)

    def test_two_losses_of_two_calls(self, device, rng):
        # a discriminator on a real and a fake batch
        net, opt, X1, X2, Y = self.make(device, rng)
        real, fake = net(X1), net(X2)
        loss_real, loss_fake = nn.mse(real, Y), nn.mse(fake, Y)
        with pytest.raises(RuntimeError, match="earlier output"):
            net.backward(loss_real.grad)
        net.backward(loss_fake.grad)  # belongs to the last call
        value, grad = nn.mse(net(X1), Y)  # unpacked losses are followed too
        net(X2)
        with pytest.raises(RuntimeError, match="earlier output"):
            net.backward(grad)

    def test_predict_in_between(self, device, rng):
        net, opt, X1, X2, Y = self.make(device, rng)
        loss = nn.mse(net(X1), Y)
        net.predict(X2)
        with pytest.raises(RuntimeError, match=r"predict\(\) ran in between"):
            net.backward(loss.grad)
        loss = nn.mse(net(X1), Y)
        net.jvp(X1, X2)
        with pytest.raises(RuntimeError, match=r"jvp\(\) ran in between"):
            net.backward(loss.grad)

    def test_what_still_works(self, device, rng):
        net, opt, X1, X2, Y = self.make(device, rng)
        net.predict(X2)                      # validation before the step
        for _ in range(3):                   # the usual loop
            opt.zero_grad()
            loss = nn.mse(net(X1), Y)
            net.backward(loss.grad)
            net.backward(loss.grad)          # twice on one call: the gradients add up
            opt.step()
        out = net(X1)
        out = net(X2)
        net.backward((out - Y) * 0.1)        # a gradient the user computed is not followed
        other = nn.Sequential(nn.Dense(1, 4), nn.Linear(1), device=device)
        loss = nn.mse(other(net(X1)), Y)     # two networks in a chain
        net.backward(other.backward(loss.grad))


class TestArguments:
    def test_adamw_takes_numpy_numbers(self, device):
        net = nn.Sequential(nn.Dense(4, 8), device=device)
        opt = optim.AdamW(net.parameters(), lr=np.float32(0.01), weight_decay=np.float64(0), eps=np.float32(1e-8),
                          betas=[np.float32(0.9), 0.999])
        assert opt.lr == pytest.approx(0.01)
        with pytest.raises(TypeError, match="lr must be a number .0 or more., got str"):
            optim.AdamW(net.parameters(), lr="0.01")
        with pytest.raises(ValueError, match="lr must be 0 or more, got -1"):
            optim.AdamW(net.parameters(), lr=-1)
        with pytest.raises(ValueError, match="eps must be more than 0"):
            optim.AdamW(net.parameters(), eps=0)
        with pytest.raises(ValueError, match="weight_decay must be 0 or more, got nan"):
            optim.AdamW(net.parameters(), weight_decay=float("nan"))
        with pytest.raises(TypeError, match="betas must be two numbers"):
            optim.AdamW(net.parameters(), betas=0.9)

    def test_embedding_sizes_are_not_init(self):
        # PositionEmbedding(seq_len, dim) used to take dim as the standard deviation
        with pytest.raises(TypeError, match=r"PositionEmbedding\(seq_len\) takes one number.*init=32"):
            nn.PositionEmbedding(16, 32)
        with pytest.raises(TypeError, match=r"Embedding\(vocab, dim\) takes two numbers.*init=16"):
            nn.Embedding(10, 8, 16)
        assert nn.PositionEmbedding(16, init=0.1).init == 0.1 and nn.Embedding(10, 8, init=0.1).init == 0.1

    def test_ignore_index(self, device, rng):
        scores = jz.array(rng.standard_normal((6, 3)), device=device)
        labels = np.array([0, 1, 2, -1, 1, 0])
        one_hot = jz.array(np.eye(3)[[0, 1, 2, 0, 1, 0]], device=device)
        weights = np.ones(6, np.float32)
        calls = (lambda **k: nn.softmax_cross_entropy(scores, k.pop("labels", labels), **k),
                 lambda **k: nn.masked_softmax_cross_entropy(scores, k.pop("labels", labels), weights, **k),
                 lambda **k: nn.error_rate(scores, k.pop("labels", labels), **k))
        for call in calls:
            call(ignore_index=np.int64(-1))
            for wrong in (1.7, None, "x", True):  # 1.7 used to be taken as 1
                with pytest.raises(TypeError, match="ignore_index must be a whole number"):
                    call(ignore_index=wrong)
            with pytest.raises(ValueError, match="has no effect on a matrix of one-hot rows"):
                call(labels=one_hot, ignore_index=0)
            call(labels=one_hot)

    def test_conv_transpose_is_named_in_its_errors(self):
        with pytest.raises(TypeError, match="ConvTranspose2d"):
            nn.ConvTranspose2d(3, 8, 3, 8, 2, output_padding=1)


def readme_blocks():
    text = (pathlib.Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_readme_code_runs(capsys):
    blocks = readme_blocks()
    assert len(blocks) == 3
    names = {"jz": jz}  # the blocks are read one after another: the first one imports jz
    for block in blocks:
        if "datasets.mnist" in block:  # needs the data set and a minute; python/examples/fit_mnist.py is the same
            continue
        exec(compile(block, "README.md", "exec"), names)
    assert "next token: 11" in capsys.readouterr().out
