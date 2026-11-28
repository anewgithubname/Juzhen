"""Mistakes that used to give a wrong result without an error.

Each test is one of them: what the user writes, and that it now either
works or stops with a message that names the way out. The last class is
about calls that did stop, but with the binding's list of C++ signatures.
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


class TestSentencesNotSignatures:
    """Calls that the binding refuses answer with a sentence, not with nanobind's list of C++ overloads."""

    @staticmethod
    def refused(call, error, match):
        with pytest.raises(error, match=match) as info:
            call()
        assert "incompatible function arguments" not in str(info.value)

    def test_functions_given_something_else(self, device, rng):
        a = rng.standard_normal((2, 3)).astype(np.float32)
        A = jz.array(a, device=device)
        for name in ("exp", "log", "tanh", "sin", "cos", "sqrt", "square", "abs", "relu", "sigmoid", "sum", "mean",
                     "max", "min", "argmax", "softmax", "logsumexp"):
            f = getattr(jz, name)
            self.refused(lambda: f(a), TypeError, rf"jz\.{name} takes a juzhen matrix, got a numpy array.*jz\.array\(a\)")
            self.refused(lambda: f(2.0), TypeError, rf"jz\.{name} takes a juzhen matrix, got the number 2\.0")
            self.refused(lambda: f(None), TypeError, rf"jz\.{name} takes a juzhen matrix, got None")
        self.refused(lambda: jz.topk(a, 2), TypeError, r"jz\.topk takes a juzhen matrix, got a numpy array")
        self.refused(lambda: jz.exp([1.0, 2.0]), TypeError, "got a list")
        self.refused(lambda: jz.exp(nn.mse(A, A)), TypeError, r"got a Loss.*loss\.value")
        close(jz.exp(m=A), np.exp(a))

    def test_axis(self, device, rng):
        _, A = rng, jz.randn(2, 3, device=device)
        for name in ("sum", "mean", "max", "min"):
            f = getattr(jz, name)
            for axis in (1.0, (0, 1), "rows"):
                self.refused(lambda: f(A, axis), ValueError, rf"jz\.{name}: axis must be 0 .*, 1 .* or None")
            assert f(A, np.int64(1)).shape == (2, 1)
        for name in ("argmax", "softmax", "logsumexp"):
            self.refused(lambda: getattr(jz, name)(A, 1.5), ValueError, rf"jz\.{name}: axis must be 0 .* or 1 ")
        self.refused(lambda: jz.softmax(A, None), ValueError, "got None")
        self.refused(lambda: jz.max(A, 0 * A), ValueError, r"got a matrix.*B \+ jz\.relu\(A - B\)")  # np.maximum
        self.refused(lambda: jz.min(A, 0.0), ValueError, r"got 0\.0.*B - jz\.relu\(B - A\)")
        self.refused(lambda: jz.topk(A, "2"), ValueError, "k must be a whole number")
        self.refused(lambda: jz.topk(A, 2, largest="no"), TypeError, "largest must be True or False")
        self.refused(lambda: jz.topk(A, 2, axis=None), ValueError, r"jz\.topk: axis must be")
        assert jz.topk(A, 2.0)[1].shape == (2, 2)

    def test_manual_seed(self):
        for seed in (2 ** 40, 1.5, "0", -1, True):
            self.refused(lambda: jz.manual_seed(seed), ValueError, "whole number from 0 to 2147483647")
        jz.manual_seed(np.int64(3))

    def test_operators(self, device):
        a = np.arange(6, dtype=np.float32).reshape(2, 3) + 1
        A = jz.array(a, device=device)
        for call in (lambda: A + None, lambda: None + A, lambda: A + "x", lambda: A - {}, lambda: A @ None,
                     lambda: A ** "x", lambda: 2 ** A):
            self.refused(call, TypeError, "unsupported operand")   # Python's own sentence
        self.refused(lambda: A @ 2, TypeError, r"@ is the matrix product.*A \* 2")
        self.refused(lambda: 2 @ A, TypeError, "@ is the matrix product")
        self.refused(lambda: A ** A, TypeError, r"the power must be a number.*jz\.exp\(B \* jz\.log\(A\)\)")
        self.refused(lambda: A ** 3, ValueError, r"only the powers 0\.5, 1 and 2.*jz\.exp\(p \* jz\.log\(A\)\)")
        close(A ** 2, a ** 2)
        close(A ** np.float32(0.5), a ** 0.5)
        close(A + 2, a + 2)
        close(2 + A, a + 2)
        close(A * np.int64(3), a * 3)
        close(np.float32(2) * A, a * 2)

        def scaled(op, other):
            B = A.copy()
            if op == "*":
                B *= other
            else:
                B /= other
            return B

        self.refused(lambda: scaled("*", A), TypeError, r"A \*= B changes A in place for a number only.*A = A \* B")
        self.refused(lambda: scaled("/", A[0:1]), TypeError, r"A /= B changes A in place.*A = A / B")
        self.refused(lambda: scaled("*", a), TypeError, r"jz\.array\(a, device=")
        self.refused(lambda: scaled("*", None), TypeError, "unsupported operand")
        close(scaled("*", 2), a * 2)
        close(scaled("/", np.float32(2)), a / 2)
        close(scaled("*", jz.sum(A)), a * a.sum())      # a 1 x 1 matrix counts as its number
        loss = nn.mse(A, A)
        self.refused(lambda: A * loss, TypeError, "a Loss is not a number in arithmetic")
        other = "gpu" if device == "cpu" else "cpu"
        if other in ("cpu",) or jz.has_gpu():
            B = A.to(other)
            for call in (lambda: A + B, lambda: A @ B.T, lambda: A ** B, lambda: scaled("*", B)):
                self.refused(call, ValueError, "different devices")

    def test_indexing(self, device):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        A = jz.array(a, device=device)
        allowed = r"indexed with whole numbers, slices or lists of whole numbers.*A\[\[0, 2\]\]; got "
        self.refused(lambda: A[None], TypeError, allowed + r"None .*A\[:, j\]")
        self.refused(lambda: A[...], TypeError, allowed + r"\.\.\. ")
        self.refused(lambda: A[..., 0], TypeError, allowed)
        self.refused(lambda: A["a"], TypeError, allowed + "a str")
        self.refused(lambda: A[1.0], TypeError, allowed + r"a float \(use int\(i\)")
        self.refused(lambda: A[np.array([0.0, 1.0])], TypeError, allowed + r"a float.*idx\.astype\(int\)")
        self.refused(lambda: A[np.array([[0], [1]])], TypeError, allowed + r"an array with more than one dimension.*ravel")
        self.refused(lambda: A[5], IndexError, "row index 5 is out of range for a matrix with 2 rows")
        self.refused(lambda: A[0, 9], IndexError, "column index 9 is out of range for a matrix with 3 columns")
        self.refused(lambda: A[-3], IndexError, "row index -3 is out of range")
        self.refused(lambda: A[0, [0, -4]], IndexError, "column index -4 is out of range")
        assert A[-1, -1] == 5.0
        close(A[np.array([1, 0])], a[[1, 0]])

    def test_parameters(self, device):
        net = nn.Sequential(nn.Dense(3, 4), nn.Linear(2), device=device)
        ps = net.parameters()
        names = [p.name for p in ps]
        assert names == ["layer0.weight", "layer0.bias", "layer1.weight", "layer1.bias"]
        assert ps[-1].name == names[-1] and ps[np.int64(1)].name == names[1] and ps[-4].name == names[0]
        assert ps["layer1.weight"].shape == (2, 4)
        self.refused(lambda: ps[99], IndexError, "parameter index 99 is out of range: the network has 4 parameters")
        self.refused(lambda: ps[-9], IndexError, "parameter index -9 is out of range")
        self.refused(lambda: ps["nope"], KeyError, "no parameter is called 'nope'; the names are 'layer0.weight'")
        for key in (slice(0, 2), None, 1.0, True):
            self.refused(lambda: ps[key], TypeError, r"takes a position .* or a name.*for p in net\.parameters\(\)")
        A = jz.randn(2, 2, device=device)
        self.refused(lambda: optim.AdamW(A), TypeError, r"AdamW takes net\.parameters\(\), got a matrix.*nn\.Embedding")
        self.refused(lambda: optim.AdamW([A]), TypeError, r"AdamW takes net\.parameters\(\), got list")

    def test_checkpoint_and_build_arguments(self, device, tmp_path):
        net = nn.Sequential(nn.Dense(3, 4), nn.Linear(2), device=device)
        path = tmp_path / "c.jz"
        for progress in (dict(epoch=-1), dict(epoch=2.5), dict(step="3"), dict(data_position=None)):
            name = next(iter(progress))
            self.refused(lambda: nn.save_checkpoint(net, path, **progress), ValueError,
                         rf"save_checkpoint: {name} must be a whole number, 0 or more")
        self.refused(lambda: nn.save_checkpoint(path, net), TypeError, "takes the network")
        self.refused(lambda: nn.load_checkpoint(path, net), TypeError, "takes the network")
        nn.save_checkpoint(net, path, epoch=2.0, step=np.int64(7))
        assert nn.load_checkpoint(net, path) == {"epoch": 2, "step": 7, "data_position": 0}
        self.refused(lambda: nn.Sequential(nn.Dense(4), device=device).build("a"), ValueError,
                     "build: in_features must be a positive whole number")
        assert nn.Sequential(nn.Dense(4), device=device).build(3.0).in_features == 3

    def test_layers_written_in_python(self, device):
        class NoCreate(nn.Layer):
            def out_features(self, n):
                return n

        class ReturnsItself(NoCreate):
            def create(self, core, n):
                return self

        class Preset(nn.Layer):  # a subclass that presets a C++ layer is fine
            def out_features(self, n):
                return 2 * n

            def create(self, core, n):
                return core.ReLUDense(2 * n, n, "auto", 0.0)

        way_out = r"cannot be written in Python yet.*first\.backward\(second\.backward\(grad\) \* df\(h\)\)"
        self.refused(lambda: nn.Sequential(nn.Dense(3, 4), NoCreate(), device=device), NotImplementedError,
                     "NoCreate does not define create.*" + way_out)
        self.refused(lambda: nn.Sequential(nn.Dense(3, 4), ReturnsItself(), device=device), TypeError,
                     r"layer 1: ReturnsItself\.create\(\) returned a ReturnsItself.*" + way_out)
        self.refused(lambda: nn.Sequential(nn.Dense(3, 4), nn.Layer(), device=device), NotImplementedError, way_out)
        net = nn.Sequential(nn.Dense(3, 4), Preset(), nn.Linear(1), device=device)
        assert net(jz.randn(5, 3, device=device)).shape == (5, 1)

    @pytest.mark.parametrize("layer", [nn.Conv2d, nn.ConvTranspose2d])
    def test_convolution_arguments(self, layer):
        name = layer.__name__
        size = "must be one positive whole number or a \\(height, width\\) pair"
        whole = "must be a positive whole number"
        for arguments, error, match in [
            (dict(image_size=7.5), ValueError, f"{name}: image_size {size}, got 7.5"),
            (dict(kernel_size="3"), ValueError, f"{name}: kernel_size {size}, got '3'"),
            (dict(kernel_size=(3, 3, 3)), ValueError, f"{name}: kernel_size {size}"),
            (dict(kernel_size=[3, "5"]), ValueError, f"{name}: kernel_size: the width {whole}, got '5'"),
            (dict(kernel_size=0), ValueError, f"{name}: kernel_size {size}, got 0"),
            (dict(kernel_size=-1), ValueError, f"{name}: kernel_size {size}, got -1"),
            (dict(kernel_size=None), ValueError, f"{name}: kernel_size {size}, got None"),
            (dict(in_channels=0), ValueError, f"{name}: in_channels {whole}, got 0"),
            (dict(in_channels=-1), ValueError, f"{name}: in_channels {whole}, got -1"),
            (dict(in_channels=2.5), ValueError, f"{name}: in_channels {whole}, got 2.5"),
            (dict(out_channels="8"), ValueError, f"{name}: out_channels {whole}, got '8'"),
            (dict(out_channels=0), ValueError, f"{name}: out_channels {whole}, got 0"),
            (dict(padding=1.5), ValueError, f"{name}: padding must be a whole number, 0 or more, got 1.5"),
            (dict(padding=-1), ValueError, f"{name}: padding must be a whole number, 0 or more, got -1"),
            (dict(padding="same"), ValueError, f"{name}: padding .*got 'same'; it is one number.*kernel_size - 1"),
            (dict(padding=(1, 1)), ValueError, f"{name}: padding .*one number for all four sides"),
            (dict(relu="no"), TypeError, f"{name}: relu must be True or False, got 'no'"),
            (dict(relu=0), TypeError, f"{name}: relu must be True or False, got 0"),
            (dict(stride=(2, 2)), ValueError, f"{name}: stride {whole}"),
            (dict(batch_size=0), ValueError, f"{name}: batch_size {whole}, got 0"),
        ]:
            given = dict(in_channels=1, out_channels=2, kernel_size=3, image_size=8, batch_size=4)
            given.update(arguments)
            self.refused(lambda: layer(**given), error, match)
        # whole numbers in other clothes are taken: 28 / 2, numpy integers, pairs
        made = layer(1.0, np.int64(2), 3.0, 28 / 2, np.int64(4), padding=np.int64(1), relu=np.bool_(False))
        assert (made.c_in, made.c_out, made.kh, made.kw, made.h, made.w, made.rows, made.padding, made.relu) == \
            (1, 2, 3, 3, 14, 14, 4, 1, False)
        assert all(type(v) is int for v in (made.c_in, made.c_out, made.kh, made.h, made.rows, made.padding))
        assert layer(1, 2, (3, 5), [8, 6], 4).in_features == 48

    def test_convolutions_that_do_not_fit_together(self, device):
        C = nn.Conv2d
        try:
            net = nn.Sequential(C(1, 2, 3.0, np.int64(8), 4, padding=1), C(2, 4, (3, 3), [8, 8], 4), nn.Linear(3),
                                device=device)
        except ValueError as e:  # no convolution for this device in this build
            other = "gpu" if device == "cpu" else "cpu"
            if jz.has_gpu():
                assert f"this build has its convolutions on '{other}': use nn.Sequential(..., device='{other}')" in str(e)
            pytest.skip(str(e))
        assert net(jz.randn(4, 64, device=device)).shape == (4, 3)
        gives = r"takes {} images \(channels x height x width, {} features\), but the layer before gives {} \({} features\)"
        self.refused(lambda: nn.Sequential(C(1, 2, 3, 8, 4), C(3, 4, 3, 6, 4), device=device), ValueError,
                     gives.format("3 x 6 x 6", 108, "2 x 6 x 6", 72))      # channels
        self.refused(lambda: nn.Sequential(C(1, 2, 3, 8, 4), C(2, 4, 3, 8, 4), device=device), ValueError,
                     gives.format("2 x 8 x 8", 128, "2 x 6 x 6", 72))      # image size
        # the feature counts agree (8 x 14 x 14 = 2 x 28 x 28), the images do not: this used to be accepted
        self.refused(lambda: nn.Sequential(C(1, 8, 3, 28, 4, padding=1, stride=2), C(2, 16, 3, 28, 4, padding=1),
                                           device=device), ValueError,
                     gives.format("2 x 28 x 28", 1568, "8 x 14 x 14", 1568))
        self.refused(lambda: nn.Sequential(nn.Dense(10, 50), C(1, 2, 3, 8, 4), device=device), ValueError,
                     r"takes 1 x 8 x 8 images \(channels x height x width, 64 features\), but gets 50 features")
        up = nn.Sequential(C(1, 2, 3, 8, 4, stride=2, padding=1), nn.ConvTranspose2d(2, 1, 4, 4, 4, stride=2, padding=1),
                           device=device)
        assert up(jz.randn(4, 64, device=device)).shape == (4, 64)


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
