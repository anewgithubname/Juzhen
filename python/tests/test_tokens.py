"""Token layers, class-number losses, the Loss object and the default initialisation."""

import warnings

import numpy as np
import pytest
from conftest import DEVICES, close

import juzhen as jz
from juzhen import nn, optim


def lm(device, batch=2, tied=True):
    return nn.Sequential(nn.Embedding(11, 8), nn.PositionEmbedding(4),
                         nn.Transformer(8, 8, 16, seq_len=4, batch_size=batch, heads=2), nn.LayerNorm(),
                         nn.Linear(11, tied=True) if tied else nn.Linear(11), device=device)


def test_embedding_lookup_and_gradient(device, rng):
    net = nn.Sequential(nn.Embedding(7, 3), device=device)
    table = net.parameters()[0]
    assert (table.name, table.shape) == ("layer0.weight", (7, 3))
    E = rng.standard_normal((7, 3)).astype(np.float32)
    table.assign(jz.array(E, device=device))
    ids = np.array([[2, 2], [6, 0]])
    close(net(ids), E[ids.ravel()], tol=0)                 # one row per token, row after row
    close(net(ids.ravel().tolist()), E[ids.ravel()], tol=0)
    g = rng.standard_normal((4, 3)).astype(np.float32)
    returned = net.backward(jz.array(g, device=device))
    close(returned, g, tol=0)                               # the gradient of the embedded tokens
    expected = np.zeros((7, 3), np.float32)
    np.add.at(expected, ids.ravel(), g)                     # repeated ids add up
    close(table.grad, expected, tol=1e-6)


def test_position_embedding(device, rng):
    net = nn.Sequential(nn.Linear(3, 5), nn.PositionEmbedding(4), device=device)
    W, b, P = (p.value.numpy() for p in net.parameters())
    assert net.parameters()[2].shape == (4, 5)
    x = rng.standard_normal((12, 3)).astype(np.float32)     # three sequences of four tokens
    close(net(jz.array(x, device=device)), x @ W.T + b.T + np.tile(P, (3, 1)), tol=1e-5)
    g = rng.standard_normal((12, 5)).astype(np.float32)
    net.backward(jz.array(g, device=device))
    close(net.parameters()[2].grad, g.reshape(3, 4, 5).sum(0), tol=1e-5)
    assert net.predict(jz.array(x[:4], device=device)).shape == (4, 5)   # any number of sequences


def test_tied_model_parameters_and_state(device, rng):
    net = lm(device)
    names = [p.name for p in net.parameters()]
    assert names[:2] == ["layer0.weight", "layer1.weight"] and len(names) == 17
    assert len(set(names)) == 17 and not any(n.startswith("layer4.") for n in names)
    ids = rng.integers(0, 11, (2, 4))
    expected = net.predict(ids).numpy()
    E = net.parameters()[0].value.numpy()
    untied = nn.Sequential(*net.layers[:4], device=device)   # the same layers without the head
    untied.load_state_dict(net.state_dict())
    close(untied.predict(ids).numpy() @ E.T, expected, tol=1e-4)   # logits = h @ E.T
    one = lm(device, batch=1)
    one.load_state_dict(net.state_dict())                           # the generation network
    close(one.predict(ids[1]), expected[4:], tol=1e-4)


def test_class_number_losses_match_one_hot(device, rng):
    z = jz.array(rng.standard_normal((6, 5)), device=device)
    y = rng.integers(0, 5, 6)
    Y = nn.one_hot(y, 5, device=device)
    a, b = nn.softmax_cross_entropy(z, y), nn.softmax_cross_entropy(z, Y)
    assert a.item() == pytest.approx(b.item(), rel=1e-5)
    close(a.grad, b.grad.numpy(), tol=1e-6)
    close(nn.softmax_cross_entropy(z, y.reshape(2, 3)).grad, a.grad.numpy(), tol=0)
    assert nn.error_rate(z, y).item() == pytest.approx(nn.error_rate(z, Y).item())
    w = np.array([.5, 0, 1, .25, 0, 2], np.float32)
    c, d = nn.masked_softmax_cross_entropy(z, y, w), nn.masked_softmax_cross_entropy(z, Y, w)
    assert c.item() == pytest.approx(d.item(), rel=1e-5)
    close(c.grad, d.grad.numpy(), tol=1e-6)
    skipped = np.where(np.arange(6) % 2 == 0, y, -100)
    e = nn.softmax_cross_entropy(z, skipped)
    f = nn.masked_softmax_cross_entropy(z, Y, (skipped != -100) / 3)
    assert e.item() == pytest.approx(f.item(), rel=1e-5)
    close(e.grad, f.grad.numpy(), tol=1e-6)
    nothing = nn.softmax_cross_entropy(z, np.full(6, -100))
    assert nothing.item() == 0 and not nothing.grad.numpy().any()


def test_step_trains_and_devices_agree(rng):
    ids = rng.integers(0, 11, (2, 4))
    targets = np.roll(ids, -1, axis=1)
    reference = None
    for device in DEVICES:
        net = lm(device)
        if reference is None:
            reference = {k: v.copy() for k, v in net.state_dict().items()}
        net.load_state_dict(reference)
        opt = optim.AdamW(net.parameters(), lr=1e-2, weight_decay=0)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = nn.softmax_cross_entropy(net(ids), targets)
            net.backward(loss.grad)
            assert opt.step() == 17
            losses.append(loss.item())
        assert losses[-1] < 0.3 * losses[0]
        if device == "cpu":
            cpu_losses = losses
        else:
            np.testing.assert_allclose(losses, cpu_losses, rtol=2e-3)   # the same weights, the same numbers


def test_fit_and_evaluate_with_class_numbers_and_ids(device, rng):
    jz.manual_seed(0)
    centers = np.array([[0, 6], [6, 0], [-6, -6]])
    labels = rng.integers(0, 3, 600)
    X = (centers[labels] + rng.standard_normal((600, 2))).astype(np.float32)
    model = jz.Sequential(jz.Dense(16, "relu"), jz.Linear(3), device=device)
    history = model.fit(X, labels, epochs=5, batch_size=20, lr=1e-2, verbose=False)
    result = model.evaluate(X, labels)
    assert history[-1] < history[0] and result["accuracy"] > 0.9
    same = model.evaluate(X, np.eye(3, dtype=np.float32)[labels])     # one-hot labels give the same numbers
    assert same["accuracy"] == pytest.approx(result["accuracy"]) and same["loss"] == pytest.approx(result["loss"], abs=1e-6)

    # a fixed-batch token model: 9 sequences in batches of 2, the last batch is filled up
    net = lm(device)
    Xi = rng.integers(0, 11, (9, 4))
    Yi = np.roll(Xi, -1, axis=1)
    got = net.evaluate(Xi, Yi)
    one = lm(device, batch=1)
    one.load_state_dict(net.state_dict())
    losses, errors = zip(*[(nn.softmax_cross_entropy(one.predict(x), y).item(), nn.error_rate(one.predict(x), y).item())
                           for x, y in zip(Xi, Yi)])
    assert got["loss"] == pytest.approx(np.mean(losses), rel=1e-4)
    assert got["accuracy"] == pytest.approx(1 - np.mean(errors), abs=1e-6)
    assert len(net.fit(Xi, Yi, epochs=2, verbose=False)) == 2


def test_deep_relu_network_is_alive_at_the_start(device, rng):
    jz.manual_seed(3)
    net = jz.Sequential(*[jz.Dense(256, "relu") for _ in range(8)], jz.Linear(10), device=device)
    out = net.predict(jz.array(rng.standard_normal((64, 32)), device=device)).numpy()
    assert out.std(0).mean() > 0.1            # the output depends on the input (was about 1e-10)
    weight, bias = net.parameters()[2], net.parameters()[3]
    assert weight.value.numpy().std() == pytest.approx(np.sqrt(2 / 256), rel=0.05)
    assert not bias.value.numpy().any()


def test_checkpoint_of_a_tied_model(device, rng, tmp_path):
    net, other = lm(device), lm(device, batch=1)
    ids = rng.integers(0, 11, (2, 4))
    jz.save_checkpoint(net, tmp_path / "lm.ckpt", step=7)
    assert jz.load_checkpoint(other, tmp_path / "lm.ckpt")["step"] == 7
    close(other.predict(ids[0]), net.predict(ids).numpy()[:4], tol=1e-4)
    with pytest.raises(RuntimeError, match="parameter 'weight': the file has shape 11 x 8"):
        jz.load_checkpoint(nn.Sequential(nn.Embedding(12, 8), nn.PositionEmbedding(4),
                                         nn.Transformer(8, 8, 16, seq_len=4, batch_size=2, heads=2), nn.LayerNorm(),
                                         nn.Linear(12, tied=True), device=device), tmp_path / "lm.ckpt")


def test_evaluate_counts_every_sample_once(device, rng):
    net = lm(device, batch=4)
    one = net.copy(batch_size=1)
    for n in (1, 3, 4, 5, 7):                                  # fewer than a batch, a remainder, exact
        Xi = rng.integers(0, 11, (n, 4))
        Yi = np.roll(Xi, -1, axis=1)
        got = net.evaluate(Xi, Yi)
        losses = [nn.softmax_cross_entropy(one.predict(x), y).item() for x, y in zip(Xi, Yi)]
        assert got["loss"] == pytest.approx(np.mean(losses), rel=1e-4), n


def test_copy_for_another_batch_size_and_device(device, rng):
    net = lm(device, batch=2)
    ids = rng.integers(0, 11, (2, 4))
    out = net(ids).numpy()
    one = net.copy(batch_size=1)
    assert one.device == net.device and one.batch_rows == 4 and net.batch_rows == 8
    close(one.predict(ids[0]), out[:4], tol=1e-5)
    close(one.predict(ids[1]), out[4:], tol=1e-5)
    # a copy, not a view: training the original leaves the copy alone
    before = one.state_dict()["layer0.weight"].copy()
    opt = optim.AdamW(net.parameters(), lr=1e-2)
    opt.zero_grad()
    net.backward(nn.softmax_cross_entropy(net(ids), np.roll(ids, -1, axis=1)).grad)
    opt.step()
    assert np.array_equal(one.state_dict()["layer0.weight"], before)
    assert not np.array_equal(net.state_dict()["layer0.weight"], before)
    on_cpu = net.copy(device="cpu")
    assert on_cpu.device == "cpu"
    close(on_cpu.predict(ids), net.predict(ids).numpy(), tol=1e-4)
    with pytest.raises(RuntimeError, match="not built yet"):
        nn.Sequential(nn.Dense(3), device=device).copy()
    with pytest.raises(ValueError, match=r"use net\.copy\(batch_size"):
        net(ids[0])


def test_loss_behaves_as_its_number(device, rng):
    out = jz.array(rng.standard_normal((6, 3)), device=device)
    loss = nn.softmax_cross_entropy(out, [0, 1, 2, 0, 1, 2])
    number = loss.item()
    value, grad = loss                                         # still unpacks
    assert value.shape == (1, 1) and grad.shape == (6, 3)
    assert float(loss) == number and f"{loss:.4f}" == f"{number:.4f}" and "%.3f" % loss == "%.3f" % number
    assert round(loss, 2) == round(number, 2)
    assert loss < number + 1 and loss > number - 1 and loss <= number and loss >= number
    assert repr(loss).startswith("Loss(") and "6 x 3" in repr(loss)
    for misuse in (lambda: loss + loss, lambda: 0.0 + loss, lambda: loss * 2, lambda: loss / 2, lambda: sum([loss, loss])):
        with pytest.raises(TypeError, match="loss.item"):
            misuse()
    with pytest.raises(RuntimeError, match=r"net\.backward\(loss\.grad\)"):
        loss.backward()


def test_labels_are_class_numbers_or_one_hot_matrices(device, rng):
    out = jz.array(rng.standard_normal((4, 3)), device=device)
    one_hot = np.eye(3)[[0, 2, 1, 1]]
    with pytest.raises(ValueError, match=r"argmax\(1\)"):      # whole numbers shaped like the logits
        nn.softmax_cross_entropy(out, one_hot.astype(int))
    with pytest.raises(TypeError, match="jz.array"):           # one-hot rows as a float numpy array
        nn.softmax_cross_entropy(out, one_hot.astype(np.float32))
    with pytest.raises(TypeError, match="astype"):             # class numbers stored as floats
        nn.softmax_cross_entropy(out, np.array([0.0, 2.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="subtract 1"):
        nn.softmax_cross_entropy(out, [1, 2, 3, 1])
    X = rng.standard_normal((8, 2)).astype(np.float32)
    model = jz.Sequential(jz.Dense(4, "relu"), jz.Linear(3), device=device)
    with pytest.raises(ValueError, match=r"argmax\(1\)"):
        model.fit(X, np.eye(3, dtype=int)[[0, 1, 2, 0, 1, 2, 0, 1]], verbose=False)
    with pytest.raises(TypeError, match="astype"):
        model.fit(X, np.array([0.0, 1, 2, 0, 1, 2, 0, 1]), verbose=False)
    with pytest.raises(TypeError, match="return_inverse"):
        model.fit(X, np.array(["a", "b", "c", "a", "b", "c", "a", "b"]), verbose=False)
    model.fit(X, np.array([[0], [1], [2], [0], [1], [2], [0], [1]]), batch_size=4, verbose=False)   # (N, 1) is fine


def test_step_without_a_gradient_is_an_error(device, rng):
    net = nn.Sequential(nn.Dense(2, 3), device=device)
    opt = optim.AdamW(net.parameters())
    opt.zero_grad()
    with pytest.raises(RuntimeError, match="no parameter has a gradient"):
        opt.step()
    net.backward(nn.mse(net(jz.array(rng.standard_normal((4, 2)), device=device)),
                        jz.zeros(4, 3, device=device)).grad)
    assert opt.step() == 2


def test_numpy_numbers_next_to_a_matrix(device, rng):
    values = rng.standard_normal((3, 2)).astype(np.float32)
    A = jz.array(values, device=device)
    for result, expected in ((np.float32(2) * A, 2 * values), (np.float64(1) - A, 1 - values),
                             (np.sqrt(2.0) * A, np.float32(np.sqrt(2.0)) * values), (A / np.int64(4), values / 4),
                             (np.array(3.0) + A, 3 + values)):
        assert isinstance(result, jz.Matrix)
        close(result, expected, tol=1e-6)
    with pytest.raises(TypeError, match="jz.array"):
        values + A
    np.testing.assert_allclose(np.exp(A), np.exp(values), rtol=1e-6)      # numpy functions still work
    assert np.sum(A) == pytest.approx(values.sum(), rel=1e-5)


def test_layer_arguments_are_explained(device):
    with pytest.raises(TypeError, match="no longer takes batch_size"):
        nn.LayerNorm(batch_size=32)
    with pytest.raises(ValueError, match="init must be 'auto'"):
        nn.Dense(4, init="he")
    with pytest.raises(ValueError, match="heads"):
        nn.Transformer(8, 8, 16, seq_len=4, batch_size=1, heads=0)
    with pytest.raises(ValueError, match="stride"):
        nn.Conv2d(1, 2, 3, image_size=6, batch_size=1, stride=0)
    with pytest.warns(UserWarning, match="PositionEmbedding"):
        nn.Sequential(nn.Embedding(11, 8), nn.Transformer(8, 8, 16, seq_len=4, batch_size=1), device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # with positions, and without a Transformer: quiet
        lm(device)
        nn.Sequential(nn.Embedding(11, 8), nn.Linear(11, tied=True), device=device)
