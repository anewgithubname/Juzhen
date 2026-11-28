"""Which element-by-element operations follow one another, and what one pass instead of several could save.

    python fusion_survey.py [cpu] [gpu] [--small]

Each formula below is compiled with jz.compile and timed line by line (f.profile). Lines are put in one
group when a line's result is read by exactly one later line and both work element by element (the last
line of a group may be a sum, max or min, or an in-place update): such a group could be one pass over the
matrices instead of one pass per line. For every group the script prints the expression, the time its
lines take now and an estimate of what one pass would save:

    the time of all its + - * / lines but one, when the group has no exp, log, tanh, sin or cos;
    the time of all its + - * / lines, when it has one (that pass stays and can take them in).

The estimate is an upper bound of the simple kind: it assumes a pass costs the same whatever it computes.

The formulas marked "repo" are the ones written out in this repository's examples (examples/*.cu and
python/examples); the ones marked "common" are formulas often written by hand, chosen by the author of
this script.
"""
import collections
import re
import sys

import numpy as np

import juzhen as jz
from juzhen._compile import _V

SMALL = "--small" in sys.argv
DEVICES = [d for d in sys.argv[1:] if d in ("cpu", "gpu")] or (["cpu"] + (["gpu"] if jz.has_gpu() else []))

ELEMENTWISE = {"add", "sub", "mul", "truediv", "neg", "pow", "exp", "log", "tanh", "sin", "cos", "sqrt", "square",
               "abs", "relu", "eq", "ne", "lt", "le", "gt", "ge"}
IN_PLACE = {"iadd", "isub", "imul", "itruediv"}
REDUCTIONS = {"sum", "max", "min"}
SLOW = {"exp", "log", "tanh", "sin", "cos"}  # the function takes the time, not the pass over memory


# ---- the formulas: (source, function, shapes of its matrix arguments in units of n and d, float arguments) --

def pairwise_distances(X, A):                      # examples/knn.cu, python/examples/knn.py
    return jz.sum(jz.square(X), 1) + jz.sum(jz.square(A), 1).T - 2 * (X @ A.T)


def flow_mix(x0, x1, t):                           # examples/demo_rectified.cu, python/examples/rectified_flow.py
    return x0 * (1 - t) + x1 * t, x1 - x0


def flow_loss(prediction, target):                 # examples/demo_cnn_rectified.cu
    diff = prediction - target
    return jz.sum(jz.square(diff)) / diff.shape[0]


def euler_step(z, v, dt):                          # python/examples/rectified_flow.py
    return z + v * dt


def word2vec_loss(Vc, Uo, Un):                     # examples/demo_word2vec.cu, negative sampling
    spos = jz.sum(Vc * Uo, 1)
    sneg = Vc @ Un.T
    sigp = 1 / (jz.exp(-spos) + 1)
    sigg = 1 / (jz.exp(-sneg) + 1)
    return (-jz.sum(jz.log(sigp + 1e-12)) - jz.sum(jz.log(1 - sigg + 1e-12))) / Vc.shape[0]


def word2vec_softmax_loss(Vc, Uo, Un, tau):        # examples/demo_word2vec.cu, the sampled softmax
    spos = jz.sum(Vc * Uo, 1)
    E = jz.exp((Vc @ Un.T - spos) / tau)
    Z = 1 + jz.sum(E, 1)
    return jz.sum(jz.log(Z)) / Vc.shape[0]


def density_ratio_gradient(Xp, Xq, theta):         # examples/demo.cu
    n = Xp.shape[0]
    g1 = jz.sum(Xp, 0) / n
    e = jz.exp(Xq @ theta)
    N = jz.sum(e, 0) / n
    return g1 - e.T @ Xq / N / n


def pagerank_step(M, p, damping):                  # examples/pagerank.cu
    p = damping * (M @ p) + (1 - damping) / M.shape[0]
    return p / jz.sum(p)


def softmax_by_hand(S):                            # examples/demo_transformer.cu (sampling)
    e = jz.exp(S - jz.max(S, 1))
    return e / jz.sum(e, 1)


def helloworld(A, B):                              # examples/helloworld.cu, python/examples/helloworld.py
    return jz.log(jz.exp(A @ B) + 1) / 5


def squared_error(A, B):
    return jz.sum((A - B) * (A - B))


def axpy(A, B):
    return A + 0.5 * B


def sigmoid(A):
    return 1 / (1 + jz.exp(-A))


def swish(A):
    return A * (1 / (1 + jz.exp(-A)))


def adam_by_hand(W, G, M, V, lr):
    M_new = 0.9 * M + 0.1 * G
    V_new = 0.999 * V + 0.001 * (G * G)
    W -= lr * M_new / (jz.sqrt(V_new) + 1e-8)
    return M_new, V_new


def momentum_by_hand(W, G, V, lr):
    V_new = 0.9 * V + G
    W -= lr * V_new
    return V_new


def tanh_regression_step(W, X, Y, lr):
    P = jz.tanh(X @ W)
    E = P - Y
    W -= lr * (X.T @ (E * (1 - P * P))) / X.shape[0]
    return jz.mean(E * E)


def logistic_gradient(w, X, y):
    return X.T @ (1 / (1 + jz.exp(-(X @ w))) - y) / X.shape[0]


def rbf_kernel(X, Y):
    D = jz.sum(X * X, 1) + jz.sum(Y * Y, 1).T - 2 * (X @ Y.T)
    return jz.exp(-D / (2 * 1.5 * 1.5))


def standardize(X):
    mean = jz.mean(X, 0)
    centred = X - mean
    return centred / (jz.sqrt(jz.mean(centred * centred, 0)) + 1e-6)


def gaussian_log_density(X, mu, s):
    Z = (X - mu) / s
    return -0.5 * jz.sum(Z * Z, 1) - jz.sum(jz.log(s), 1)


def cosine_similarity(A, B):
    return jz.sum(A * B, 1) / (jz.sqrt(jz.sum(A * A, 1)) * jz.sqrt(jz.sum(B * B, 1)))


def clip(A):
    return jz.relu(A + 1) - jz.relu(A - 1) - 1


# n is the large dimension, d the small one
FORMULAS = [
    ("repo", pairwise_distances, [("n", "d"), ("d", "d")], []),
    ("repo", flow_mix, [("n", "d"), ("n", "d"), ("n", 1)], []),
    ("repo", flow_loss, [("n", "d"), ("n", "d")], []),
    ("repo", euler_step, [("n", "d"), ("n", "d")], [0.05]),
    ("repo", word2vec_loss, [("n", "d"), ("n", "d"), (16, "d")], []),
    ("repo", word2vec_softmax_loss, [("n", "d"), ("n", "d"), (16, "d")], [0.5]),
    ("repo", density_ratio_gradient, [("n", "d"), ("n", "d"), ("d", 1)], []),
    ("repo", pagerank_step, [("d", "d"), ("d", 1)], [0.85]),
    ("repo", softmax_by_hand, [("n", "d")], []),
    ("repo", helloworld, [("n", "d"), ("d", "d")], []),
    ("common", squared_error, [("n", "d"), ("n", "d")], []),
    ("common", axpy, [("n", "d"), ("n", "d")], []),
    ("common", sigmoid, [("n", "d")], []),
    ("common", swish, [("n", "d")], []),
    ("common", adam_by_hand, [("n", "d"), ("n", "d"), ("n", "d"), ("n", "d")], [0.0]),
    ("common", momentum_by_hand, [("n", "d"), ("n", "d"), ("n", "d")], [0.0]),
    ("common", tanh_regression_step, [("d", 16), ("n", "d"), ("n", 16)], [0.0]),
    ("common", logistic_gradient, [("d", 1), ("n", "d"), ("n", 1)], []),
    ("common", rbf_kernel, [("d2", 64), ("d2", 64)], []),
    ("common", standardize, [("n", "d")], []),
    ("common", gaussian_log_density, [("n", "d"), (1, "d"), (1, "d")], []),
    ("common", cosine_similarity, [("n", "d"), ("n", "d")], []),
    ("common", clip, [("n", "d")], []),
]


def groups_of(lines):
    """The groups of lines that could be one pass: lists of line numbers, each ending in its last line."""
    readers = collections.defaultdict(set)
    for i, line in enumerate(lines):
        for a in line["args"]:
            if type(a) is _V:
                readers[a].add(i)
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, line in enumerate(lines):
        if line["op"] not in ELEMENTWISE or line["out"] is None or line["returned"]:
            continue
        if len(readers[line["out"]]) != 1:
            continue  # read by several lines (or by none): it has to exist as a matrix
        j, = readers[line["out"]]
        reader = lines[j]
        if reader["op"] in ELEMENTWISE or reader["op"] in REDUCTIONS:
            parent[find(i)] = find(j)
        elif reader["op"] in IN_PLACE and reader["args"][0] != line["out"]:
            parent[find(i)] = find(j)  # W -= (this): the update takes it in
    groups = collections.defaultdict(list)
    for i in range(len(lines)):
        groups[find(i)].append(i)
    return [members for members in groups.values() if len(members) > 1]


def expression(lines, members):
    """The group as one expression, its matrices named a, b, c in the order they appear and numbers s."""
    made_by = {lines[i]["out"]: i for i in members if lines[i]["out"] is not None}
    letters = {}

    def text(i):
        line, parts = lines[i], []
        for a in line["args"]:
            if type(a) is not _V:
                parts.append("s")
            elif a in made_by and made_by[a] != i:
                form = lines[made_by[a]]["form"]
                call = re.match(r"[A-Za-z_]+\(", form) and form.endswith(")")  # exp({0}) needs no brackets
                parts.append(text(made_by[a]) if call else f"({text(made_by[a])})")
            else:
                parts.append(letters.setdefault(a, "abcdefghijk"[min(len(letters), 10)]))
        return line["form"].format(*parts)

    return text(members[-1])


def estimate(lines, members, ms):
    """(the time of the group's lines now, what one pass could save)."""
    plain = [ms[i] for i in members if lines[i]["op"] not in SLOW]
    now = sum(ms[i] for i in members)
    if len(plain) < len(members):
        return now, sum(plain)
    return now, sum(plain) - max(plain)


rng = np.random.default_rng(0)


def matrix(device, shape):
    """Stored column by column, as the results of juzhen's own operations are; positive where a log follows."""
    M = jz.zeros(*shape, device=device)
    M += jz.array(np.abs(rng.standard_normal(shape, dtype=np.float32)) * 0.5 + 0.1, device=device)
    return M


sizes = {"n": 512, "d": 64, "d2": 256} if SMALL else {"n": 16384, "d": 256, "d2": 2048}
print(f"backend {jz.backend()}, n = {sizes['n']}, d = {sizes['d']}")
for device in DEVICES:
    by_expression = collections.defaultdict(lambda: [0, 0.0, 0.0])
    pairs = collections.Counter()
    totals = {"repo": [0.0, 0.0, 0.0], "common": [0.0, 0.0, 0.0]}
    print(f"\n=== {device}: each formula; ms of all its lines, of the lines in groups, and the estimated saving")
    for source, function, shapes, numbers in FORMULAS:
        args = [matrix(device, tuple(sizes.get(s, s) for s in shape)) for shape in shapes] + numbers
        compiled = jz.compile(function)
        ms = compiled.profile(*args, show=False)
        record, = compiled._records.values()
        lines = record.lines
        found = groups_of(lines)
        in_groups = saved = 0.0
        for members in found:
            now, saving = estimate(lines, members, ms)
            in_groups += now
            saved += saving
            entry = by_expression[expression(lines, members)]
            entry[0] += 1
            entry[1] += now
            entry[2] += saving
            for i in members[:-1]:
                reader = next(j for j in members if lines[i]["out"] in lines[j]["args"])
                pairs[(lines[i]["op"], lines[reader]["op"])] += 1
        total = sum(ms)
        for k, value in enumerate((total, in_groups, saved)):
            totals[source][k] += value
        print(f"{source:6} {function.__name__:24} {total:9.3f} {in_groups:9.3f} {saved:9.3f}  "
              f"{100 * saved / total:5.1f}%  groups: " + "; ".join(str(len(m)) for m in found))
    for source, (total, in_groups, saved) in totals.items():
        print(f"{source:6} {'all':24} {total:9.3f} {in_groups:9.3f} {saved:9.3f}  {100 * saved / total:5.1f}%")

    print(f"\n=== {device}: the groups as expressions, largest estimated saving first (times seen, ms now, ms saved)")
    everything = sum(entry[2] for entry in by_expression.values()) or 1.0
    covered = 0.0
    for text, (count, now, saving) in sorted(by_expression.items(), key=lambda item: -item[1][2]):
        covered += saving
        print(f"{count:2}x {now:9.3f} {saving:9.3f}  {100 * covered / everything:5.1f}%  {text}")
    print(f"{len(by_expression)} different expressions in {sum(e[0] for e in by_expression.values())} groups")

    print(f"\n=== {device}: which operation's result goes into which (times seen in the groups)")
    for (first, second), count in pairs.most_common(15):
        print(f"{count:3}  {first} -> {second}")
