"""New signed-reward shell policies; no hidden trace or participant fields in API.

The anatomical initialization follows model.Learner. Fixed initial catalog scaling
and old-head eligibility follow reward_linear.NormalizedLearner's published
implementation without importing its unrelated experiment orchestration.
"""
import numpy as np

from .graph import rewire, random_ann
from .model import Learner


CONDITIONS = ("fixed", "plastic", "rewired_fixed", "rewired_plastic",
              "ann_fixed", "ann_plastic", "linear", "q", "uniform", "bias", "persistence")
ADAPTIVE = CONDITIONS[:8]
CATALOG = np.zeros((6, 2, 7))
CATALOG[np.arange(6), 1, np.arange(6)] = 1.


def logistic(value):
    return float(1. / (1. + np.exp(-np.clip(value, -40., 40.))))


def build_graphs(graph, seed):
    return {"real": graph, "rewired": rewire(graph, seed + 110000),
            "ann": random_ann(graph, seed + 120000)}


class ShellActor(Learner):
    def __init__(self, graph, seed, rate, plastic):
        super().__init__(graph, seed + 100000, plastic=plastic, learning=True,
                         head_lr=rate, internal_lr=.1 * rate)
        self.head[:] = 0.
        self.pre, self.post = graph.pre, graph.post
        self._kc = Learner.kc(self, CATALOG)
        initial = Learner.mbon(self, self._kc)
        self.feature_scale = float(np.sqrt(np.mean(np.sum((initial[:, 1] - initial[:, 0]) ** 2, axis=-1))))
        if not np.isfinite(self.feature_scale) or self.feature_scale <= 0:
            raise ValueError("Positive catalog feature-gap RMS required")
        self._fixed_mb = initial if not plastic else None
        self.value_updates = 0

    def _parts(self, shell):
        if type(shell) is not int or not 1 <= shell <= 6:
            raise ValueError("Shell ID must be 1..6")
        kc = self._kc[shell - 1]
        mb = self._fixed_mb[shell - 1] if self._fixed_mb is not None else Learner.mbon(self, kc)
        return kc, mb, mb / self.feature_scale

    def probability(self, shell):
        _, _, features = self._parts(shell)
        return logistic((features[1] - features[0]) @ self.head)

    def observe(self, shell, action, reward):
        validate_observation(action, reward)
        kc, _, features = self._parts(shell)
        p = logistic((features[1] - features[0]) @ self.head)
        factor = (action - p) * reward
        old_head = self.head.copy()
        delta = self.head_lr * factor * (features[1] - features[0])
        self.head += delta
        self.head_updates += 1
        internal_step = 0.
        if self.plastic:
            old_weights = self.weights.copy()
            eligibility = old_head[self.post] * (kc[1, self.pre] - kc[0, self.pre]) / self.feature_scale
            self.weights = np.clip(self.weights + self.internal_lr * factor * eligibility, 0, self.upper)
            internal_step = float(np.linalg.norm(self.weights - old_weights))
            self.internal_updates += 1
        return float(np.linalg.norm(self.head - old_head)), internal_step, 0.

    def witness(self):
        mb = Learner.mbon(self, self._kc)
        return {"head": self.head.copy(), "weights": self.weights.copy(),
                "probe_kc": self._kc.copy(), "probe_mbon": mb,
                "probe_decision": mb / self.feature_scale,
                "encoder": self.encoder.copy(), "bias": self.bias.copy(),
                "pre": self.pre.copy(), "post": self.post.copy(),
                "initial_weights": self.initial_weights.copy(), "upper": self.upper.copy(),
                "feature_scale": np.asarray(self.feature_scale), "value_table": np.empty((0,))}


def validate_observation(action, reward):
    if type(action) is not int or action not in (0, 1) or reward not in (-1, 0, 1):
        raise ValueError("Binary action and signed observed reward required")
    if action == 0 and reward != 0:
        raise ValueError("NoGo cannot receive a hidden Go outcome")


class SimplePolicy:
    def __init__(self, condition, rate, bias):
        if condition not in CONDITIONS[6:] or not 0 < bias < 1:
            raise ValueError("Unsupported simple policy or bias")
        self.condition, self.rate, self.fixed_bias = condition, rate, bias
        self.head = np.zeros(7) if condition == "linear" else np.empty(0)
        self.weights = np.empty(0)
        self.values = np.zeros((6, 2)) if condition == "q" else np.empty(0)
        self.last = np.full(6, -1, dtype=int)
        self.head_updates = self.internal_updates = self.value_updates = 0

    def probability(self, shell):
        if type(shell) is not int or not 1 <= shell <= 6:
            raise ValueError("Shell ID must be 1..6")
        if self.condition == "linear":
            return logistic(self.head[shell - 1])
        if self.condition == "q":
            return logistic(4. * (self.values[shell - 1, 1] - self.values[shell - 1, 0]))
        if self.condition == "uniform":
            return .5
        if self.condition == "bias":
            return self.fixed_bias
        last = self.last[shell - 1]
        return .5 if last < 0 else (.9 if last == 1 else .1)

    def observe(self, shell, action, reward):
        validate_observation(action, reward)
        p = self.probability(shell)
        head_step = value_step = 0.
        if self.condition == "linear":
            old = float(self.head[shell - 1])
            step = self.rate * (action - p) * reward
            self.head[shell - 1] += step
            self.head_updates += 1
            head_step = abs(float(self.head[shell - 1]) - old)
        elif self.condition == "q":
            old = float(self.values[shell - 1, action])
            step = self.rate * (reward - self.values[shell - 1, action])
            self.values[shell - 1, action] += step
            self.value_updates += 1
            value_step = abs(float(self.values[shell - 1, action]) - old)
        elif self.condition == "persistence":
            self.last[shell - 1] = action
        return head_step, 0., value_step

    def witness(self):
        empty = np.empty((6, 2, 0))
        decision = CATALOG.copy() if self.condition == "linear" else empty.copy()
        return {"head": self.head.copy(), "weights": self.weights.copy(),
                "probe_kc": empty.copy(), "probe_mbon": empty.copy(), "probe_decision": decision,
                "encoder": np.empty((0, 7)), "bias": np.empty(0), "pre": np.empty(0, dtype=int),
                "post": np.empty(0, dtype=int), "initial_weights": np.empty(0), "upper": np.empty(0),
                "feature_scale": np.asarray(1.), "value_table": self.values.copy(),
                "last_action_state": self.last.copy()}


def make_model(condition, seed, rate, calibration_bias, graphs):
    if condition not in CONDITIONS or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Condition and positive learning rate required")
    if condition in CONDITIONS[:6]:
        kind = "real" if condition in ("fixed", "plastic") else condition.split("_")[0]
        return ShellActor(graphs[kind], seed, rate, condition.endswith("plastic"))
    return SimplePolicy(condition, rate, calibration_bias)
