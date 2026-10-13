"""Reward-only computational policy gradient, not a calibrated dopamine rule."""
import numpy as np

class Learner:
    def __init__(self, graph, seed, plastic=False, learning=True, head_lr=0.1, internal_lr=0.01,
                 head_learning=True):
        self.graph = graph
        rng = np.random.default_rng(seed)
        self.encoder = rng.normal(0, 0.8, (graph.n_kc, 7))
        self.bias = rng.uniform(-0.25, 0.25, graph.n_kc)
        self.head = rng.normal(0, 0.02, graph.n_mbon)
        self.normalizer = np.maximum(np.bincount(graph.post, weights=graph.weights,
                                                minlength=graph.n_mbon), 1.0)
        self.weights = graph.weights / self.normalizer[graph.post]
        self.initial_weights = self.weights.copy()
        self.upper = 2 * self.initial_weights
        self.plastic, self.learning = plastic, learning
        self.head_learning = head_learning
        self.head_lr, self.internal_lr = head_lr, internal_lr
        self.head_updates = self.internal_updates = 0

    def kc(self, inputs):
        return np.maximum(inputs @ self.encoder.T + self.bias, 0)

    def mbon(self, kc):
        # Batch scatter into each postsynaptic MBON; no anatomical changes outside mask.
        output = np.zeros((*kc.shape[:-1], self.graph.n_mbon))
        flat = kc.reshape(-1, self.graph.n_kc)
        out = output.reshape(-1, self.graph.n_mbon)
        for i, row in enumerate(flat):
            out[i] = np.bincount(self.graph.post, weights=self.weights * row[self.graph.pre],
                                  minlength=self.graph.n_mbon)
        return output

    def features(self, inputs):
        kc = self.kc(inputs)
        return kc, self.mbon(kc)

    def probabilities(self, inputs):
        _, z = self.features(inputs)
        score = z @ self.head
        gap = np.clip(score[..., 1] - score[..., 0], -40, 40)
        return 1 / (1 + np.exp(-gap))

    def update(self, inputs, action, reward):
        # The optimizer receives selected action/reward only, never the correct label.
        if action not in (0, 1) or reward not in (0, 1):
            raise ValueError("Expected binary action/reward")
        kc, z = self.features(inputs)
        gap = float((z[1] - z[0]) @ self.head)
        probability = 1 / (1 + np.exp(-np.clip(gap, -40, 40)))
        factor = (action - probability) * (reward - 0.5)
        old_head = self.head.copy()
        if self.learning:
            if self.head_learning:
                self.head += self.head_lr * factor * (z[1] - z[0])
                self.head_updates += 1
            if self.plastic:
                eligibility = old_head[self.graph.post] * (kc[1, self.graph.pre] - kc[0, self.graph.pre])
                self.weights = np.clip(self.weights + self.internal_lr * factor * eligibility, 0, self.upper)
                self.internal_updates += 1
        return probability

    def snapshot(self, probe_inputs):
        kc, mb = self.features(probe_inputs)
        return {"internal_weights": self.weights.copy(), "readout_weights": self.head.copy(),
                "kc_activity": kc, "mbon_activity": mb,
                "choice_probability": self.probabilities(probe_inputs)}
