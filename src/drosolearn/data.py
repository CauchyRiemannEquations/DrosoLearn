"""Artificial quantity representation; no visual numerosity extraction."""
from dataclasses import dataclass
import hashlib
import numpy as np

NUISANCE_NAMES = ("object_size", "total_area", "density", "spacing", "layout", "visual_extent")

@dataclass
class Dataset:
    counts: np.ndarray
    nuisance: np.ndarray
    family_ids: list[str]
    split: str

    def target(self, larger=True):
        return (np.argmax(self.counts, axis=1) if larger else np.argmin(self.counts, axis=1))

    def inputs(self, include_quantity=True):
        quantity = self.counts[..., None] / 8.0 if include_quantity else np.zeros((*self.counts.shape, 1))
        return np.concatenate((quantity, 0.1 * (self.nuisance - 0.5)), axis=-1)

    def fingerprints(self):
        # Canonical stimulus identity is independent of left/right orientation.
        return {hashlib.sha256(self.nuisance[i, np.argsort(self.counts[i])].tobytes()
                               + np.sort(self.counts[i]).tobytes()).hexdigest()
                for i in range(len(self.counts))}


def make_dataset(seed, split, n, pair=(2, 4), domain="iid"):
    if n < 4 or n % 4 or pair[0] >= pair[1] or not 1 <= pair[0] < pair[1] <= 8:
        raise ValueError("n must be a positive multiple of four; 1 <= low < high <= 8")
    if domain not in ("iid", "ood"):
        raise ValueError("Unknown nuisance domain")
    rng = np.random.default_rng(seed)
    counts, nuisance, ids = [], [], []
    for group in range(n // 4):
        uv = rng.uniform(0.25, 0.75, (2, 6))
        if domain == "ood":
            uv = rng.uniform(0, 0.25, (2, 6)) + rng.integers(0, 2, (2, 6)) * 0.75
        low, high = pair
        # Exact crossing: each nuisance vector occurs equally at both quantities/sides.
        for cs, ns in [((low, high), uv), ((high, low), uv[::-1]),
                       ((low, high), uv[::-1]), ((high, low), uv)]:
            counts.append(cs)
            nuisance.append(ns.copy())
            ids.append(f"{split}:{group}")
    order = rng.permutation(n)
    return Dataset(np.asarray(counts, dtype=np.int64)[order], np.asarray(nuisance)[order],
                   [ids[i] for i in order], split)


def assert_disjoint(datasets):
    seen = set()
    for dataset in datasets:
        current = dataset.fingerprints()
        if seen & current:
            raise ValueError("Stimulus overlap between splits")
        seen |= current


def analytical_accuracy(dataset, larger=True, nuisance_index=None):
    signal = dataset.counts if nuisance_index is None else dataset.nuisance[..., nuisance_index]
    diff = signal[:, 1] - signal[:, 0]
    if not larger:
        diff = -diff
    target = dataset.target(larger)
    correct = np.where(diff == 0, 0.5, ((diff > 0).astype(int) == target).astype(float))
    return float(np.mean(correct))
