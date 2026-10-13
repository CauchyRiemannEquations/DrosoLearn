"""Load pinned anatomical counts; restrict computation to KC -> MBON pairs."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass
class Projection:
    pre: np.ndarray
    post: np.ndarray
    weights: np.ndarray
    n_kc: int
    n_mbon: int
    kc_ids: list[str]
    mbon_ids: list[str]
    metadata: dict

    def copy(self):
        return Projection(self.pre.copy(), self.post.copy(), self.weights.copy(), self.n_kc,
                          self.n_mbon, self.kc_ids.copy(), self.mbon_ids.copy(), self.metadata.copy())

def load_projection(directory):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    for file, key in [("edges.csv", "edges_sha256"), ("neurons.json", "neurons_sha256"),
                      ("annotations.csv", "roles_sha256")]:
        if sha256(directory / file) != provenance[key]:
            raise ValueError(f"Source checksum mismatch: {file}")
    ids = json.loads((directory / "neurons.json").read_text())
    rows = list(csv.DictReader((directory / "annotations.csv").open(encoding="utf-8", newline="")))
    if len(set(ids)) != len(ids) or {r["root_id"] for r in rows} != set(ids) or len(rows) != len(ids):
        raise ValueError("Invalid annotation IDs")
    for row in rows:
        if row["role"] in ("KC", "MBON") and row["cell_class"] != {"KC": "Kenyon_Cell", "MBON": "MBON"}[row["role"]]:
            raise ValueError("Role/cell-class mismatch")
    kc = sorted(r["root_id"] for r in rows if r["role"] == "KC")
    mbon = sorted(r["root_id"] for r in rows if r["role"] == "MBON")
    ki, mi = {x: i for i, x in enumerate(kc)}, {x: i for i, x in enumerate(mbon)}
    pre, post, weights = [], [], []
    with (directory / "edges.csv").open(encoding="utf-8", newline="") as f:
        for e in csv.DictReader(f):
            if e["pre"] in ki and e["post"] in mi:
                if int(e["sign"]) != 1 or int(e["count"]) <= 0:
                    raise ValueError("This simplified projection requires positive KC contacts")
                pre.append(ki[e["pre"]]); post.append(mi[e["post"]]); weights.append(float(e["count"]))
    if len(set(zip(pre, post))) != len(pre) or not pre:
        raise ValueError("Empty/duplicate anatomical pairs")
    return Projection(np.array(pre), np.array(post), np.array(weights), len(kc), len(mbon), kc, mbon,
                      {"source": provenance, "restricted_pairs": len(pre), "contacts": sum(weights),
                       "active_kcs": len(set(pre)), "active_mbons": len(set(post)),
                       "scope": "one-hop KC->MBON; DAN/APL/recurrent paths excluded"})

def topology_checks(original, control):
    same_pre = np.array_equal(np.bincount(original.pre, minlength=original.n_kc),
                             np.bincount(control.pre, minlength=control.n_kc))
    same_post = np.array_equal(np.bincount(original.post, minlength=original.n_mbon),
                              np.bincount(control.post, minlength=control.n_mbon))
    row_weights = all(np.array_equal(np.sort(original.weights[original.post == p]),
                                     np.sort(control.weights[control.post == p]))
                      for p in range(original.n_mbon))
    return {"same_edge_count": len(original.pre) == len(control.pre), "same_kc_outdegree": same_pre,
            "same_mbon_indegree": same_post, "same_incoming_weight_multisets": row_weights,
            "duplicate_free": len(set(zip(control.pre, control.post))) == len(control.pre),
            "changed_edge_fraction": len(set(zip(original.pre, original.post))
                                         - set(zip(control.pre, control.post))) / len(original.pre)}

def rewire(original, seed, swaps_per_edge=10):
    result = original.copy()
    rng = np.random.default_rng(seed)
    occupied = set(zip(result.pre, result.post))
    accepted, attempts, goal = 0, 0, len(result.pre) * swaps_per_edge
    while accepted < goal and attempts < goal * 100:
        attempts += 1
        a, b = rng.choice(len(result.pre), 2, replace=False)
        u, v = int(result.pre[a]), int(result.pre[b])
        p, q = int(result.post[a]), int(result.post[b])
        if u == v or p == q or (v, p) in occupied or (u, q) in occupied:
            continue
        occupied.remove((u, p)); occupied.remove((v, q))
        occupied.add((v, p)); occupied.add((u, q))
        result.pre[a], result.pre[b] = v, u
        accepted += 1
    if accepted < goal:
        raise ValueError("Rewiring budget did not complete")
    checks = topology_checks(original, result)
    if not all(checks[k] for k in ("same_edge_count", "same_kc_outdegree", "same_mbon_indegree",
                                   "same_incoming_weight_multisets", "duplicate_free")):
        raise ValueError("Rewiring invariant failed")
    result.metadata = {"control": "bipartite degree-preserving swaps", "accepted_swaps": accepted,
                       "attempts": attempts, "checks": checks, "seed": seed,
                       "not_preserved": ["KC outgoing strength", "type/compartment wiring", "recurrence"]}
    return result

def random_ann(original, seed):
    result = original.copy()
    rng = np.random.default_rng(seed)
    for post in range(result.n_mbon):
        loc = np.flatnonzero(result.post == post)
        result.pre[loc] = rng.choice(result.n_kc, len(loc), replace=False)
    result.metadata = {"control": "ordinary random sparse feedforward network", "seed": seed,
                       "checks": topology_checks(original, result)}
    return result
