"""Independent audit of the frozen-policy input-order diagnostic.

No learner, data, experiment, encoding or summary helpers are imported.  Every
actual training update is replayed; the exhaustive code panel is evaluation of
one frozen state, not 720 additional fits or independent observations.
"""
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


_FREE = (1, 3, 5, 6, 7, 8)
_PAIRS = tuple(p for p in combinations(range(1, 9), 2) if p != (2, 4))
_PRIMARY = np.asarray([i for i, p in enumerate(_PAIRS) if set(p) <= set(_FREE)])
_ANCHORED = np.asarray([i for i, p in enumerate(_PAIRS) if not set(p) <= set(_FREE)])
_FITS = ("quantity_present", "quantity_removed")
_FLAGS = {"fixed": ("real", False), "plastic": ("real", True),
          "rewired_fixed": ("rewired", False), "rewired_plastic": ("rewired", True),
          "ann_fixed": ("ann", False), "ann_plastic": ("ann", True)}
_TRIAL_FIELDS = {"seed", "condition", "fit", "trial", "family_id", "left_count", "right_count", "action", "reward",
                 "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "fit", "trial", "split", "p_target", "greedy_accuracy", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2"}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability", "encoder", "bias", "pre", "post"}
_PRESENT_FIELDS = {"code_values", "score_cache", "map_p_target", "map_greedy_accuracy", "ordered_p_right"}
_REMOVED_FIELDS = {"code_values", "score_cache", "p_target", "greedy_accuracy", "p_right"}
_INT_FIELDS = {"seed", "trial", "left_count", "right_count", "action", "reward"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _near(actual, expected, label, atol=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a, b, rtol=1e-9, atol=atol), f"Mismatch: {label}")


def _integer(row, field):
    result = int(row[field])
    _require(str(result) == str(row[field]), f"Noninteger CSV: {field}")
    return result


def _number(row, field):
    result = float(row[field])
    _require(np.isfinite(result), f"Nonfinite CSV: {field}")
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path, fields, keys):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(set(reader.fieldnames or ()) == fields, f"CSV schema: {path}")
        for row in reader:
            _require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
            key = tuple(_integer(row, f) if f in _INT_FIELDS else row[f] for f in keys)
            _require(key not in result, f"Duplicate CSV key: {key}")
            result[key] = row
    return result


def _npz(path, fields):
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(fields), f"NPZ schema: {path}")
        return {k: archive[k].copy() for k in archive.files}


def _regenerate_panel():
    # Explicit row construction is independent of the producer's tile/scatter.
    rows = []
    for order in permutations(_FREE):
        mapping = dict(zip(_FREE, order))
        mapping.update({2: 2, 4: 4})
        rows.append([mapping[n] for n in range(1, 9)])
    return np.asarray(rows, dtype=np.int64)


def _regenerate_dataset(seed, name, n, pair):
    rng = np.random.default_rng(seed)
    counts, nuisance, families = [], [], []
    for group in range(n // 4):
        uv = rng.uniform(0.25, 0.75, (2, 6))
        if name == "ood":
            uv = rng.uniform(0, 0.25, (2, 6)) + rng.integers(0, 2, (2, 6)) * 0.75
        low, high = pair
        for cs, ns in (((low, high), uv), ((high, low), uv[::-1]),
                       ((low, high), uv[::-1]), ((high, low), uv)):
            counts.append(cs)
            nuisance.append(ns.copy())
            families.append(f"{name}:{group}")
    order = rng.permutation(n)
    return {"counts": np.asarray(counts, dtype=np.int64)[order], "nuisance": np.asarray(nuisance)[order],
            "family_ids": np.asarray(families)[order]}


def _dataset(stimuli, name, n, pair, fingerprints):
    data = {k: stimuli[f"{name}_{k}"] for k in ("counts", "nuisance", "family_ids")}
    counts, nuisance, families = (data[k] for k in ("counts", "nuisance", "family_ids"))
    _require(counts.shape == (n, 2) and np.issubdtype(counts.dtype, np.integer), f"Counts schema: {name}")
    _require(nuisance.shape == (n, 2, 6) and np.isfinite(nuisance).all(), f"Nuisance schema: {name}")
    _require(families.shape == (n,) and np.all(np.sort(counts, axis=1) == pair), f"Family/pair schema: {name}")
    domain = ((nuisance <= .25) | (nuisance >= .75)) if name == "ood" else ((nuisance >= .25) & (nuisance <= .75))
    _require(np.all((nuisance >= 0) & (nuisance <= 1)) and np.all(domain), f"Descriptor domain: {name}")
    unique, multiplicity = np.unique(families, return_counts=True)
    _require(set(map(str, unique)) == {f"{name}:{i}" for i in range(n // 4)} and np.all(multiplicity == 4), f"Crossed family coverage: {name}")
    for family in unique:
        assignments = {}
        for i in np.flatnonzero(families == family):
            for side in (0, 1):
                vector = np.asarray(nuisance[i, side], dtype=np.float64).tobytes()
                assignments.setdefault(vector, []).append((int(counts[i, side]), side))
        expected = sorted((int(count), side) for count in pair for side in (0, 1))
        _require(len(assignments) == 2 and all(sorted(a) == expected for a in assignments.values()), f"Joint nuisance crossing: {name}/{family}")
    target = (counts[:, 1] > counts[:, 0]).astype(int)
    _near(np.mean(np.argmax(counts, axis=1) == target), 1, "Privileged quantity oracle", atol=0)
    for j in range(6):
        diff = nuisance[:, 1, j] - nuisance[:, 0, j]
        correct = np.where(diff == 0, .5, ((diff > 0).astype(int) == target).astype(float))
        _near(correct.mean(), .5, f"Single nuisance chance: {name}/{j}", atol=0)
    current = set()
    for row_c, row_n in zip(counts, nuisance):
        order = np.argsort(row_c)
        canonical = np.asarray(row_n[order], dtype=np.float64).tobytes() + np.asarray(row_c[order], dtype=np.int64).tobytes()
        current.add(hashlib.sha256(canonical).hexdigest())
    _require(not current.intersection(fingerprints), f"Canonical stimulus leakage: {name}")
    fingerprints.update(current)
    return data


def _inputs(data, codes):
    quantity = np.asarray(codes)[data["counts"] - 1, None] / 8.0
    return np.concatenate((quantity, .1 * (data["nuisance"] - .5)), axis=-1)


def _mbon(kc, pre, post, weights, n_mbon):
    flat = kc.reshape(-1, kc.shape[-1])
    result = np.zeros((len(flat), n_mbon))
    for i, row in enumerate(flat):
        np.add.at(result[i], post, row[pre] * weights)
    return result.reshape(*kc.shape[:-1], n_mbon)


def _forward(inputs, encoder, bias, pre, post, weights, head):
    kc = np.maximum(np.matmul(inputs, encoder.T) + bias, 0.0)
    mb = _mbon(kc, pre, post, weights, len(head))
    gap = np.matmul(mb, head)[..., 1] - np.matmul(mb, head)[..., 0]
    return kc, mb, 1 / (1 + np.exp(-np.clip(gap, -40, 40)))


def _metrics(probability, counts):
    target = (counts[:, 1] > counts[:, 0]).astype(int)
    correct = np.where(np.abs(probability - .5) < 1e-12, .5,
                       ((probability > .5).astype(int) == target).astype(float))
    return {"p_target": float(np.mean(np.where(target == 1, probability, 1 - probability))),
            "greedy_accuracy": float(correct.mean())}


def _regenerate_topology(real, seed, kind, n_kc, n_mbon):
    pre, post = real["pre"].copy(), real["post"]
    rng = np.random.default_rng(seed + (50000 if kind == "rewired" else 60000))
    if kind == "ann":
        for p in range(n_mbon):
            loc = np.flatnonzero(post == p)
            pre[loc] = rng.choice(n_kc, len(loc), replace=False)
        return pre, None, None
    occupied = set(zip(map(int, pre), map(int, post)))
    accepted = attempts = 0
    goal = len(pre) * 10
    while accepted < goal and attempts < goal * 100:
        attempts += 1
        a, b = rng.choice(len(pre), 2, replace=False)
        old_a, old_b = (int(pre[a]), int(post[a])), (int(pre[b]), int(post[b]))
        new_a, new_b = (old_b[0], old_a[1]), (old_a[0], old_b[1])
        if old_a[0] == old_b[0] or old_a[1] == old_b[1] or new_a in occupied or new_b in occupied:
            continue
        occupied.difference_update((old_a, old_b))
        occupied.update((new_a, new_b))
        pre[a], pre[b] = new_a[0], new_b[0]
        accepted += 1
    _require(accepted == goal, "Independent rewiring budget")
    return pre, accepted, attempts


def _topologies(path, records, seed, n_kc, n_mbon, source_metadata):
    graphs = {k: _npz(path / f"{k}_topology.npz", {"pre", "post", "counts"}) for k in ("real", "rewired", "ann")}
    real = graphs["real"]
    _require(records["real"]["metadata"] == source_metadata, "Real topology source metadata")
    for kind, graph in graphs.items():
        pre, post, counts = (graph[k] for k in ("pre", "post", "counts"))
        _require(pre.ndim == post.ndim == counts.ndim == 1 and pre.shape == post.shape == counts.shape == real["pre"].shape and len(pre) > 0, "Topology dimensions/edge count")
        _require(np.issubdtype(pre.dtype, np.integer) and np.issubdtype(post.dtype, np.integer), "Topology integer indices")
        _require(np.all((pre >= 0) & (pre < n_kc)) and np.all((post >= 0) & (post < n_mbon)), "Topology index bounds")
        _require(np.isfinite(counts).all() and np.all(counts > 0) and np.all(counts == np.floor(counts)), "Positive integral anatomical counts")
        _require(len(set(zip(pre.tolist(), post.tolist()))) == len(pre), "Duplicate topology pairs")
        if kind == "real":
            _near(len(pre), source_metadata["restricted_pairs"], "Source restricted pairs", atol=0)
            _near(counts.sum(), source_metadata["contacts"], "Source contacts", atol=0)
            _near(len(set(pre)), source_metadata["active_kcs"], "Source active KC count", atol=0)
            _near(len(set(post)), source_metadata["active_mbons"], "Source active MBON count", atol=0)
            continue
        actual = {"same_edge_count": len(pre) == len(real["pre"]),
                  "same_kc_outdegree": np.array_equal(np.bincount(pre, minlength=n_kc), np.bincount(real["pre"], minlength=n_kc)),
                  "same_mbon_indegree": np.array_equal(np.bincount(post, minlength=n_mbon), np.bincount(real["post"], minlength=n_mbon)),
                  "same_incoming_weight_multisets": all(np.array_equal(np.sort(counts[post == p]), np.sort(real["counts"][real["post"] == p])) for p in range(n_mbon)),
                  "duplicate_free": True,
                  "changed_edge_fraction": len(set(zip(real["pre"].tolist(), real["post"].tolist())) - set(zip(pre.tolist(), post.tolist()))) / len(pre)}
        recorded = records[kind]["metadata"]
        _require(set(recorded["checks"]) == set(actual), "Topology metadata check schema")
        for key, value in actual.items():
            _near(recorded["checks"][key], value, f"Topology resource check: {kind}/{key}", atol=0)
        _require(all(actual[k] for k in ("same_edge_count", "same_mbon_indegree", "same_incoming_weight_multisets", "duplicate_free")), "Matched topology resources")
        if kind == "rewired":
            _require(actual["same_kc_outdegree"], "Rewired KC outdegree")
        regenerated, accepted, attempts = _regenerate_topology(real, seed, kind, n_kc, n_mbon)
        _require(np.array_equal(pre, regenerated) and np.array_equal(post, real["post"]) and np.array_equal(counts, real["counts"]), f"Registered topology RNG: {kind}")
        _require(recorded["seed"] == seed + (50000 if kind == "rewired" else 60000), "Control topology seed")
        if kind == "rewired":
            _require(recorded["accepted_swaps"] == accepted and recorded["attempts"] == attempts, "Independent rewiring counters")
    return graphs


def _cache(datasets, codes, encoder, bias, pre, post, weights, head):
    # Independent accumulation of KC -> MBON -> readout coefficients.  The
    # producer uses bincount; this audit accumulates with add.at, then scores
    # each scalar code separately rather than broadcasting a code dimension.
    gain = np.zeros(len(encoder))
    np.add.at(gain, pre, weights * head[post])
    result = []
    for pair in _PAIRS:
        data = datasets[f"pair_{pair[0]}_{pair[1]}"]
        values = []
        for code in codes:
            inputs = np.concatenate((np.full((*data["counts"].shape, 1), float(code) / 8), .1 * (data["nuisance"] - .5)), axis=-1)
            kc = np.maximum(np.matmul(inputs, encoder.T) + bias, 0)
            values.append(np.matmul(kc, gain))
        result.append(np.stack(values, axis=-1))
    return np.asarray(result)


def _transfer(saved, fit, datasets, panel, encoder, bias, pre, post, weights, head):
    codes = np.arange(1, 9) if fit == "quantity_present" else np.asarray([0])
    _require(np.array_equal(saved["code_values"], codes), "Transfer code values")
    cache = _cache(datasets, codes, encoder, bias, pre, post, weights, head)
    _near(saved["score_cache"], cache, "Independently recomputed frozen score cache")
    means, greedy, ordered_cases = [], [], []
    for j, pair in enumerate(_PAIRS):
        data = datasets[f"pair_{pair[0]}_{pair[1]}"]
        counts = data["counts"]
        if fit == "quantity_present":
            # Flattened score addresses provide a separate indexing path from
            # the producer's left/right advanced-array indexing.
            selection = panel[:, counts - 1] - 1
            offsets = np.arange(len(counts))[None, :, None] * 16 + np.arange(2)[None, None, :] * 8
            scores = cache[j].reshape(-1)[offsets + selection]
            probability = 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))
            target = (counts[:, 1] > counts[:, 0]).astype(int)[None, :]
            correct = np.where(np.abs(probability - .5) < 1e-12, .5,
                               ((probability > .5).astype(int) == target).astype(float))
            means.append(np.mean(np.where(target == 1, probability, 1 - probability), axis=1))
            greedy.append(correct.mean(axis=1))
            ordered_cases.append(probability[0])
        else:
            probability = 1 / (1 + np.exp(-np.clip(cache[j, :, 1, 0] - cache[j, :, 0, 0], -40, 40)))
            metrics = _metrics(probability, counts)
            means.append(metrics["p_target"])
            greedy.append(metrics["greedy_accuracy"])
            ordered_cases.append(probability)
    if fit == "quantity_present":
        expected = {"map_p_target": np.asarray(means).T, "map_greedy_accuracy": np.asarray(greedy).T,
                    "ordered_p_right": np.asarray(ordered_cases)}
        _near(expected["map_p_target"][:, _PRIMARY].mean(axis=0), np.full(15, .5), "Full-panel unseen probability symmetry", atol=1e-12)
        _near(expected["map_greedy_accuracy"][:, _PRIMARY].mean(axis=0), np.full(15, .5), "Full-panel unseen greedy symmetry", atol=1e-12)
    else:
        expected = {"p_target": np.asarray(means), "greedy_accuracy": np.asarray(greedy), "p_right": np.asarray(ordered_cases)}
        _near(expected["p_target"], np.full(27, .5), "Quantity-removed crossed probability chance", atol=1e-12)
        _near(expected["greedy_accuracy"], np.full(27, .5), "Quantity-removed crossed greedy chance", atol=1e-12)
    for field, value in expected.items():
        _near(saved[field], value, f"Every frozen transfer entry: {fit}/{field}")
    return expected


def _summary(cfg, measurements, transfer, summary):
    n, criteria = cfg["trials"], cfg["criteria"]
    rows = {(r["seed"], r["condition"], r["fit"]): r for r in summary["outcomes"]}
    keys = {(s, c, f) for s in cfg["seeds"] for c in cfg["conditions"] for f in _FITS}
    _require(len(rows) == len(summary["outcomes"]) and set(rows) == keys, "Summary exact outcome coverage")
    contrasts = {(r["seed"], r["condition"]): r for r in summary["contrasts"]}
    expected_contrasts = {}
    for seed, condition, fit in sorted(keys):
        before, after = (measurements[seed, condition, fit, t, "iid"]["p_target"] for t in (0, n))
        acquired = bool(after >= criteria["acquisition_accuracy"] and after - before >= criteria["acquisition_gain"])
        expected = {"before_accuracy": before, "after_accuracy": after, "gain": after - before, "acquired": acquired,
                    "ood_accuracy": measurements[seed, condition, fit, n, "ood"]["p_target"],
                    "head_updates": n, "internal_updates": n if _FLAGS[condition][1] else 0}
        frozen = transfer[seed, condition, fit, "final"]
        baseline = transfer[seed, condition, fit, "baseline"]
        if fit == "quantity_present":
            means = frozen["map_p_target"]
            expected.update(ordered_unseen=float(means[0, _PRIMARY].mean()), permuted_unseen=float(means[:, _PRIMARY].mean()),
                            ordered_unseen_before=float(baseline["map_p_target"][0, _PRIMARY].mean()),
                            ordered_unseen_gain=float((means[0, _PRIMARY] - baseline["map_p_target"][0, _PRIMARY]).mean()),
                            permuted_unseen_before=float(baseline["map_p_target"][:, _PRIMARY].mean()),
                            ordered_anchored=float(means[0, _ANCHORED].mean()), permuted_anchored=float(means[:, _ANCHORED].mean()),
                            ordered_all=float(means[0].mean()), permuted_all=float(means.mean()),
                            qualified_ordered_transfer=bool(acquired and means[0, _PRIMARY].mean() >= criteria["transfer_accuracy"]),
                            qualified_permuted_transfer=bool(acquired and means[:, _PRIMARY].mean() >= criteria["transfer_accuracy"]))
            expected_contrasts[seed, condition] = {"representation_gap": expected["ordered_unseen"] - expected["permuted_unseen"], "acquired": acquired}
        else:
            expected.update(removed_unseen=float(frozen["p_target"][_PRIMARY].mean()), removed_all=float(frozen["p_target"].mean()),
                            removed_unseen_before=float(baseline["p_target"][_PRIMARY].mean()))
        _require(set(rows[seed, condition, fit]) == set(expected) | {"seed", "condition", "fit"}, "Summary outcome schema")
        for field, value in expected.items():
            if type(value) is bool:
                _require(rows[seed, condition, fit][field] is value, f"Summary acquisition/transfer gate: {field}")
            else:
                _near(rows[seed, condition, fit][field], value, f"Summary outcome: {field}")
    _require(len(contrasts) == len(summary["contrasts"]) and set(contrasts) == set(expected_contrasts), "Summary paired contrast coverage")
    for key, expected in expected_contrasts.items():
        _require(set(contrasts[key]) == set(expected) | {"seed", "condition"}, "Paired contrast schema")
        _require(contrasts[key]["acquired"] is expected["acquired"], "Paired contrast acquisition gate")
        _near(contrasts[key]["representation_gap"], expected["representation_gap"], "Paired representation gap")
    verdicts = {r["condition"]: r for r in summary["family_verdicts"]}
    _require(len(verdicts) == len(summary["family_verdicts"]) and set(verdicts) == set(cfg["conditions"]), "Family verdict coverage")
    for condition in cfg["conditions"]:
        selected = [expected_contrasts[s, condition] for s in cfg["seeds"]]
        gaps = [r["representation_gap"] for r in selected]
        qualified = [r["representation_gap"] for r in selected if r["acquired"]]
        mean, positive = float(np.mean(gaps)), sum(g > 0 for g in gaps)
        expected = {"n_seeds": len(gaps), "mean_gap": mean, "sd_gap": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
                    "positive_seed_count": positive,
                    "criterion_met": bool(mean >= criteria["representation_gap"] and positive / len(gaps) >= criteria["positive_seed_fraction"]),
                    "qualified_seed_count": len(qualified), "qualified_mean_gap": float(np.mean(qualified)) if qualified else None}
        _require(set(verdicts[condition]) == set(expected) | {"condition"}, "Family verdict schema")
        for field, value in expected.items():
            if value is None or type(value) is bool:
                _require(verdicts[condition][field] is value, f"Family gate/null: {field}")
            else:
                _near(verdicts[condition][field], value, f"Family verdict: {field}")
    _require(summary["actual_training_trials"] == len(cfg["seeds"]) * len(cfg["conditions"]) * 2 * n and summary["permutations"] == 720, "Actual fit/trial resources; permutations are not training")


def _audit(output, report):
    metadata, registration, summary = (_json(output / f) for f in ("metadata.json", "registration.json", "summary.json"))
    cfg = metadata["config"]
    _require(metadata.get("status") == summary.get("status") == "completed", "Study did not complete")
    _require(cfg == registration["config"] and summary["scope"] == cfg["scope"], "Config/scope consistency")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration hash")
    times = [datetime.fromisoformat(r[f]) for r, f in ((registration, "registered_at_utc"), (metadata, "started_at_utc"), (metadata, "completed_at_utc"))]
    _require(all(t.tzinfo is not None for t in times) and times == sorted(times), "Local registration/run chronology")
    _require(cfg["scope"] in ("preregistered_diagnostic", "integration_test"), "Study scope")
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    _require(seeds and all(type(s) is int and s >= 0 for s in seeds) and len(set(seeds)) == len(seeds), "Unique nonnegative seeds")
    _require(conditions and all(c in _FLAGS for c in conditions) and len(set(conditions)) == len(conditions), "Unique supported conditions")
    _require(tuple(cfg["fits"]) == _FITS and cfg["train_pair"] == [2, 4], "Two actual fits with anchored 2-vs-4 train pair")
    _require(metadata["rng"] == {"actions": "seed+280000", "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000", "data": "seed*100+31,+32,+33,+40+pair_index"}, "Registered RNG offsets")
    n, every, ev, probe_n = (cfg[k] for k in ("trials", "checkpoint_every", "eval_trials", "probe_rows"))
    _require(all(type(v) is int and v > 0 for v in (n, every, ev, probe_n)) and n % 4 == ev % 4 == n % every == 0 and probe_n <= ev, "Budget/crossing validity")
    if cfg["scope"] == "preregistered_diagnostic":
        _require(seeds == list(range(3001, 3011)) and tuple(conditions) == tuple(_FLAGS), "Registered ten new seeds and six model families")
        _require((n, every, ev, probe_n, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 8, .1, .01), "Fixed registered experiment resources")
    _require(all(type(cfg[k]) in (int, float) and np.isfinite(cfg[k]) and cfg[k] >= 0 for k in ("head_lr", "internal_lr")), "Learning rate validity")
    _require(set(cfg["criteria"]) == {"acquisition_accuracy", "acquisition_gain", "transfer_accuracy", "representation_gap", "positive_seed_fraction"}
             and all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in cfg["criteria"].values())
             and cfg["criteria"]["positive_seed_fraction"] > 0, "Criteria schema/ranges")
    checkpoints = list(range(0, n + 1, every))
    hashes = _json(output / "artifact_hashes.json")
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "learning_curves.csv", "permutations.npy", "code_audit.json", "code_geometry.npz"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv", "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            for fit in _FITS:
                path = f"{base}/{condition}/{fit}"
                required.update(f"{path}/checkpoint_{t:04d}.npz" for t in checkpoints)
                required.update(f"{path}/transfer_{stage}.npz" for stage in ("baseline", "final"))
    _require(required <= set(hashes), "Required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file(), f"Unsafe/missing artifact: {relative}")
        _require(_hash(path) == digest, f"Artifact hash mismatch: {relative}")
    panel = np.load(output / "permutations.npy", allow_pickle=False)
    _require(np.issubdtype(panel.dtype, np.integer) and np.array_equal(panel, _regenerate_panel()), "All720 exact lexicographic anchored panel")
    code_audit = _json(output / "code_audit.json")
    _require(set(code_audit) == {"anchors", "fully_unseen_pairs", "anchored_pairs", "code_order_oracle_unseen", "code_order_oracle_anchored", "code_order_oracle_all", "identity_aware_inverse_oracle", "inverse_oracle_scope", "code_geometry_gap_units", "oracle_is_privileged", "full_unseen_policy_chance"}, "Code audit schema")
    _require(code_audit["anchors"] == {"2": 2, "4": 4} and code_audit["fully_unseen_pairs"] == [list(_PAIRS[i]) for i in _PRIMARY]
             and code_audit["anchored_pairs"] == [list(_PAIRS[i]) for i in _ANCHORED], "Primary15 and anchored12 distinct pair sets")
    concordance = np.asarray([[mapping[b - 1] > mapping[a - 1] for a, b in _PAIRS] for mapping in panel], dtype=float)
    geometry = _npz(output / "code_geometry.npz", {"pairs", "quantity_ratios", "quantity_differences", "code_gaps", "code_order_agreement"})
    _require(np.array_equal(geometry["pairs"], np.asarray(_PAIRS)) and np.issubdtype(geometry["pairs"].dtype, np.integer), "Code geometry pair order")
    _near(geometry["quantity_ratios"], np.asarray([a / b for a, b in _PAIRS]), "Code geometry true ratios", atol=0)
    _require(np.array_equal(geometry["quantity_differences"], np.asarray([b - a for a, b in _PAIRS])), "Code geometry true differences")
    gaps = np.asarray([[mapping[b - 1] - mapping[a - 1] for a, b in _PAIRS] for mapping in panel])
    _require(np.array_equal(geometry["code_gaps"], gaps) and np.issubdtype(geometry["code_gaps"].dtype, np.integer), "Code geometry signed mapped gaps")
    _near(geometry["code_order_agreement"], concordance, "Code geometry mapped order agreement", atol=0)
    _require(code_audit["code_geometry_gap_units"] == "integer code units1..8; divide by8 for actual scalar input gap", "Code geometry gap units")
    for field, expected in (("code_order_oracle_unseen", concordance[:, _PRIMARY].mean()), ("code_order_oracle_anchored", concordance[:, _ANCHORED].mean()),
                            ("code_order_oracle_all", concordance.mean()), ("identity_aware_inverse_oracle", 1.0)):
        _near(code_audit[field], expected, f"Analytic code baseline: {field}")
    _require(code_audit["oracle_is_privileged"] is True, "Inverse-map oracle must be marked privileged")
    _require(code_audit["inverse_oracle_scope"] == "unique scalar mappings only; quantity_removed is noninvertible", "Inverse-map oracle scope excludes removed quantity")
    _require(code_audit["full_unseen_policy_chance"] == "structural symmetry fixed before execution, not discovered learning failure", "Predeclared structural-chance interpretation")
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS, ("seed", "condition", "fit", "trial", "split"))
    _require(set(curves) == {(s, c, f, t, split) for s in seeds for c in conditions for f in _FITS for t in checkpoints for split in ("iid", "ood")}, "Learning curve exact design coverage")
    records = _json(output / "controls.json")
    controls = {(r["seed"], r["kind"]): r for r in records}
    _require(len(records) == len(controls) and set(controls) == {(s, k) for s in seeds for k in ("real", "rewired", "ann")}, "Topology control coverage")
    fingerprints, measurements, transfer = set(), {}, {}
    trial_count = snapshot_count = transfer_count = matrix_entries = 0
    split_specs = [("train", (2, 4), n, 31), ("iid", (2, 4), ev, 32), ("ood", (2, 4), ev, 33)]
    split_specs += [(f"pair_{a}_{b}", (a, b), ev, 40 + i) for i, (a, b) in enumerate(_PAIRS)]
    for seed in seeds:
        seed_path = output / f"seed_{seed}"
        stimuli = _npz(seed_path / "stimuli.npz", {f"{name}_{k}" for name, _, _, _ in split_specs for k in ("counts", "nuisance", "family_ids")} | {"action_uniforms"})
        datasets = {}
        for name, pair, size, offset in split_specs:
            datasets[name] = _dataset(stimuli, name, size, pair, fingerprints)
            regenerated = _regenerate_dataset(seed * 100 + offset, name, size, pair)
            for field, expected in regenerated.items():
                _require(np.array_equal(datasets[name][field], expected), f"Registered data RNG: {name}/{field}")
        uniforms = np.random.default_rng(seed + 280000).random(n)
        _require(np.array_equal(stimuli["action_uniforms"], uniforms), "Shared action uniform RNG")
        ordered_train = _inputs(datasets["train"], np.arange(1, 9))
        _require(all(np.array_equal(ordered_train, _inputs(datasets["train"], mapping)) for mapping in panel), "Every anchored training input bitwise identical")
        for name in ("iid", "ood"):
            _require(all(np.array_equal(_inputs(datasets[name], np.arange(1, 9)), _inputs(datasets[name], mapping)) for mapping in panel), "All anchored trained-quantity evaluations identical")
        trials = _csv(seed_path / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "fit", "trial"))
        _require(set(trials) == {(seed, c, f, t) for c in conditions for f in _FITS for t in range(1, n + 1)}, "Two actual fits, exact trial design; no panel retraining")
        trial_count += len(trials)
        first = _npz(seed_path / conditions[0] / _FITS[0] / "checkpoint_0000.npz", _SAVED_FIELDS)
        _require(first["encoder"].ndim == 2, "Encoder matrix schema")
        n_kc, n_mbon = first["encoder"].shape[0], first["readout_weights"].size
        _require(n_kc > 0 and n_mbon > 0, "Positive neuron dimensions")
        graphs = _topologies(seed_path, {k: controls[seed, k] for k in ("real", "rewired", "ann")}, seed, n_kc, n_mbon, metadata["graph"])
        rng = np.random.default_rng(seed + 70000)
        encoder, bias, initial_head = rng.normal(0, .8, (n_kc, 7)), rng.uniform(-.25, .25, n_kc), rng.normal(0, .02, n_mbon)
        for condition in conditions:
            kind, plastic = _FLAGS[condition]
            graph = graphs[kind]
            pre, post = graph["pre"], graph["post"]
            normalizer = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)
            w0 = graph["counts"] / normalizer[post]
            for fit in _FITS:
                path = seed_path / condition / fit
                _require({p.name for p in path.glob("*.npz")} == {f"checkpoint_{t:04d}.npz" for t in checkpoints} | {"transfer_baseline.npz", "transfer_final.npz"}, "Exact frozen-state file coverage")
                codes = np.arange(1, 9) if fit == "quantity_present" else np.zeros(8, dtype=int)
                probe = _inputs(datasets["iid"], codes)[:probe_n]
                initial_kc, initial_mb, _ = _forward(probe, encoder, bias, pre, post, w0, initial_head)
                snapshots = {}
                for t in checkpoints:
                    snap = _npz(path / f"checkpoint_{t:04d}.npz", _SAVED_FIELDS)
                    snapshots[t] = snap
                    snapshot_count += 1
                    for field, expected in (("encoder", encoder), ("bias", bias), ("pre", pre), ("post", post)):
                        _require(np.array_equal(snap[field], expected), f"Frozen encoder/bias/topology: {field}")
                    weights, head = snap["internal_weights"], snap["readout_weights"]
                    _require(weights.shape == w0.shape and head.shape == initial_head.shape and all(np.isfinite(snap[f]).all() for f in _SAVED_FIELDS), "Finite snapshot parameter schema")
                    _require(np.all((weights >= 0) & (weights <= 2 * w0 + 1e-12)), "Internal clipping bounds")
                    if not plastic:
                        _require(np.array_equal(weights, w0), "Fixed internal weights changed")
                    kc, mb, probability = _forward(probe, encoder, bias, pre, post, weights, head)
                    for field, expected in (("kc_activity", kc), ("mbon_activity", mb), ("choice_probability", probability)):
                        _near(snap[field], expected, f"Saved probe state: {field}")
                    changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)), "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                               "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)), "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb))}
                    for split in ("iid", "ood"):
                        data = datasets[split]
                        _, _, p = _forward(_inputs(data, codes), encoder, bias, pre, post, weights, head)
                        metrics = _metrics(p, data["counts"])
                        measurements[seed, condition, fit, t, split] = metrics
                        row = curves[seed, condition, fit, t, split]
                        for field, expected in {**metrics, **changes}.items():
                            _near(_number(row, field), expected, f"Curve: {fit}/{t}/{split}/{field}")
                        if fit == "quantity_removed":
                            _near(metrics["p_target"], .5, "Quantity-removed IID/OOD probability chance", atol=1e-12)
                            _near(metrics["greedy_accuracy"], .5, "Quantity-removed IID/OOD greedy chance", atol=1e-12)
                    if t in (0, n):
                        stage = "baseline" if t == 0 else "final"
                        saved = _npz(path / f"transfer_{stage}.npz", _PRESENT_FIELDS if fit == "quantity_present" else _REMOVED_FIELDS)
                        transfer[seed, condition, fit, stage] = _transfer(saved, fit, datasets, panel, encoder, bias, pre, post, weights, head)
                        transfer_count += 1
                        matrix_entries += 720 * 27 if fit == "quantity_present" else 27
                weights, head = w0.copy(), initial_head.copy()
                _near(snapshots[0]["internal_weights"], weights, "Initialized internal weights")
                _near(snapshots[0]["readout_weights"], head, "Initialized readout weights")
                data = datasets["train"]
                train_kc = np.maximum(_inputs(data, codes) @ encoder.T + bias, 0)
                for i, kc in enumerate(train_kc):
                    row = trials[seed, condition, fit, i + 1]
                    _require(row["family_id"] == str(data["family_ids"][i]) and _integer(row, "left_count") == data["counts"][i, 0] and _integer(row, "right_count") == data["counts"][i, 1], "Shared training stimulus identity")
                    action, reward = _integer(row, "action"), _integer(row, "reward")
                    _require(action in (0, 1) and reward in (0, 1), "Binary action/reward")
                    mb = _mbon(kc, pre, post, weights, n_mbon)
                    probability = float(1 / (1 + np.exp(-np.clip((mb[1] - mb[0]) @ head, -40, 40))))
                    _near(_number(row, "p_right"), probability, "Pre-update trial probability")
                    _require(action == int(uniforms[i] < probability), "Regenerated shared action choice")
                    target = int(data["counts"][i, 1] > data["counts"][i, 0])
                    _require(reward == int(action == target), "Reward for larger true quantity")
                    factor = (action - probability) * (reward - .5)
                    _near(_number(row, "optimizer_factor"), factor, "Trial optimizer factor")
                    next_head = head + cfg["head_lr"] * factor * (mb[1] - mb[0])
                    eligibility = head[post] * (kc[1, pre] - kc[0, pre])
                    next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0) if plastic else weights.copy()
                    expected_trial = {"internal_step_l2": np.linalg.norm(next_weights - weights), "readout_step_l2": np.linalg.norm(next_head - head),
                                      "kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]), "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1])}
                    for field, expected in expected_trial.items():
                        _near(_number(row, field), expected, f"Manual trial update/activity: {field}")
                    if not plastic:
                        _require(_number(row, "internal_step_l2") == 0, "Fixed internal trial change")
                    weights, head = next_weights, next_head
                    if (i + 1) % every == 0:
                        _near(snapshots[i + 1]["internal_weights"], weights, "Replayed internal checkpoint; evaluation frozen")
                        _near(snapshots[i + 1]["readout_weights"], head, "Replayed head checkpoint; evaluation frozen")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, transfer, summary)
    report["counts"].update(artifact_hashes=len(hashes), seeds=len(seeds), conditions=len(conditions), fits_per_family=2,
                            actual_fits=len(seeds) * len(conditions) * 2, trials=trial_count, curve_rows=len(curves), checkpoints=snapshot_count,
                            transfer_files=transfer_count, permutation_panel_rows=720, transfer_probability_entries=matrix_entries,
                            primary_unseen_pairs=15, anchored_pairs=12, canonical_fingerprints=len(fingerprints))
    report["checks"].extend([
        "Required primary hashes, exact two-fit design, and local registration/run chronology",
        "All720 lexicographic anchored maps; bitwise-identical 2-vs-4 inputs under every map",
        "All30 raw splits regenerated from registered seeds; complete joint descriptor crossing and canonical disjointness",
        "Action uniforms and real/rewired/ANN resource checks; control topologies regenerated from their seeds",
        "Frozen encoder/bias/topology and distinct internal, readout and activation changes independently recomputed",
        "Every actual policy-gradient trial replayed with old-head eligibility, reward-only feedback and clipping",
        "Every baseline/final logit cache and every720x27 frozen panel probability/greedy entry independently recomputed",
        "Predeclared full-panel unseen chance and quantity-removed chance verified; not reported as discoveries",
        "Acquisition gates separate from ordered unseen transfer; paired gaps and all-seed/qualified family summaries recomputed",
        "Inverse-map analytic oracle marked privileged; code-order residual cues separately audited on primary15 and anchored12",
    ])


def verify(output: Path) -> dict:
    """Write a separate audit report while preserving every primary artifact."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib replay; no learner/data/experiment/encoding/summary helper imports",
              "input_scope": "artificial scalar codes and six crossed descriptors; exhaustive frozen-policy representation diagnostic",
              "counts": {}, "checks": [], "errors": [],
              "not_verified": [
                  "Visual numerosity extraction, LC11/DAN biological dynamics or shared human/fly mechanisms",
                  "Remote private Git publication authenticity; local chronology is self-recorded",
                  "Statistical significance, mathematical rule discovery, broad transfer or educational implications",
                  "720 maps as independent learners or seeds as independent biological animals",
              ]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
