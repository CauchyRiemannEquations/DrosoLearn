"""Input information and transfer invariants, independent of scientific success.

The full permutation chance identity is an algebraic sanity gate. A deliberately
nonmonotone code/nuisance lookup policy tests that it does not assume the trained
policy is a scalar-order comparator.
"""
from itertools import combinations, permutations
from pathlib import Path
import copy
import csv
import json
import shutil

import numpy as np
import pytest

from drosolearn.data import assert_disjoint, make_dataset
from drosolearn.encoding import (
    ANCHORED, FAMILIES, FITS, PAIRS, PRIMARY, anchored_panel, candidate_cache,
    encoded_inputs, make_splits, panel_metrics, policy_metrics, register, run,
    study_hashes, transfer_evaluate, validate_config,
)
from drosolearn.graph import Projection
from drosolearn.model import Learner


FREE_COUNTS = (1, 3, 5, 6, 7, 8)
ROOT = Path(__file__).resolve().parents[1]


def independently_enumerated_panel():
    panel = []
    for values in permutations(FREE_COUNTS):
        mapping = {2: 2, 4: 4, **dict(zip(FREE_COUNTS, values))}
        panel.append([mapping[count] for count in range(1, 9)])
    return np.asarray(panel, dtype=np.int64)


def arbitrary_consistent_candidate_scores(dataset):
    """One arbitrary score per code and entire nuisance vector, reused faithfully.

    Arbitrarily assigning separate scores to crossed rows would violate the
    fixed-policy premise. Repeated identical inputs must have identical scores.
    """
    rng = np.random.default_rng(917)
    lookup = {}
    scores = np.empty((*dataset.counts.shape, 8))
    for row in range(len(dataset.counts)):
        for side in (0, 1):
            key = dataset.nuisance[row, side].tobytes()
            if key not in lookup:
                lookup[key] = rng.normal(0, 4, 8)
            scores[row, side] = lookup[key]
    return scores


def independent_panel_probabilities(dataset, scores, panel):
    indices = np.take(panel - 1, dataset.counts - 1, axis=1)
    selected = np.take_along_axis(scores[None, ...], indices[..., None], axis=-1)[..., 0]
    gap = selected[..., 1] - selected[..., 0]
    return 1 / (1 + np.exp(-gap))


@pytest.mark.parametrize("pair", list(combinations(FREE_COUNTS, 2)))
def test_complete_anchored_panel_chance_identity_for_arbitrary_nonlinear_policy(pair):
    dataset = make_dataset(312, "algebra", 16, pair=pair)
    panel = independently_enumerated_panel()
    scores = arbitrary_consistent_candidate_scores(dataset)
    probability = independent_panel_probabilities(dataset, scores, panel)
    correct = np.where(dataset.target()[None, :] == 1, probability, 1 - probability)
    np.testing.assert_allclose(correct.mean(), 0.5, rtol=0, atol=2e-15)
    # The identity is about the full panel; individual mappings need not be at
    # chance, and this deliberately nonlinear policy makes that difference clear.
    assert np.ptp(correct.mean(axis=1)) > 0.1


def test_anchor_order_shortcut_means_are_not_all_chance():
    panel = independently_enumerated_panel()

    def oracle_mean(pairs):
        return np.mean([np.mean(panel[:, a - 1] < panel[:, b - 1]) for a, b in pairs])

    unseen = list(combinations(FREE_COUNTS, 2))
    mixed = [pair for pair in combinations(range(1, 9), 2)
             if len(set(pair) & {2, 4}) == 1]
    all_transfer = [pair for pair in combinations(range(1, 9), 2) if pair != (2, 4)]
    assert len(unseen) == 15 and len(mixed) == 12 and len(all_transfer) == 27
    np.testing.assert_allclose(oracle_mean(unseen), 1 / 2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(oracle_mean(mixed), 23 / 36, rtol=0, atol=1e-15)
    np.testing.assert_allclose(oracle_mean(all_transfer), 91 / 162, rtol=0, atol=1e-15)


def encoding_fixture_config():
    cfg = json.loads((ROOT / "configs/encoding_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[41], conditions=list(FAMILIES),
               trials=8, checkpoint_every=4, eval_trials=8, probe_rows=4)
    return cfg


def prepare_encoding_fixture(path):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = encoding_fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/encoding-protocol.md", "docs/encoding-background.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(encoding_fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_encoding_fixture(tmp_path_factory):
    root, config, registration = prepare_encoding_fixture(tmp_path_factory.mktemp("encoding"))
    output = root / "fixture-output"
    summary = run(root, config, registration, output)
    return root, config, registration, output, summary


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def toy_encoding_model(plastic=True):
    graph = Projection(
        pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
        weights=np.array([2.0, 1.0, 3.0, 2.0]), n_kc=3, n_mbon=2,
        kc_ids=["a", "b", "c"], mbon_ids=["d", "e"], metadata={"scope": "test fixture"},
    )
    return Learner(graph, seed=19, plastic=plastic)


def test_public_panel_is_exhaustive_lexicographic_bijective_and_anchored():
    panel = anchored_panel()
    np.testing.assert_array_equal(panel, independently_enumerated_panel())
    assert panel.shape == (720, 8) and len({row.tobytes() for row in panel}) == 720
    np.testing.assert_array_equal(panel[0], np.arange(1, 9))
    assert np.all(panel[:, 1] == 2) and np.all(panel[:, 3] == 4)
    np.testing.assert_array_equal(np.sort(panel, axis=1), np.tile(np.arange(1, 9), (720, 1)))
    assert len(PAIRS) == 27 and (2, 4) not in PAIRS
    assert {PAIRS[i] for i in PRIMARY} == set(combinations(FREE_COUNTS, 2))
    assert len(ANCHORED) == 12 and set(PRIMARY) | set(ANCHORED) == set(range(27))


@pytest.mark.parametrize("domain", ["iid", "ood"])
def test_all_permutations_leave_anchors_and_nuisance_inputs_byte_identical(domain):
    data = make_dataset(711, "anchors", 16, domain=domain)
    ordered = data.inputs()
    before_counts, before_nuisance = data.counts.copy(), data.nuisance.copy()
    for mapping in anchored_panel():
        inputs = encoded_inputs(data, mapping)
        assert inputs.tobytes() == ordered.tobytes()
    removed = encoded_inputs(data, np.zeros(8, dtype=int))
    assert np.all(removed[..., 0] == 0)
    np.testing.assert_array_equal(removed[..., 1:], ordered[..., 1:])
    np.testing.assert_array_equal(data.counts, before_counts)
    np.testing.assert_array_equal(data.nuisance, before_nuisance)


def test_encoding_changes_only_the_scalar_for_unseen_counts():
    data = make_dataset(77, "unseen", 8, pair=(1, 8))
    mapping = anchored_panel()[-1]
    actual = encoded_inputs(data, mapping)
    np.testing.assert_array_equal(actual[..., 0], mapping[data.counts - 1] / 8)
    np.testing.assert_array_equal(actual[..., 1:], data.inputs()[..., 1:])
    assert not np.array_equal(actual[..., 0], data.inputs()[..., 0])
    with pytest.raises(ValueError, match="eight"):
        encoded_inputs(data, [1, 2])


def test_new_splits_are_reproducible_disjoint_complete_and_pair_balanced():
    cfg = encoding_fixture_config()
    first, repeated = make_splits(cfg, 41), make_splits(cfg, 41)
    assert_disjoint(list(first.values()))
    assert set(first) == {"train", "iid", "ood", *(f"pair_{a}_{b}" for a, b in PAIRS)}
    for name, dataset in first.items():
        np.testing.assert_array_equal(dataset.counts, repeated[name].counts)
        np.testing.assert_array_equal(dataset.nuisance, repeated[name].nuisance)
        assert dataset.family_ids == repeated[name].family_ids
        assert np.mean(dataset.target()) == 0.5
    assert np.all((first["ood"].nuisance <= 0.25) | (first["ood"].nuisance >= 0.75))


def test_score_cache_and_panel_metrics_match_direct_nonlinear_forward_evaluation():
    datasets = make_splits(encoding_fixture_config(), 41)
    model = toy_encoding_model()
    train = datasets["train"]
    for row, inputs in enumerate(train.inputs()):
        model.update(inputs, action=row % 2, reward=int(row % 2 == train.target()[row]))
    probe = datasets["iid"].inputs()
    before = model.snapshot(probe)
    counters = model.head_updates, model.internal_updates
    cache = candidate_cache(model, datasets, np.arange(1, 9))
    assert cache.shape == (27, 8, 2, 8)
    panel = anchored_panel()
    means, greedy, ordered = panel_metrics(cache, datasets, panel)
    for j, (a, b) in enumerate(PAIRS):
        dataset = datasets[f"pair_{a}_{b}"]
        for code in (1, 4, 8):
            inputs = encoded_inputs(dataset, np.full(8, code))
            _, mbon = model.features(inputs)
            np.testing.assert_allclose(cache[j, ..., code - 1], mbon @ model.head, rtol=0, atol=1e-14)
        for map_id in (0, 37, 719):
            direct = policy_metrics(model, dataset, panel[map_id])
            np.testing.assert_allclose(means[map_id, j], direct["p_target"], rtol=0, atol=1e-14)
            np.testing.assert_allclose(greedy[map_id, j], direct["greedy_accuracy"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(ordered[j], model.probabilities(dataset.inputs()), rtol=0, atol=1e-14)
    evaluated = transfer_evaluate(model, datasets, "quantity_present", panel)
    np.testing.assert_allclose(evaluated["map_p_target"][:, PRIMARY].mean(axis=0), 0.5, rtol=0, atol=2e-15)
    after = model.snapshot(probe)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert (model.head_updates, model.internal_updates) == counters


def test_actual_panel_cache_satisfies_arbitrary_lookup_policy_identity():
    datasets = make_splits(encoding_fixture_config(), 42)
    caches = np.asarray([arbitrary_consistent_candidate_scores(datasets[f"pair_{a}_{b}"])
                         for a, b in PAIRS])
    panel = anchored_panel()
    means, greedy, ordered = panel_metrics(caches, datasets, panel)
    assert means.shape == greedy.shape == (720, 27)
    for j, (a, b) in enumerate(PAIRS):
        dataset = datasets[f"pair_{a}_{b}"]
        direct = independent_panel_probabilities(dataset, caches[j], panel)
        expected = np.where(dataset.target()[None, :] == 1, direct, 1 - direct).mean(axis=1)
        np.testing.assert_allclose(means[:, j], expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(ordered[j], direct[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(means[:, PRIMARY].mean(axis=0), 0.5, rtol=0, atol=2e-15)


def test_quantity_removed_transfer_stays_chance_after_actual_updates():
    datasets = make_splits(encoding_fixture_config(), 42)
    model = toy_encoding_model()
    train = encoded_inputs(datasets["train"], np.zeros(8))
    for row, inputs in enumerate(train):
        model.update(inputs, row % 2, row % 2)
    evaluated = transfer_evaluate(model, datasets, "quantity_removed", anchored_panel())
    assert evaluated["score_cache"].shape == (27, 8, 2, 1)
    np.testing.assert_allclose(evaluated["p_target"], 0.5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(evaluated["greedy_accuracy"], 0.5, rtol=0, atol=1e-15)
    for j, (a, b) in enumerate(PAIRS):
        dataset = datasets[f"pair_{a}_{b}"]
        np.testing.assert_allclose(evaluated["p_right"][j], model.probabilities(encoded_inputs(dataset, np.zeros(8))), rtol=0, atol=1e-14)


@pytest.mark.parametrize("field,value", [
    ("scope", "smoke"), ("seeds", []), ("seeds", [41, 41]), ("seeds", [True]),
    ("train_pair", [1, 3]), ("fits", list(reversed(FITS))),
    ("conditions", ["frozen"]), ("conditions", ["fixed", "fixed"]),
    ("trials", 6), ("trials", 8.0), ("checkpoint_every", 3), ("eval_trials", 6),
    ("probe_rows", 9), ("head_lr", float("nan")), ("internal_lr", -1),
])
def test_invalid_encoding_design_rejected(field, value):
    cfg = encoding_fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_diagnostic_requires_ten_registered_seeds_and_valid_thresholds():
    cfg = encoding_fixture_config()
    cfg["scope"] = "preregistered_diagnostic"
    with pytest.raises(ValueError, match="ten"):
        validate_config(cfg)
    cfg["seeds"] = list(range(3001, 3011))
    validate_config(cfg)
    for key, bad in (("representation_gap", float("nan")), ("positive_seed_fraction", 0), ("transfer_accuracy", 1.1)):
        invalid = copy.deepcopy(cfg)
        invalid["criteria"][key] = bad
        with pytest.raises(ValueError):
            validate_config(invalid)


@pytest.mark.parametrize("changed", ["config", "protocol", "background", "code"])
def test_encoding_freeze_blocks_config_protocol_background_and_source_drift(tmp_path, changed):
    root, config, registration = prepare_encoding_fixture(tmp_path)
    path = {"config": config, "protocol": root / "docs/encoding-protocol.md",
            "background": root / "docs/encoding-background.md",
            "code": root / "src/drosolearn/model.py"}[changed]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError, match="changed after registration"):
        run(root, config, registration, output)
    assert not output.exists()


def test_encoding_registration_and_existing_results_are_never_overwritten(tmp_path):
    root, config, registration = prepare_encoding_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        register(root, config, registration)
    output = root / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("old result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(root, config, registration, output)
    assert marker.read_text(encoding="utf-8") == "old result"


def test_small_run_records_two_actual_fits_not_720_trainings(completed_encoding_fixture):
    _, config, _, output, summary = completed_encoding_fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rows = csv_rows(output / "seed_41/trials.csv")
    assert len(rows) == summary["actual_training_trials"] == 96
    assert len(summary["outcomes"]) == len(FAMILIES) * 2
    assert summary["permutations"] == 720
    for condition in FAMILIES:
        for fit in FITS:
            selected = [r for r in rows if r["condition"] == condition and r["fit"] == fit]
            assert [int(r["trial"]) for r in selected] == list(range(1, 9))
            for row in selected:
                target = int(int(row["right_count"]) > int(row["left_count"]))
                assert int(row["reward"]) == int(int(row["action"]) == target)
                assert np.isfinite(float(row["p_right"])) and 0 <= float(row["p_right"]) <= 1
            outcomes = [r for r in summary["outcomes"] if r["condition"] == condition and r["fit"] == fit]
            assert outcomes[0]["head_updates"] == cfg["trials"]
            assert outcomes[0]["internal_updates"] == (cfg["trials"] if "plastic" in condition else 0)
    code_audit = json.loads((output / "code_audit.json").read_text(encoding="utf-8"))
    assert code_audit["code_order_oracle_unseen"] == 0.5
    np.testing.assert_allclose(code_audit["code_order_oracle_anchored"], 23 / 36, rtol=0, atol=1e-15)
    np.testing.assert_allclose(code_audit["code_order_oracle_all"], 91 / 162, rtol=0, atol=1e-15)
    assert code_audit["identity_aware_inverse_oracle"] == 1 and code_audit["oracle_is_privileged"]


def test_small_run_preserves_update_scopes_and_all_transfer_artifacts(completed_encoding_fixture):
    root, config, _, output, _ = completed_encoding_fixture
    curves = csv_rows(output / "learning_curves.csv")
    assert len(curves) == 6 * 2 * 3 * 2
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {"registration.json", "metadata.json", "summary.json", "permutations.npy"} <= set(hashes)
    recorded = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert recorded["source_hashes"] == study_hashes(root, json.loads(config.read_text(encoding="utf-8")))
    for condition in FAMILIES:
        for fit in FITS:
            path = output / "seed_41" / condition / fit
            with np.load(path / "checkpoint_0000.npz") as initial, np.load(path / "checkpoint_0008.npz") as final:
                for key in ("encoder", "bias", "pre", "post", "kc_activity"):
                    np.testing.assert_array_equal(initial[key], final[key])
                assert not np.array_equal(initial["readout_weights"], final["readout_weights"])
                if "plastic" in condition:
                    assert not np.array_equal(initial["internal_weights"], final["internal_weights"])
                else:
                    np.testing.assert_array_equal(initial["internal_weights"], final["internal_weights"])
            for stage in ("baseline", "final"):
                with np.load(path / f"transfer_{stage}.npz") as transfer:
                    if fit == "quantity_present":
                        assert transfer["score_cache"].shape == (27, 8, 2, 8)
                        assert transfer["map_p_target"].shape == (720, 27)
                        np.testing.assert_allclose(transfer["map_p_target"][:, PRIMARY].mean(axis=0), 0.5, rtol=0, atol=1e-14)
                    else:
                        assert transfer["score_cache"].shape == (27, 8, 2, 1)
                        np.testing.assert_allclose(transfer["p_target"], 0.5, rtol=0, atol=1e-14)
    for row in curves:
        if row["fit"] == "quantity_removed":
            np.testing.assert_allclose(float(row["p_target"]), 0.5, rtol=0, atol=1e-14)
