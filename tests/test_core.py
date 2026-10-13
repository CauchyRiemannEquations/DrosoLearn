"""Independent invariants for stimuli, anatomical controls, and reward updates.

These checks target accidental cue leakage, silent source drift, forbidden
plasticity, and evaluation-time learning rather than final task success.
"""
from pathlib import Path
import copy
import csv
import json
import shutil

import numpy as np
import pytest

from drosolearn.data import (
    Dataset, NUISANCE_NAMES, analytical_accuracy, assert_disjoint, make_dataset,
)
from drosolearn.graph import Projection, load_projection, random_ann, rewire
from drosolearn.model import Learner
from drosolearn.experiment import evaluate, make_splits, register, run, validate_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def anatomical_projection():
    return load_projection(ROOT / "data" / "flywire_783_mb_left_kc512_s701")


def toy_projection():
    return Projection(
        pre=np.array([0, 1, 1, 2]),
        post=np.array([0, 0, 1, 1]),
        weights=np.array([2.0, 1.0, 3.0, 2.0]),
        n_kc=3,
        n_mbon=2,
        kc_ids=["a", "b", "c"],
        mbon_ids=["d", "e"],
        metadata={"scope": "test fixture"},
    )


@pytest.mark.parametrize("domain", ["iid", "ood"])
@pytest.mark.parametrize("pair", [(2, 4), (1, 3), (3, 4), (4, 8)])
def test_nuisance_is_exactly_crossed_with_quantity_and_side(domain, pair):
    data = make_dataset(19, "test", 64, pair=pair, domain=domain)
    assert len(NUISANCE_NAMES) == data.nuisance.shape[-1] == 6
    # Every entire nuisance vector, not merely every marginal coordinate,
    # appears at both quantities and sides within its crossed family.
    for family in set(data.family_ids):
        members = np.flatnonzero(np.asarray(data.family_ids) == family)
        assert len(members) == 4
        assignment = {}
        for trial in members:
            for side in (0, 1):
                key = data.nuisance[trial, side].tobytes()
                assignment.setdefault(key, []).append((int(data.counts[trial, side]), side))
        assert len(assignment) == 2
        for occurrences in assignment.values():
            assert sorted(occurrences) == sorted((count, side) for count in pair for side in (0, 1))
    assert np.mean(data.target()) == 0.5
    assert analytical_accuracy(data) == 1.0
    assert analytical_accuracy(data, larger=False) == 1.0
    for cue in range(6):
        assert analytical_accuracy(data, nuisance_index=cue) == 0.5
        assert analytical_accuracy(data, larger=False, nuisance_index=cue) == 0.5
    # A nonlinear combination of all nuisance coordinates also has no signal.
    nuisance_score = (np.sin(data.nuisance * np.arange(1, 7))
                      + data.nuisance ** 2).sum(axis=-1)
    nuisance_choice = np.argmax(nuisance_score, axis=1)
    assert np.mean(nuisance_choice == data.target()) == 0.5


def test_quantity_ablation_really_removes_oracle_coordinate():
    data = make_dataset(4, "train", 32)
    full, ablated = data.inputs(), data.inputs(include_quantity=False)
    np.testing.assert_array_equal(full[..., 0], data.counts / 8.0)
    np.testing.assert_array_equal(ablated[..., 0], np.zeros_like(data.counts))
    np.testing.assert_array_equal(full[..., 1:], ablated[..., 1:])


def test_nuisance_domains_and_independent_splits_are_frozen_and_disjoint():
    train = make_dataset(10, "train", 32)
    validation = make_dataset(11, "validation", 32)
    test = make_dataset(12, "test", 32, domain="ood")
    assert_disjoint([train, validation, test])
    assert np.all((train.nuisance >= 0.25) & (train.nuisance <= 0.75))
    assert np.all((test.nuisance <= 0.25) | (test.nuisance >= 0.75))
    repeated = make_dataset(10, "train", 32)
    np.testing.assert_array_equal(train.counts, repeated.counts)
    np.testing.assert_array_equal(train.nuisance, repeated.nuisance)
    assert train.family_ids == repeated.family_ids


def test_split_overlap_detection_ignores_names_and_left_right_orientation():
    train = make_dataset(31, "train", 32)
    # Renaming family IDs does not make identical stimuli independent.
    renamed = make_dataset(31, "unrelated-name", 32)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([train, renamed])
    swapped = Dataset(train.counts[:, ::-1].copy(), train.nuisance[:, ::-1].copy(),
                      [f"renamed:{i}" for i in range(len(train.counts))], "swapped")
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([train, swapped])


@pytest.mark.parametrize("n,pair,domain", [(3, (2, 4), "iid"), (4, (4, 2), "iid"),
                                           (4, (0, 4), "iid"), (4, (2, 9), "iid"),
                                           (4, (2, 4), "unknown")])
def test_invalid_stimulus_specification_rejected(n, pair, domain):
    with pytest.raises(ValueError):
        make_dataset(1, "bad", n, pair=pair, domain=domain)


def test_actual_projection_scope_and_inactive_mbon_outputs(anatomical_projection):
    graph = anatomical_projection
    assert (graph.n_kc, graph.n_mbon, len(graph.pre)) == (512, 48, 1474)
    assert len(set(graph.pre)) == 489
    assert len(set(graph.post)) == 29
    assert len(set(zip(graph.pre, graph.post))) == 1474
    assert np.all(graph.weights > 0)
    inactive = np.flatnonzero(np.bincount(graph.post, minlength=graph.n_mbon) == 0)
    assert len(inactive) == 19
    model = Learner(graph, seed=5, plastic=True)
    output = model.mbon(np.ones((2, graph.n_kc)))
    np.testing.assert_array_equal(output[:, inactive], np.zeros((2, 19)))
    # Incoming-normalized active MBONs have unit response to unit KC activity.
    active = np.setdiff1d(np.arange(graph.n_mbon), inactive)
    np.testing.assert_allclose(output[:, active], 1.0, atol=1e-12)


@pytest.mark.parametrize("filename", ["edges.csv", "neurons.json", "annotations.csv"])
def test_source_checksum_drift_is_rejected(tmp_path, filename):
    source = ROOT / "data" / "flywire_783_mb_left_kc512_s701"
    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    # Even harmless whitespace changes must fail the pinned source hash.
    with (copied / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_projection(copied)


def assert_incoming_preserved(original, control):
    assert len(control.pre) == len(original.pre)
    assert (control.n_kc, control.n_mbon) == (original.n_kc, original.n_mbon)
    assert control.kc_ids == original.kc_ids
    assert control.mbon_ids == original.mbon_ids
    assert len(set(zip(control.pre, control.post))) == len(control.pre)
    np.testing.assert_array_equal(np.bincount(control.post, minlength=control.n_mbon),
                                  np.bincount(original.post, minlength=original.n_mbon))
    for target in range(original.n_mbon):
        np.testing.assert_array_equal(np.sort(control.weights[control.post == target]),
                                      np.sort(original.weights[original.post == target]))


def test_degree_preserving_rewiring_changes_graph_without_modifying_source(anatomical_projection):
    graph = anatomical_projection
    source_pre, source_post, source_weights = graph.pre.copy(), graph.post.copy(), graph.weights.copy()
    control = rewire(graph, seed=8)
    assert_incoming_preserved(graph, control)
    np.testing.assert_array_equal(np.bincount(control.pre, minlength=control.n_kc),
                                  np.bincount(graph.pre, minlength=graph.n_kc))
    assert set(zip(control.pre, control.post)) != set(zip(graph.pre, graph.post))
    assert control.metadata["accepted_swaps"] == 10 * len(graph.pre)
    repeated = rewire(graph, seed=8)
    np.testing.assert_array_equal(control.pre, repeated.pre)
    np.testing.assert_array_equal(graph.pre, source_pre)
    np.testing.assert_array_equal(graph.post, source_post)
    np.testing.assert_array_equal(graph.weights, source_weights)


def test_random_ann_preserves_postsynaptic_resources_but_not_kc_degree(anatomical_projection):
    graph = anatomical_projection
    control = random_ann(graph, seed=8)
    assert_incoming_preserved(graph, control)
    assert not np.array_equal(np.bincount(control.pre, minlength=control.n_kc),
                              np.bincount(graph.pre, minlength=graph.n_kc))


def test_frozen_model_and_nonplastic_model_have_different_update_scope():
    graph = toy_projection()
    data = make_dataset(7, "train", 4).inputs()[0]
    frozen = Learner(graph, seed=2, plastic=True, learning=False)
    initial = frozen.snapshot(data)
    frozen.update(data, action=1, reward=1)
    for key, value in initial.items():
        np.testing.assert_array_equal(frozen.snapshot(data)[key], value)
    assert frozen.head_updates == frozen.internal_updates == 0
    fixed = Learner(graph, seed=2, plastic=False)
    initial_weights, initial_head = fixed.weights.copy(), fixed.head.copy()
    fixed.update(data, action=1, reward=1)
    np.testing.assert_array_equal(fixed.weights, initial_weights)
    assert not np.array_equal(fixed.head, initial_head)
    assert fixed.head_updates == 1 and fixed.internal_updates == 0


def test_internal_only_condition_preserves_readout_but_changes_internal_synapses():
    model = Learner(toy_projection(), seed=2, plastic=True, head_learning=False)
    inputs = make_dataset(7, "train", 4).inputs()[0]
    initial_head, initial_weights = model.head.copy(), model.weights.copy()
    model.update(inputs, action=1, reward=1)
    np.testing.assert_array_equal(model.head, initial_head)
    assert not np.array_equal(model.weights, initial_weights)
    assert model.head_updates == 0 and model.internal_updates == 1


def test_plastic_update_uses_only_existing_1474_edges_and_preserves_fixed_encoder(anatomical_projection):
    graph = anatomical_projection
    model = Learner(graph, seed=3, plastic=True)
    original_encoder, original_bias = model.encoder.copy(), model.bias.copy()
    original_pre, original_post = graph.pre.copy(), graph.post.copy()
    original_weights = model.weights.copy()
    inputs = make_dataset(3, "train", 4).inputs()[0]
    model.update(inputs, action=0, reward=0)
    assert model.weights.shape == (1474,)
    assert np.all(model.weights >= 0) and np.all(model.weights <= model.upper)
    assert not np.array_equal(model.weights, original_weights)
    assert model.head_updates == model.internal_updates == 1
    np.testing.assert_array_equal(graph.pre, original_pre)
    np.testing.assert_array_equal(graph.post, original_post)
    np.testing.assert_array_equal(model.encoder, original_encoder)
    np.testing.assert_array_equal(model.bias, original_bias)


def test_evaluation_and_returned_snapshots_cannot_change_learning_state():
    model = Learner(toy_projection(), seed=1, plastic=True)
    data = make_dataset(2, "test", 8).inputs()
    weights, head, encoder = model.weights.copy(), model.head.copy(), model.encoder.copy()
    before = model.snapshot(data)
    p = model.probabilities(data)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    model.features(data)
    after = model.snapshot(data)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    after["internal_weights"][:] = -100
    after["readout_weights"][:] = -100
    np.testing.assert_array_equal(model.weights, weights)
    np.testing.assert_array_equal(model.head, head)
    np.testing.assert_array_equal(model.encoder, encoder)
    assert model.head_updates == model.internal_updates == 0


@pytest.mark.parametrize("action,reward", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_head_and_internal_reward_gradients_match_finite_difference(action, reward):
    model = Learner(toy_projection(), seed=6, plastic=True, head_lr=1e-4, internal_lr=1e-4)
    inputs = make_dataset(21, "gradient", 4).inputs()[0]
    # Stay away from clipping and use the same pre-update parameters for both
    # gradients: d[(reward - baseline) * log pi(action)] / d(parameter).
    initial_head, initial_weights = model.head.copy(), model.weights.copy()
    epsilon = 1e-6

    def objective():
        probability = float(model.probabilities(inputs))
        chosen = probability if action else 1 - probability
        return (reward - 0.5) * np.log(chosen)

    def finite_gradient(parameter):
        gradient = np.zeros_like(parameter)
        for coordinate in range(len(parameter)):
            value = parameter[coordinate]
            parameter[coordinate] = value + epsilon
            plus = objective()
            parameter[coordinate] = value - epsilon
            minus = objective()
            parameter[coordinate] = value
            gradient[coordinate] = (plus - minus) / (2 * epsilon)
        return gradient

    expected_head = finite_gradient(model.head)
    expected_internal = finite_gradient(model.weights)
    assert np.max(np.abs(expected_head)) > 0
    assert np.max(np.abs(expected_internal)) > 0
    model.update(inputs, action, reward)
    np.testing.assert_allclose((model.head - initial_head) / model.head_lr,
                               expected_head, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose((model.weights - initial_weights) / model.internal_lr,
                               expected_internal, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("action,reward", [(2, 1), (-1, 0), (1, -1), (0, 2)])
def test_invalid_reward_observation_rejected_without_parameter_mutation(action, reward):
    model = Learner(toy_projection(), seed=6, plastic=True)
    inputs = make_dataset(21, "train", 4).inputs()[0]
    weights, head = model.weights.copy(), model.head.copy()
    with pytest.raises(ValueError):
        model.update(inputs, action, reward)
    np.testing.assert_array_equal(model.weights, weights)
    np.testing.assert_array_equal(model.head, head)
    assert model.head_updates == model.internal_updates == 0


def smoke_config():
    return json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("key,value", [
    ("checkpoint_every", 0), ("checkpoint_every", -4), ("checkpoint_every", 3.5),
    ("trials_per_phase", 0), ("trials_per_phase", 402), ("trials_per_phase", 400.0),
    ("eval_trials", -4), ("eval_trials", 4.0),
    ("head_lr", float("nan")), ("internal_lr", -1),
    ("seeds", []), ("seeds", [1, 1]), ("seeds", [True]), ("seeds", [-1]),
    ("conditions", ["fixed", "fixed"]), ("conditions", ["unknown"]),
    ("train_pair", [4, 2]), ("train_pair", [2, 9]), ("train_pair", [2, 2]),
    ("train_pair", [2]), ("transfer_pairs", [[1, 3], [1, 3]]),
    ("transfer_pairs", [[0, 3]]),
])
def test_invalid_config_rejected_before_registration(key, value):
    config = smoke_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("key,value", [
    ("acquisition_accuracy", -0.1), ("acquisition_accuracy", float("nan")),
    ("acquisition_gain", 1.1), ("generalization_accuracy", float("inf")),
    ("reversal_accuracy", 1.2), ("confirmatory_seed_fraction", 0),
    ("confirmatory_seed_fraction", 1.1),
])
def test_invalid_criterion_rejected_before_registration(key, value):
    config = smoke_config()
    config["criteria"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_confirmatory_requires_ten_unique_registered_seeds():
    config = smoke_config()
    config["scope"] = "confirmatory"
    config["seeds"] = list(range(9))
    with pytest.raises(ValueError):
        validate_config(config)
    config["seeds"] = list(range(10))
    validate_config(config)


def test_every_split_is_reproducible_and_has_no_stimulus_leakage():
    config = smoke_config()
    first = make_splits(config, 101)
    repeated = make_splits(config, 101)
    assert_disjoint(list(first.values()))
    assert set(first) == {"train_acquisition", "train_reversal", "iid", "ood",
                          "transfer_1_3", "transfer_3_4", "transfer_4_8"}
    for name in first:
        np.testing.assert_array_equal(first[name].counts, repeated[name].counts)
        np.testing.assert_array_equal(first[name].nuisance, repeated[name].nuisance)
        assert first[name].family_ids == repeated[name].family_ids


def test_nuisance_only_policy_evaluates_at_exact_chance(anatomical_projection):
    data = make_dataset(2, "probe", 32)
    model = Learner(anatomical_projection, seed=3, plastic=True)
    # Remove the number coordinate at the model boundary without altering
    # targets. Exact crossed nuisance families force expected choice chance
    # for every deterministic nonlinear encoder/head, regardless of seed.
    model.encoder[:, 0] = 0
    metrics, _, _ = evaluate(model, data, larger=True)
    assert metrics["expected_reward"] == pytest.approx(0.5, abs=1e-12)
    assert metrics["accuracy"] == pytest.approx(0.5, abs=1e-12)


@pytest.fixture
def small_registered_run(tmp_path, monkeypatch):
    # A separate temporary root prevents synthetic integration-test outputs
    # from being mistaken for registered study runs in the real repository.
    root = tmp_path / "study"
    root.mkdir()
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "data", root / "data")
    (root / "docs").mkdir()
    (root / "docs" / "experimental-protocol.md").write_text("integration-test fixture\n", encoding="utf-8")
    shutil.copyfile(ROOT / "pyproject.toml", root / "pyproject.toml")
    config = copy.deepcopy(smoke_config())
    config.update(seeds=[31], conditions=["frozen"], trials_per_phase=8,
                  eval_trials=8, checkpoint_every=4)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    # Test source/config locking independently of whether Git is installed.
    monkeypatch.setattr("drosolearn.experiment.git_state", lambda _: {"commit": "fixture", "status": ""})
    registration = root / "registration.json"
    register(root, config_path, registration)
    return root, config_path, registration


@pytest.mark.parametrize("changed", ["config", "protocol", "code"])
def test_registered_config_protocol_and_code_drift_block_execution(small_registered_run, changed):
    root, config_path, registration = small_registered_run
    path = {"config": config_path,
            "protocol": root / "docs" / "experimental-protocol.md",
            "code": root / "src" / "drosolearn" / "model.py"}[changed]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "run-output"
    with pytest.raises(ValueError, match="changed after registration"):
        run(root, config_path, registration, output)
    assert not output.exists()


def test_registration_and_result_directories_cannot_overwrite(small_registered_run):
    root, config_path, registration = small_registered_run
    with pytest.raises(FileExistsError):
        register(root, config_path, registration)
    output = root / "already-exists"
    output.mkdir()
    (output / "preserve.txt").write_text("prior result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(root, config_path, registration, output)
    assert (output / "preserve.txt").read_text(encoding="utf-8") == "prior result"


def test_small_frozen_run_logs_both_phases_and_preserves_identical_weights(small_registered_run):
    root, config_path, registration = small_registered_run
    output = root / "run-output"
    outcomes = run(root, config_path, registration, output)
    assert len(outcomes) == 1 and outcomes[0]["condition"] == "frozen"
    assert outcomes[0]["head_updates"] == outcomes[0]["internal_updates"] == 0
    with (output / "trials.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row["phase"] for row in rows} == {"acquisition", "reversal"}
    for row in rows:
        assert float(row["internal_step_l2"]) == float(row["readout_step_l2"]) == 0
        assert row["reward"] == row["correct"]
    path = output / "seed_31" / "frozen"
    first = np.load(path / "acquisition_0000.npz")
    final = np.load(path / "reversal_0008.npz")
    for key in ("internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability"):
        np.testing.assert_array_equal(first[key], final[key])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["scope"] == "smoke"
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "trials.csv" in hashes and "registration.json" in hashes


def test_independent_full_audit_accepts_all_conditions(small_registered_run):
    from drosolearn.audit import verify
    root, config_path, _ = small_registered_run
    cfg = json.loads(config_path.read_text())
    cfg["conditions"] = smoke_config()["conditions"]
    config_path.write_text(json.dumps(cfg) + "\n", encoding="utf-8")
    registration = root / "all-condition-registration.json"
    register(root, config_path, registration)
    output = root / "all-conditions-output"
    run(root, config_path, registration, output)
    audited = verify(output)
    assert audited["passed"], audited["errors"]
    assert audited["counts"]["trials"] == 16 * len(cfg["conditions"])


@pytest.mark.parametrize("tamper", ["raw_bytes", "summary_with_rehashed_manifest"])
def test_independent_audit_detects_corruption(small_registered_run, tamper):
    from drosolearn.audit import verify
    root, config_path, registration = small_registered_run
    output = root / "audit-output"
    run(root, config_path, registration, output)
    if tamper == "raw_bytes":
        with (output / "trials.csv").open("a", encoding="utf-8") as stream:
            stream.write("corrupt\n")
    else:
        path = output / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["outcomes"][0]["gain"] = 0.999
        path.write_text(json.dumps(value), encoding="utf-8")
        manifest_path = output / "artifact_hashes.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["summary.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audited = verify(output)
    assert not audited["passed"] and audited["errors"]
