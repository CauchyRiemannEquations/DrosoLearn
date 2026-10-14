"""Controlled input spacing, direct policies and prospective freeze invariants.

These tests use tiny integration fixtures, never any registered scientific seed.
Geometry checks reconstruct the four maps independently. A full KC -> MBON ->
readout forward pass checks the accelerated evaluation without using its algebra.
"""
from itertools import combinations
from pathlib import Path
import copy
import csv
import json
import shutil

import numpy as np
import pytest

from drosolearn import spacing as study
from drosolearn.data import analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.model import Learner


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_ALPHAS = np.array([0.25, 0.5, 1.0, 1.5])
INDEPENDENT_PAIRS = tuple(pair for pair in combinations(range(1, 9), 2) if pair != (2, 4))


def independently_constructed_panel():
    return np.array([[1, 2, 3, 4, 6 - alpha, 6, 6 + alpha, 8]
                     for alpha in INDEPENDENT_ALPHAS], dtype=float)


def spacing_fixture_config():
    cfg = json.loads((ROOT / "configs/spacing_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[61], conditions=list(study.FAMILIES),
               trials=8, checkpoint_every=4, eval_trials=8, probe_rows=4)
    return cfg


def prepare_spacing_fixture(path):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = spacing_fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/spacing-protocol.md", "docs/spacing-background.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(spacing_fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    study.register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_spacing_fixture(tmp_path_factory):
    root, config, registration = prepare_spacing_fixture(tmp_path_factory.mktemp("spacing"))
    output = root / "fixture-output"
    summary = study.run(root, config, registration, output)
    return root, config, registration, output, summary


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def toy_spacing_model(plastic=True):
    graph = Projection(
        pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
        weights=np.array([2.0, 1.0, 3.0, 2.0]), n_kc=3, n_mbon=2,
        kc_ids=["a", "b", "c"], mbon_ids=["d", "e"], metadata={"scope": "test fixture"},
    )
    return Learner(graph, seed=19, plastic=plastic)


def test_spacing_panel_preserves_order_anchors_range_and_focal_midpoint():
    panel = study.spacing_panel()
    np.testing.assert_array_equal(panel, independently_constructed_panel())
    assert panel.shape == (4, 8)
    assert np.all(np.diff(panel, axis=1) > 0)
    for count in (1, 2, 3, 4, 6, 8):
        np.testing.assert_array_equal(panel[:, count - 1], np.full(4, count))
    np.testing.assert_array_equal(panel[:, 6] - panel[:, 4], 2 * INDEPENDENT_ALPHAS)
    np.testing.assert_array_equal((panel[:, 6] + panel[:, 4]) / 2, np.full(4, 6))
    np.testing.assert_array_equal(panel.min(axis=1), np.ones(4))
    np.testing.assert_array_equal(panel.max(axis=1), np.full(4, 8))
    np.testing.assert_array_equal(panel[2], np.arange(1, 9))
    assert tuple(study.PAIRS) == INDEPENDENT_PAIRS
    assert len(study.PAIRS) == 27


@pytest.mark.parametrize("split,domain", [("train", "iid"), ("iid", "iid"), ("ood", "ood")])
def test_every_spacing_map_leaves_training_and_anchor_evaluations_byte_identical(split, domain):
    dataset = make_dataset(61, split, 8, domain=domain)
    initial_counts, initial_nuisance = dataset.counts.copy(), dataset.nuisance.copy()
    for mapping in independently_constructed_panel():
        assert study.encoded_inputs(dataset, mapping).tobytes() == dataset.inputs().tobytes()
    np.testing.assert_array_equal(dataset.counts, initial_counts)
    np.testing.assert_array_equal(dataset.nuisance, initial_nuisance)


def test_focal_pair_changes_only_spacing_while_labels_nuisance_and_midpoint_stay_fixed():
    dataset = make_dataset(62, "focal", 8, pair=(5, 7))
    for alpha, mapping in zip(INDEPENDENT_ALPHAS, independently_constructed_panel()):
        inputs = study.encoded_inputs(dataset, mapping)
        np.testing.assert_array_equal(inputs[..., 1:], dataset.inputs()[..., 1:])
        np.testing.assert_allclose(np.mean(inputs[..., 0], axis=1), 6 / 8, rtol=0, atol=0)
        np.testing.assert_allclose(np.abs(np.diff(inputs[..., 0], axis=1))[:, 0],
                                   2 * alpha / 8, rtol=0, atol=0)
        np.testing.assert_array_equal(np.argmax(inputs[..., 0], axis=1), dataset.target())
    with pytest.raises(ValueError):
        study.encoded_inputs(dataset, [1, 2])


def test_new_splits_are_reproducible_disjoint_complete_balanced_and_nuisance_crossed():
    cfg = spacing_fixture_config()
    first, repeated = study.make_splits(cfg, 61), study.make_splits(cfg, 61)
    assert_disjoint(list(first.values()))
    assert set(first) == {"train", "iid", "ood", *(f"pair_{a}_{b}" for a, b in INDEPENDENT_PAIRS)}
    for name, dataset in first.items():
        np.testing.assert_array_equal(dataset.counts, repeated[name].counts)
        np.testing.assert_array_equal(dataset.nuisance, repeated[name].nuisance)
        assert dataset.family_ids == repeated[name].family_ids
        assert np.mean(dataset.target()) == 0.5
        assert analytical_accuracy(dataset) == 1.0
        assert all(analytical_accuracy(dataset, nuisance_index=i) == 0.5 for i in range(6))
    assert np.all((first["ood"].nuisance <= 0.25) | (first["ood"].nuisance >= 0.75))


def test_all_four_maps_and_all_pairs_match_full_forward_and_preserve_model_state():
    datasets = study.make_splits(spacing_fixture_config(), 61)
    model = toy_spacing_model()
    for row, inputs in enumerate(datasets["train"].inputs()):
        action = row % 2
        model.update(inputs, action, int(action == datasets["train"].target()[row]))
    probe = datasets["iid"].inputs()
    before = model.snapshot(probe)
    counters = model.head_updates, model.internal_updates
    panel = independently_constructed_panel()
    code_values = np.unique(panel)
    cache = study.candidate_cache(model, datasets, code_values)
    assert cache.shape == (27, 8, 2, 14)
    means, greedy, cases = study.panel_metrics(cache, datasets, panel, code_values)
    assert means.shape == greedy.shape == (4, 27)
    assert cases.shape == (4, 27, 8)
    for j, (a, b) in enumerate(INDEPENDENT_PAIRS):
        dataset = datasets[f"pair_{a}_{b}"]
        for code_index, code in enumerate(code_values):
            inputs = study.encoded_inputs(dataset, np.full(8, code))
            _, mbon = model.features(inputs)
            np.testing.assert_allclose(cache[j, ..., code_index], mbon @ model.head, rtol=0, atol=1e-14)
        for map_id, mapping in enumerate(panel):
            inputs = study.encoded_inputs(dataset, mapping)
            direct_probability = model.probabilities(inputs)
            target = dataset.target()
            expected = np.where(target == 1, direct_probability, 1 - direct_probability).mean()
            expected_greedy = np.where(np.abs(direct_probability - .5) < 1e-12, .5,
                                       ((direct_probability > .5).astype(int) == target).astype(float)).mean()
            np.testing.assert_allclose(means[map_id, j], expected, rtol=0, atol=1e-14)
            np.testing.assert_allclose(greedy[map_id, j], expected_greedy, rtol=0, atol=1e-14)
            np.testing.assert_allclose(cases[map_id, j], direct_probability, rtol=0, atol=1e-14)
            direct = study.policy_metrics(model, dataset, mapping)
            np.testing.assert_allclose(direct["p_target"], expected, rtol=0, atol=1e-14)
            np.testing.assert_allclose(direct["greedy_accuracy"], expected_greedy, rtol=0, atol=1e-14)
    evaluated = study.transfer_evaluate(model, datasets, panel)
    np.testing.assert_array_equal(evaluated["code_values"], code_values)
    for actual, expected in ((evaluated["score_cache"], cache), (evaluated["map_p_target"], means),
                             (evaluated["map_greedy_accuracy"], greedy), (evaluated["map_p_right"], cases)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    after = model.snapshot(probe)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert (model.head_updates, model.internal_updates) == counters


def test_panel_metrics_handles_nonmonotone_scores_fractional_code_lookup_and_ties():
    datasets = study.make_splits(spacing_fixture_config(), 62)
    panel = independently_constructed_panel()
    levels = np.unique(panel)
    # Scores depend on the exact fractional scalar and all nuisance values. This
    # policy deliberately has no assumption of order, linearity or acquisition.
    caches = []
    for pair in INDEPENDENT_PAIRS:
        ds = datasets[f"pair_{pair[0]}_{pair[1]}"]
        nuisance_term = (ds.nuisance ** 2).sum(axis=-1)
        caches.append(np.sin(levels * 3.7)[None, None, :] + nuisance_term[..., None])
    caches = np.asarray(caches)
    means, greedy, cases = study.panel_metrics(caches, datasets, panel, levels)
    for map_id, mapping in enumerate(panel):
        for j, pair in enumerate(INDEPENDENT_PAIRS):
            ds = datasets[f"pair_{pair[0]}_{pair[1]}"]
            mapped = mapping[ds.counts - 1]
            score = np.sin(mapped * 3.7) + (ds.nuisance ** 2).sum(axis=-1)
            probability = 1 / (1 + np.exp(-np.clip(score[:, 1] - score[:, 0], -40, 40)))
            target = ds.target()
            np.testing.assert_allclose(cases[map_id, j], probability, rtol=0, atol=1e-14)
            np.testing.assert_allclose(means[map_id, j], np.where(target == 1, probability, 1 - probability).mean(), rtol=0, atol=1e-14)
            np.testing.assert_allclose(greedy[map_id, j], ((probability > .5).astype(int) == target).mean(), rtol=0, atol=1e-14)
    tied = study.panel_metrics(np.zeros_like(caches), datasets, panel, levels)
    for metric in tied:
        np.testing.assert_array_equal(metric, np.full(metric.shape, .5))


@pytest.mark.parametrize("field,value", [
    ("scope", "smoke"), ("seeds", []), ("seeds", [61, 61]), ("seeds", [True]),
    ("train_pair", [1, 3]), ("primary_pair", [3, 5]),
    ("code_alphas", [0.5, 0.25, 1.0, 1.5]), ("code_alphas", [0.25, 0.5, 1.0]),
    ("code_alphas", [0.25, 0.5, 1.0, 2.0]),
    ("conditions", ["frozen"]), ("conditions", ["fixed", "fixed"]),
    ("trials", 6), ("trials", 8.0), ("checkpoint_every", 3), ("eval_trials", 6),
    ("probe_rows", 9), ("head_lr", float("nan")), ("internal_lr", -1),
])
def test_invalid_spacing_design_rejected(field, value):
    cfg = spacing_fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_diagnostic_requires_ten_registered_seeds_and_finite_valid_thresholds():
    cfg = spacing_fixture_config()
    cfg["scope"] = "preregistered_diagnostic"
    with pytest.raises(ValueError, match="[Tt]en"):
        study.validate_config(cfg)
    # Validate the scientific configuration without fitting its registered
    # seeds. A small engineering fixture cannot silently become this study by
    # changing only its scope and seed list.
    cfg = json.loads((ROOT / "configs/spacing_diagnostic.json").read_text(encoding="utf-8"))
    study.validate_config(cfg)
    for field, changed in (("seeds", list(range(4002, 4012))), ("trials", 200),
                           ("checkpoint_every", 50), ("eval_trials", 64),
                           ("probe_rows", 4), ("head_lr", .2), ("internal_lr", .02)):
        invalid = copy.deepcopy(cfg)
        invalid[field] = changed
        with pytest.raises(ValueError):
            study.validate_config(invalid)
    for key, changed in (("acquisition_accuracy", .8), ("acquisition_gain", .2),
                         ("transfer_accuracy", .8), ("sensitivity_increase", .05),
                         ("positive_seed_fraction", .7), ("rank_accuracy", .95)):
        invalid = copy.deepcopy(cfg)
        invalid["criteria"][key] = changed
        with pytest.raises(ValueError):
            study.validate_config(invalid)
    for key, bad in (("sensitivity_increase", float("nan")), ("positive_seed_fraction", 0),
                     ("transfer_accuracy", 1.1), ("rank_accuracy", -0.1)):
        invalid = copy.deepcopy(cfg)
        invalid["criteria"][key] = bad
        with pytest.raises(ValueError):
            study.validate_config(invalid)


def synthetic_summary_inputs(cfg, cases):
    """Build arithmetic fixtures without any learner or policy evaluator."""
    curves, transfers = [], {}
    focal = INDEPENDENT_PAIRS.index((5, 7))
    for seed, case in zip(cfg["seeds"], cases):
        for split in ("iid", "ood"):
            for trial, value in ((0, case["iid_before"]), (cfg["trials"], case["iid_after"])):
                curves.append(dict(seed=seed, condition="fixed", trial=trial, split=split, p_target=value))
        for stage in ("baseline", "final"):
            means = np.full((4, 27), case.get(f"{stage}_all", .8 if stage == "final" else .5))
            means[:, focal] = case[stage]
            greedy = np.full((4, 27), case.get("greedy_after", 1.0) if stage == "final" else .5)
            transfers[seed, "fixed", stage] = {"map_p_target": means, "map_greedy_accuracy": greedy}
    return curves, transfers


def test_adjusted_sensitivity_includes_failed_acquisition_and_preserves_separate_gates():
    cfg = spacing_fixture_config()
    cfg.update(seeds=[61, 62], conditions=["fixed"])
    cases = [
        dict(iid_before=.5, iid_after=.8, baseline=[.51, .52, .63, .64], final=[.71, .72, .85, .86]),
        dict(iid_before=.5, iid_after=.6, baseline=[.5, .5, .5, .5], final=[.7, .7, .9, .9]),
    ]
    curves, transfers = synthetic_summary_inputs(cfg, cases)
    summary = study.summarize(cfg, curves, transfers)
    first, second = summary["outcomes"]
    assert first["acquired"] and not second["acquired"]
    assert first["raw_focal_gap_before"] == pytest.approx(.12)
    assert first["raw_focal_gap_after"] == pytest.approx(.14)
    assert first["sensitivity_increase"] == pytest.approx(.02)
    np.testing.assert_allclose(first["focal_gain"], [.20, .20, .22, .22], rtol=0, atol=1e-14)
    assert second["sensitivity_increase"] == pytest.approx(.20)
    assert first["qualified_focal_transfer"] == [False, False, True, True]
    assert first["qualified_unseen_transfer"] == [True] * 4
    assert second["qualified_focal_transfer"] == [False] * 4
    assert second["qualified_unseen_transfer"] == [False] * 4
    assert first["rank_robust"] is True and second["rank_robust"] is False
    verdict = summary["family_verdicts"][0]
    assert verdict["n_seeds"] == 2
    assert verdict["mean_sensitivity_increase"] == pytest.approx(.11)
    assert verdict["sd_sensitivity_increase"] == pytest.approx(np.sqrt((.02 - .11) ** 2 + (.20 - .11) ** 2))
    assert verdict["positive_seed_count"] == 2 and verdict["criterion_met"] is True
    assert verdict["qualified_seed_count"] == 1
    assert verdict["qualified_mean_sensitivity_increase"] == pytest.approx(.02)
    assert verdict["focal_transfer_counts"] == [0, 0, 1, 1]
    assert verdict["unseen_transfer_counts"] == [1, 1, 1, 1]
    assert verdict["rank_robust_count"] == 1
    assert summary["actual_training_trials"] == 2 * 1 * 8


def test_positive_seed_fraction_can_fail_even_when_mean_adjusted_effect_is_large():
    cfg = spacing_fixture_config()
    cfg.update(seeds=[61, 62], conditions=["fixed"])
    cases = [
        dict(iid_before=.5, iid_after=.8, baseline=[.5] * 4, final=[.5, .6, .8, .9]),
        dict(iid_before=.5, iid_after=.8, baseline=[.5, .5, .6, .6], final=[.5, .5, .59, .59]),
    ]
    summary = study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))
    verdict = summary["family_verdicts"][0]
    assert verdict["mean_sensitivity_increase"] == pytest.approx(.145)
    assert verdict["positive_seed_count"] == 1
    assert verdict["criterion_met"] is False


def test_exact_gate_boundaries_and_zero_sensitivity_are_handled_separately():
    cfg = spacing_fixture_config()
    cfg.update(seeds=[61, 62], conditions=["fixed"])
    cfg["criteria"].update(acquisition_gain=.25, transfer_accuracy=.75,
                           sensitivity_increase=.125, positive_seed_fraction=.5)
    cases = [
        dict(iid_before=.5, iid_after=.75, baseline=[.5] * 4, final=[.75] * 4,
             final_all=.75, greedy_after=.9),
        dict(iid_before=.5, iid_after=.75, baseline=[.5] * 4, final=[.5, .6, .75, .85],
             greedy_after=.899999),
    ]
    summary = study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))
    first, second = summary["outcomes"]
    assert first["acquired"] is True
    assert first["qualified_focal_transfer"] == [True] * 4
    assert first["rank_robust"] is True
    assert second["rank_robust"] is False
    assert first["sensitivity_increase"] == 0
    verdict = summary["family_verdicts"][0]
    assert verdict["mean_sensitivity_increase"] == .125
    assert verdict["positive_seed_count"] == 1
    assert verdict["criterion_met"] is True


@pytest.mark.parametrize("changed", ["config", "protocol", "background", "code"])
def test_spacing_freeze_blocks_source_config_protocol_drift_before_result_creation(tmp_path, changed):
    root, config, registration = prepare_spacing_fixture(tmp_path)
    path = {"config": config, "protocol": root / "docs/spacing-protocol.md",
            "background": root / "docs/spacing-background.md",
            "code": root / "src/drosolearn/model.py"}[changed]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError, match="changed after registration"):
        study.run(root, config, registration, output)
    assert not output.exists()


def test_spacing_registration_and_existing_results_are_never_overwritten(tmp_path):
    root, config, registration = prepare_spacing_fixture(tmp_path)
    original = registration.read_bytes()
    with pytest.raises(FileExistsError):
        study.register(root, config, registration)
    assert registration.read_bytes() == original
    output = root / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("old result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.run(root, config, registration, output)
    assert marker.read_text(encoding="utf-8") == "old result"


def test_small_run_records_one_actual_fit_not_four_map_training_trajectories(completed_spacing_fixture):
    _, config, _, output, summary = completed_spacing_fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rows = csv_rows(output / "seed_61/trials.csv")
    assert len(rows) == summary["actual_training_trials"] == 48
    assert len(summary["outcomes"]) == len(study.FAMILIES)
    with np.load(output / "seed_61/stimuli.npz") as stimuli:
        action_uniforms = stimuli["action_uniforms"].copy()
    np.testing.assert_array_equal(action_uniforms, np.random.default_rng(61 + 280000).random(8))
    for condition in study.FAMILIES:
        selected = [r for r in rows if r["condition"] == condition]
        assert [int(r["trial"]) for r in selected] == list(range(1, 9))
        for index, row in enumerate(selected):
            target = int(int(row["right_count"]) > int(row["left_count"]))
            probability = float(row["p_right"])
            assert int(row["action"]) == int(action_uniforms[index] < probability)
            assert int(row["reward"]) == int(int(row["action"]) == target)
            assert np.isfinite(probability) and 0 <= probability <= 1
        outcome = [r for r in summary["outcomes"] if r["condition"] == condition][0]
        assert outcome["head_updates"] == cfg["trials"]
        assert outcome["internal_updates"] == (cfg["trials"] if "plastic" in condition else 0)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rng"]["actions"] == "seed+280000"
    np.testing.assert_array_equal(np.load(output / "spacing_codes.npy"), independently_constructed_panel())


def test_small_run_preserves_update_scopes_bounds_and_four_frozen_transfer_artifacts(completed_spacing_fixture):
    root, config, _, output, _ = completed_spacing_fixture
    curves = csv_rows(output / "learning_curves.csv")
    assert len(curves) == 6 * 3 * 2
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {"registration.json", "metadata.json", "summary.json", "spacing_codes.npy"} <= set(hashes)
    registered = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert registered["source_hashes"] == study.study_hashes(root, json.loads(config.read_text(encoding="utf-8")))
    for condition in study.FAMILIES:
        path = output / "seed_61" / condition
        with np.load(path / "checkpoint_0000.npz") as initial, np.load(path / "checkpoint_0008.npz") as final:
            for key in ("encoder", "bias", "pre", "post", "kc_activity"):
                np.testing.assert_array_equal(initial[key], final[key])
            assert not np.array_equal(initial["readout_weights"], final["readout_weights"])
            if "plastic" in condition:
                assert not np.array_equal(initial["internal_weights"], final["internal_weights"])
            else:
                np.testing.assert_array_equal(initial["internal_weights"], final["internal_weights"])
            assert np.all(final["internal_weights"] >= 0)
            assert np.all(final["internal_weights"] <= 2 * initial["internal_weights"])
        for stage in ("baseline", "final"):
            with np.load(path / f"transfer_{stage}.npz") as transfer:
                assert transfer["score_cache"].shape == (27, 8, 2, 14)
                assert transfer["map_p_target"].shape == transfer["map_greedy_accuracy"].shape == (4, 27)
                assert transfer["map_p_right"].shape == (4, 27, 8)
                assert np.all((transfer["map_p_target"] >= 0) & (transfer["map_p_target"] <= 1))
                np.testing.assert_array_equal(transfer["code_values"], np.unique(independently_constructed_panel()))


def test_external_spacing_config_paths_are_saved_as_reproducible_absolute_paths(tmp_path):
    root, original, _ = prepare_spacing_fixture(tmp_path)
    external = tmp_path / "outside config.json"
    external.write_bytes(original.read_bytes())
    registration = tmp_path / "outside registration.json"
    study.register(root, external, registration)
    output = tmp_path / "external-run"
    study.run(root, external, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    args = metadata["reproduce_argv"]
    assert args[args.index("--config") + 1] == external.resolve().as_posix()
