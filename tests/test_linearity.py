"""Score-distillation arithmetic, leakage guards and frozen policy invariants.

Only small engineering seeds 71 and 72 are fit here. The linear references
receive privileged teacher scores; they are not independent reward learners.
"""
from itertools import combinations
from pathlib import Path
import copy
import csv
import json
import shutil

import numpy as np
import pytest

from drosolearn import linearity as study
from drosolearn.data import Dataset, analytical_accuracy, assert_disjoint, make_dataset
from drosolearn.graph import Projection
from drosolearn.model import Learner


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_ALPHAS = np.array([0.25, 0.5, 1.0, 1.5])
INDEPENDENT_PAIRS = tuple(p for p in combinations(range(1, 9), 2) if p != (2, 4))
INDEPENDENT_UNSEEN = np.array([i for i, (a, b) in enumerate(INDEPENDENT_PAIRS)
                              if a not in (2, 4) and b not in (2, 4)])
INDEPENDENT_ANCHORED = np.array([i for i in range(27) if i not in INDEPENDENT_UNSEEN])


def independently_constructed_panel():
    return np.array([[1, 2, 3, 4, 6 - alpha, 6, 6 + alpha, 8]
                     for alpha in INDEPENDENT_ALPHAS], dtype=float)


def linearity_fixture_config():
    cfg = json.loads((ROOT / "configs/linearity_diagnostic.json").read_text(encoding="utf-8"))
    cfg.update(scope="integration_test", seeds=[71], conditions=list(study.FAMILIES),
               trials=8, checkpoint_every=4, eval_trials=8, calibration_trials=32, probe_rows=4)
    return cfg


def prepare_linearity_fixture(path):
    root = Path(path) / "study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    graph = linearity_fixture_config()["graph"]
    shutil.copytree(ROOT / graph, root / graph)
    for name in ("data/source-manifest.json", "data/LICENSE.md", "pyproject.toml",
                 "requirements-lock.txt", "docs/linearity-protocol.md", "docs/linearity-background.md"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    config = root / "config.json"
    config.write_text(json.dumps(linearity_fixture_config()) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    study.register(root, config, registration)
    return root, config, registration


@pytest.fixture(scope="module")
def completed_linearity_fixture(tmp_path_factory):
    root, config, registration = prepare_linearity_fixture(tmp_path_factory.mktemp("linearity"))
    output = root / "fixture-output"
    summary = study.run(root, config, registration, output)
    return root, config, registration, output, summary


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def toy_linearity_model(plastic=True):
    graph = Projection(
        pre=np.array([0, 1, 1, 2]), post=np.array([0, 0, 1, 1]),
        weights=np.array([2.0, 1.0, 3.0, 2.0]), n_kc=3, n_mbon=2,
        kc_ids=["a", "b", "c"], mbon_ids=["d", "e"], metadata={"scope": "test fixture"},
    )
    return Learner(graph, seed=19, plastic=plastic)


def test_four_spacing_maps_preserve_training_anchors_and_only_change_frozen_evaluation():
    panel = study.spacing_panel()
    np.testing.assert_array_equal(panel, independently_constructed_panel())
    assert tuple(study.PAIRS) == INDEPENDENT_PAIRS
    np.testing.assert_array_equal(study.PRIMARY, INDEPENDENT_UNSEEN)
    np.testing.assert_array_equal(study.ANCHORED, INDEPENDENT_ANCHORED)
    for name, pair in (("train", (2, 4)), ("calibration", (2, 4)), ("focal", (5, 7))):
        ds = make_dataset(71, name, 32, pair=pair)
        original = ds.inputs()
        for alpha, codes in zip(INDEPENDENT_ALPHAS, panel):
            mapped = study.encoded_inputs(ds, codes)
            np.testing.assert_array_equal(mapped[..., 1:], original[..., 1:])
            if name != "focal":
                assert mapped.tobytes() == original.tobytes()
            else:
                np.testing.assert_allclose(mapped[..., 0].mean(axis=1), .75, rtol=0, atol=0)
                np.testing.assert_allclose(np.abs(np.diff(mapped[..., 0], axis=1))[:, 0],
                                           2 * alpha / 8, rtol=0, atol=0)
                np.testing.assert_array_equal(np.argmax(mapped[..., 0], axis=1), ds.target())


def test_calibration_and_all_evaluation_splits_are_independent_reproducible_and_crossed():
    cfg = linearity_fixture_config()
    first, repeated = study.make_splits(cfg, 71), study.make_splits(cfg, 71)
    assert_disjoint(list(first.values()))
    assert set(first) == {"train", "iid", "ood", "calibration",
                          *(f"pair_{a}_{b}" for a, b in INDEPENDENT_PAIRS)}
    expected_calibration = make_dataset(71 * 100 + 34, "calibration", 32)
    np.testing.assert_array_equal(first["calibration"].counts, expected_calibration.counts)
    np.testing.assert_array_equal(first["calibration"].nuisance, expected_calibration.nuisance)
    assert first["calibration"].family_ids == expected_calibration.family_ids
    assert len(first["calibration"].counts) == 32
    for name, dataset in first.items():
        np.testing.assert_array_equal(dataset.counts, repeated[name].counts)
        np.testing.assert_array_equal(dataset.nuisance, repeated[name].nuisance)
        assert dataset.family_ids == repeated[name].family_ids
        assert np.mean(dataset.target()) == .5
        assert analytical_accuracy(dataset) == 1.
        assert all(analytical_accuracy(dataset, nuisance_index=i) == .5 for i in range(6))
    assert np.all((first["ood"].nuisance <= .25) | (first["ood"].nuisance >= .75))
    design = np.column_stack((first["calibration"].inputs().reshape(-1, 7),
                              np.ones(2 * cfg["calibration_trials"])))
    assert np.linalg.matrix_rank(design) == 8


@pytest.mark.parametrize("changed", ["config", "protocol", "background", "code"])
def test_registration_blocks_drift_before_creating_results(tmp_path, changed):
    root, config, registration = prepare_linearity_fixture(tmp_path)
    target = {"config": config, "protocol": root / "docs/linearity-protocol.md",
              "background": root / "docs/linearity-background.md",
              "code": root / "src/drosolearn/model.py"}[changed]
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = root / "must-not-exist"
    with pytest.raises(ValueError, match="changed after registration"):
        study.run(root, config, registration, output)
    assert not output.exists()


def test_registration_and_previous_results_are_never_overwritten(tmp_path):
    root, config, registration = prepare_linearity_fixture(tmp_path)
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


def affine_fixture(coefficients=None):
    """Known score function with a distinct contribution from every input."""
    calibration = make_dataset(72 * 100 + 34, "calibration", 32)
    inputs = calibration.inputs()
    coeff = np.array([.37, 2.6, -.8, .4, 1.7, -2.1, .63, -.27]) if coefficients is None else np.asarray(coefficients)
    design = np.concatenate((np.ones((*inputs.shape[:-1], 1)), inputs), axis=-1)
    return inputs, design @ coeff, coeff


def test_affine_ols_recovers_known_intercept_and_all_seven_input_coefficients():
    inputs, scores, coefficients = affine_fixture()
    references = study.fit_references(inputs, scores)
    np.testing.assert_allclose(references["affine_coefficients"], coefficients, rtol=0, atol=2e-13)
    assert references["affine_rank"] == 8
    assert references["affine_singular_values"].shape == (8,)
    assert np.all(references["affine_singular_values"] > 0)
    assert references["affine_calibration_score_rmse"] < 1e-14
    # Both anchors see exactly the same nuisance vectors. The scalar reference
    # retains the nuisance mean in its intercept and only quantity in its slope.
    means = np.array([scores[inputs[..., 0] == q].mean() for q in (.25, .5)])
    slope = (means[1] - means[0]) / .25
    intercept = means[0] - .25 * slope
    np.testing.assert_allclose(references["scalar_anchor_means"], means, rtol=0, atol=1e-14)
    np.testing.assert_allclose(references["scalar_coefficients"], [intercept, slope], rtol=0, atol=1e-14)
    assert slope == pytest.approx(coefficients[1], abs=1e-14)
    expected_scalar = intercept + slope * inputs[..., 0]
    assert references["scalar_calibration_score_rmse"] == pytest.approx(np.sqrt(np.mean((expected_scalar - scores) ** 2)))
    assert references["scalar_calibration_score_rmse"] > 0
    # Recovering coefficients does not depend on the order of calibration rows.
    order = np.arange(len(inputs))[::-1]
    reordered = study.fit_references(inputs[order], scores[order])
    for key in ("affine_coefficients", "scalar_coefficients", "scalar_anchor_means"):
        np.testing.assert_allclose(references[key], reordered[key], rtol=0, atol=2e-13)


def test_affine_reference_matches_independent_direct_predictions_on_new_pairs_and_spacing():
    inputs, teacher_scores, coefficients = affine_fixture()
    references = study.fit_references(inputs, teacher_scores)
    # An always-active ReLU bank makes an exact affine teacher within the study
    # input range. The projection/head below reconstruct the stated score.
    graph = Projection(pre=np.arange(8), post=np.arange(8), weights=np.ones(8),
                       n_kc=8, n_mbon=8, kc_ids=list("abcdefgh"), mbon_ids=list("ABCDEFGH"),
                       metadata={"scope": "exact affine mathematical fixture"})
    model = Learner(graph, seed=17, plastic=False)
    model.encoder = np.vstack((np.zeros(7), np.eye(7)))
    model.bias = np.full(8, 2.)
    model.head = coefficients.copy()
    # The constant KC handles the aggregate activation offset exactly.
    model.head[0] = (coefficients[0] - 2 * coefficients[1:].sum()) / 2
    datasets = study.make_splits(linearity_fixture_config(), 72)
    result = study.transfer_evaluate(model, datasets, independently_constructed_panel(), references)
    for map_id, mapping in enumerate(independently_constructed_panel()):
        for pair_id, pair in enumerate(INDEPENDENT_PAIRS):
            dataset = datasets[f"pair_{pair[0]}_{pair[1]}"]
            heldout_inputs = study.encoded_inputs(dataset, mapping)
            design = np.concatenate((np.ones((*heldout_inputs.shape[:-1], 1)), heldout_inputs), axis=-1)
            direct_scores = design @ coefficients
            margin = direct_scores[:, 1] - direct_scores[:, 0]
            probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
            for key in ("teacher_map_scores", "affine_map_scores"):
                np.testing.assert_allclose(result[key][map_id, pair_id], direct_scores, rtol=0, atol=3e-13)
            for key in ("teacher_map_margin", "affine_map_margin"):
                np.testing.assert_allclose(result[key][map_id, pair_id], margin, rtol=0, atol=3e-13)
            for key in ("map_p_right", "affine_map_p_right"):
                np.testing.assert_allclose(result[key][map_id, pair_id], probability, rtol=0, atol=5e-14)
            scalar_coeff = references["scalar_coefficients"]
            scalar_score = scalar_coeff[0] + scalar_coeff[1] * heldout_inputs[..., 0]
            np.testing.assert_allclose(result["scalar_map_scores"][map_id, pair_id], scalar_score, rtol=0, atol=1e-14)
    assert np.max(result["affine_probability_mae"]) < 1e-13


@pytest.mark.parametrize("changed", ["inputs_shape", "scores_shape", "input_nan", "score_inf", "anchor", "rank"])
def test_fit_references_rejects_invalid_or_heldout_anchor_geometry(changed):
    inputs, scores, _ = affine_fixture()
    if changed == "inputs_shape":
        inputs = inputs[..., :6]
    elif changed == "scores_shape":
        scores = scores[:, :1]
    elif changed == "input_nan":
        inputs[0, 0, 1] = np.nan
    elif changed == "score_inf":
        scores[0, 0] = np.inf
    elif changed == "anchor":
        inputs[0, 0, 0] = .625
    else:
        inputs[..., 1:] = 0
    with pytest.raises(ValueError):
        study.fit_references(inputs, scores)


def test_calibration_uses_only_its_supplied_dataset_and_changes_no_model_state():
    datasets = study.make_splits(linearity_fixture_config(), 71)
    model = toy_linearity_model()
    state = model.snapshot(datasets["iid"].inputs())
    counters = model.head_updates, model.internal_updates
    expected_scores = model.features(datasets["calibration"].inputs())[1] @ model.head
    calibrated = study.calibrate(model, datasets["calibration"])
    np.testing.assert_array_equal(calibrated["calibration_inputs"], datasets["calibration"].inputs())
    np.testing.assert_allclose(calibrated["calibration_scores"], expected_scores, rtol=0, atol=1e-14)
    expected = study.fit_references(datasets["calibration"].inputs(), expected_scores)
    for key in expected:
        np.testing.assert_allclose(calibrated[key], expected[key], rtol=0, atol=0)
    after = model.snapshot(datasets["iid"].inputs())
    for key in state:
        np.testing.assert_array_equal(state[key], after[key])
    assert counters == (model.head_updates, model.internal_updates)
    altered = copy.deepcopy(datasets)
    for name, dataset in altered.items():
        if name != "calibration":
            dataset.nuisance[:] = np.nan
            dataset.counts[:] = 8
    repeated = study.calibrate(model, altered["calibration"])
    for key in calibrated:
        np.testing.assert_allclose(repeated[key], calibrated[key], rtol=0, atol=0)


def test_frozen_transfer_never_refits_references_or_consumes_calibration_and_training(monkeypatch):
    datasets = study.make_splits(linearity_fixture_config(), 71)
    model = toy_linearity_model()
    references = study.calibrate(model, datasets["calibration"])
    original_references = copy.deepcopy(references)
    expected = study.transfer_evaluate(model, datasets, independently_constructed_panel(), references)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("held-out evaluation must not fit or calibrate a reference")
    monkeypatch.setattr(study, "fit_references", forbidden)
    monkeypatch.setattr(study, "calibrate", forbidden)
    poisoned = copy.deepcopy(datasets)
    for name in ("train", "iid", "ood", "calibration"):
        poisoned[name].counts[:] = 8
        poisoned[name].nuisance[:] = np.nan
    actual = study.transfer_evaluate(model, poisoned, independently_constructed_panel(), references)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in references:
        np.testing.assert_array_equal(references[key], original_references[key])


def synthetic_transfer(affine_error=.01, scalar_error=.04, teacher_margin=None,
                       affine_margin=None, scalar_margin=None):
    """Complete policy artifact assembled without a learner or scorer."""
    shape = (4, 27, 8)
    teacher_p = np.full(shape, .7)
    if teacher_margin is None:
        teacher_m = np.log(teacher_p / (1 - teacher_p))
    else:
        teacher_m = np.full(shape, teacher_margin)
        teacher_p = 1 / (1 + np.exp(-np.clip(teacher_m, -40, 40)))
    result = {
        "code_values": np.unique(independently_constructed_panel()),
        "score_cache": np.zeros((27, 8, 2, 14)),
        "map_p_target": teacher_p.mean(axis=2),
        "map_greedy_accuracy": np.ones((4, 27)),
        "map_p_right": teacher_p,
        "teacher_map_margin": teacher_m,
        "teacher_map_scores": np.stack((np.zeros(shape), teacher_m), axis=-1),
    }
    for name, error, raw_margin in (("affine", affine_error, affine_margin),
                                     ("scalar", scalar_error, scalar_margin)):
        if raw_margin is None:
            p = teacher_p - error
            margin = np.log(p / (1 - p))
        else:
            margin = np.full(shape, raw_margin)
            p = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
        result[f"{name}_map_scores"] = np.stack((np.zeros(shape), margin), axis=-1)
        result[f"{name}_map_margin"] = margin
        result[f"{name}_map_p_right"] = p
        result[f"{name}_map_p_target"] = p.mean(axis=2)
        result[f"{name}_map_greedy_accuracy"] = np.ones((4, 27))
        result[f"{name}_probability_mae"] = np.abs(p - teacher_p).mean(axis=2)
    return result


def test_probability_fidelity_and_raw_unclipped_margin_fidelity_are_separate():
    transfer = synthetic_transfer(teacher_margin=80, affine_margin=120, scalar_margin=160)
    measured = study.metrics(transfer, "affine")
    assert measured["unseen_probability_mae"] == 0
    assert measured["unseen_margin_mae"] == 40
    assert measured["unseen_margin_rmse"] == 40
    assert measured["unseen_score_rmse"] == pytest.approx(np.sqrt(40 ** 2 / 2))
    assert measured["unseen_greedy_disagreement"] == 0
    scalar = study.metrics(transfer, "scalar")
    assert scalar["unseen_probability_mae"] == 0
    assert scalar["unseen_margin_mae"] == 80
    # Probability saturation can hide a large raw-score mismatch. It cannot be
    # reported as an exact linear neural score function.


def test_metrics_reduce_each_case_and_keep_unseen_anchored_and_focal_groups_separate():
    transfer = synthetic_transfer(.0, .0)
    # Different pair groups and maps give reductions an independently known mean.
    p = np.full((4, 27, 8), .7)
    p[:, INDEPENDENT_UNSEEN, :] = .67
    p[:, INDEPENDENT_ANCHORED, :] = .62
    focal = INDEPENDENT_PAIRS.index((5, 7))
    p[:, focal, :] = np.array([.61, .63, .65, .69])[:, None]
    transfer["affine_map_p_right"] = p
    transfer["affine_map_p_target"] = p.mean(axis=2)
    transfer["affine_probability_mae"] = np.abs(p - transfer["map_p_right"]).mean(axis=2)
    measured = study.metrics(transfer, "affine")
    assert measured["unseen_probability_mae"] == pytest.approx(np.abs(p[:, INDEPENDENT_UNSEEN] - .7).mean())
    assert measured["anchored_probability_mae"] == pytest.approx(.08)
    assert measured["all_probability_mae"] == pytest.approx(np.abs(p - .7).mean())
    assert measured["focal_probability_mae"] == pytest.approx(.055)
    np.testing.assert_allclose(measured["focal_reference_p_correct"], [.61, .63, .65, .69])
    # Tie-versus-decision difference is .5; opposite hard decisions differ by1.
    teacher_p = transfer["map_p_right"]
    ref_p = transfer["affine_map_p_right"]
    teacher_p[:, INDEPENDENT_UNSEEN, :4] = .5
    ref_p[:, INDEPENDENT_UNSEEN, :4] = .8
    teacher_p[:, INDEPENDENT_UNSEEN, 4:] = .8
    ref_p[:, INDEPENDENT_UNSEEN, 4:] = .2
    measured = study.metrics(transfer, "affine")
    assert measured["unseen_greedy_disagreement"] == pytest.approx(.75)


def synthetic_summary_inputs(cfg, cases):
    curves, transfer = [], {}
    for seed, case in zip(cfg["seeds"], cases):
        for split in ("iid", "ood"):
            for trial, probability in ((0, case.get("before", .5)),
                                       (cfg["trials"], case["after"])):
                curves.append(dict(seed=seed, condition="fixed", trial=trial,
                                   split=split, p_target=probability))
        transfer[seed, "fixed", "baseline"] = synthetic_transfer(.04, .05)
        transfer[seed, "fixed", "final"] = synthetic_transfer(case["error"], .04)
    return curves, transfer


def test_primary_fidelity_includes_failed_acquisition_and_separates_qualified_summary():
    cfg = linearity_fixture_config()
    cfg.update(seeds=[71, 72], conditions=["fixed"])
    cases = [dict(after=.8, error=.01), dict(after=.6, error=.01)]
    summary = study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))
    first, second = summary["outcomes"]
    assert first["acquired"] is True and second["acquired"] is False
    assert first["fidelity_passed"] is True and second["fidelity_passed"] is True
    assert first["qualified_fidelity"] is True and second["qualified_fidelity"] is False
    verdict = summary["family_verdicts"][0]
    assert verdict["n_seeds"] == 2
    assert verdict["passing_seed_count"] == 2
    assert verdict["criterion_met"] is True
    assert verdict["acquisition_count"] == verdict["qualified_count"] == 1
    assert verdict["mean_affine_probability_mae"] == pytest.approx(.01)
    assert verdict["qualified_mean_affine_probability_mae"] == pytest.approx(.01)
    assert summary["actual_training_trials"] == 16
    assert summary["frozen_evaluation_maps"] == 4 and summary["reference_models"] == 2


def test_seed_fraction_can_fail_when_family_mean_probability_mae_passes():
    cfg = linearity_fixture_config()
    cfg.update(seeds=[71, 72], conditions=["fixed"])
    cases = [dict(after=.8, error=.0), dict(after=.8, error=.03)]
    summary = study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))
    verdict = summary["family_verdicts"][0]
    assert verdict["mean_affine_probability_mae"] == pytest.approx(.015)
    assert verdict["passing_seed_count"] == 1
    assert verdict["criterion_met"] is False
    assert verdict["qualified_count"] == 1
    assert verdict["sd_affine_probability_mae"] == pytest.approx(np.std([0., .03], ddof=1))


def test_unqualified_reference_fidelity_is_reported_with_null_qualified_mean():
    cfg = linearity_fixture_config()
    cfg.update(seeds=[71, 72], conditions=["fixed"])
    cases = [dict(after=.6, error=.01), dict(after=.6, error=.01)]
    verdict = study.summarize(cfg, *synthetic_summary_inputs(cfg, cases))["family_verdicts"][0]
    assert verdict["criterion_met"] is True
    assert verdict["acquisition_count"] == verdict["qualified_count"] == 0
    assert verdict["qualified_mean_affine_probability_mae"] is None


@pytest.mark.parametrize("field,value", [
    ("scope", "smoke"), ("seeds", []), ("seeds", [71, 71]), ("seeds", [True]),
    ("train_pair", [1, 3]),
    ("primary_pairs", "all27"), ("references", ["scalar", "affine"]),
    ("references", ["affine"]), ("code_alphas", [.5, .25, 1., 1.5]),
    ("conditions", ["frozen"]), ("conditions", ["fixed", "fixed"]),
    ("trials", 6), ("trials", 8.), ("checkpoint_every", 3), ("eval_trials", 6),
    ("calibration_trials", 6), ("calibration_trials", 0), ("calibration_trials", 32.),
    ("probe_rows", 9), ("head_lr", float("nan")), ("internal_lr", -1),
])
def test_invalid_linearity_config_rejected(field, value):
    cfg = linearity_fixture_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_scientific_configuration_is_exact_without_executing_scientific_seeds():
    cfg = json.loads((ROOT / "configs/linearity_diagnostic.json").read_text(encoding="utf-8"))
    study.validate_config(cfg)
    assert cfg["seeds"] == list(range(5001, 5011))
    for field, value in (("seeds", list(range(5002, 5012))), ("trials", 200),
                         ("checkpoint_every", 50), ("eval_trials", 64),
                         ("calibration_trials", 64), ("probe_rows", 4),
                         ("head_lr", .2), ("internal_lr", .02),
                         ("conditions", ["fixed"]), ("references", ["scalar"]),
                         ("primary_pairs", "all27")):
        changed = copy.deepcopy(cfg)
        changed[field] = value
        with pytest.raises(ValueError):
            study.validate_config(changed)
    for field, value in (("acquisition_accuracy", .8), ("acquisition_gain", .2),
                         ("probability_mae", .03), ("seed_pass_fraction", .7)):
        changed = copy.deepcopy(cfg)
        changed["criteria"][field] = value
        with pytest.raises(ValueError):
            study.validate_config(changed)
    engineering = linearity_fixture_config()
    engineering["scope"] = "preregistered_diagnostic"
    with pytest.raises(ValueError, match="[Tt]en"):
        study.validate_config(engineering)


@pytest.mark.parametrize("field,value", [("probability_mae", float("nan")),
                                           ("probability_mae", -1),
                                           ("acquisition_accuracy", 1.1),
                                           ("seed_pass_fraction", 0)])
def test_invalid_evaluation_criteria_rejected(field, value):
    cfg = linearity_fixture_config()
    cfg["criteria"][field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_small_run_records_one_reward_fit_and_two_frozen_references_per_stage(completed_linearity_fixture):
    _, config, _, output, summary = completed_linearity_fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    rows = csv_rows(output / "seed_71/trials.csv")
    assert len(rows) == summary["actual_training_trials"] == 48
    assert len(summary["outcomes"]) == len(study.FAMILIES)
    with np.load(output / "seed_71/stimuli.npz") as stimuli:
        uniforms = stimuli["action_uniforms"].copy()
        np.testing.assert_array_equal(stimuli["calibration_counts"],
                                      make_dataset(71 * 100 + 34, "calibration", 32).counts)
        calibration_ids = set(stimuli["calibration_family_ids"].tolist())
        train_ids = set(stimuli["train_family_ids"].tolist())
        assert calibration_ids.isdisjoint(train_ids)
    np.testing.assert_array_equal(uniforms, np.random.default_rng(71 + 280000).random(8))
    for condition in study.FAMILIES:
        selected = [r for r in rows if r["condition"] == condition]
        assert [int(r["trial"]) for r in selected] == list(range(1, 9))
        for index, row in enumerate(selected):
            target = int(int(row["right_count"]) > int(row["left_count"]))
            probability = float(row["p_right"])
            assert int(row["action"]) == int(uniforms[index] < probability)
            assert int(row["reward"]) == int(int(row["action"]) == target)
            assert np.isfinite(probability) and 0 <= probability <= 1
        outcome = [r for r in summary["outcomes"] if r["condition"] == condition][0]
        assert outcome["head_updates"] == cfg["trials"]
        assert outcome["internal_updates"] == (cfg["trials"] if "plastic" in condition else 0)
        path = output / "seed_71" / condition
        for stage in ("baseline", "final"):
            with np.load(path / f"reference_{stage}.npz") as reference:
                assert set(reference.files) == {
                    "calibration_inputs", "calibration_scores", "affine_coefficients", "affine_rank",
                    "affine_singular_values", "affine_calibration_score_rmse", "scalar_coefficients",
                    "scalar_anchor_means", "scalar_calibration_score_rmse",
                }
                assert reference["calibration_inputs"].shape == (32, 2, 7)
                assert reference["calibration_scores"].shape == (32, 2)
                assert reference["affine_rank"] == 8
                expected = study.fit_references(reference["calibration_inputs"], reference["calibration_scores"])
                for key, value in expected.items():
                    np.testing.assert_allclose(reference[key], value, rtol=0, atol=1e-14)
            with np.load(path / f"transfer_{stage}.npz") as transfer:
                assert len(transfer.files) == 19
                assert transfer["score_cache"].shape == (27, 8, 2, 14)
                for ref in ("affine", "scalar"):
                    assert transfer[f"{ref}_map_scores"].shape == (4, 27, 8, 2)
                    assert transfer[f"{ref}_map_margin"].shape == transfer[f"{ref}_map_p_right"].shape == (4, 27, 8)
                    assert transfer[f"{ref}_probability_mae"].shape == (4, 27)
                    np.testing.assert_allclose(transfer[f"{ref}_probability_mae"],
                                               np.abs(transfer[f"{ref}_map_p_right"] - transfer["map_p_right"]).mean(axis=2),
                                               rtol=0, atol=0)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rng"]["actions"] == "seed+280000"
    assert "+34" in metadata["rng"]["data"]


def test_small_run_keeps_learning_scopes_and_primary_hashes_distinct_from_reference_fits(completed_linearity_fixture):
    root, config, _, output, _ = completed_linearity_fixture
    curves = csv_rows(output / "learning_curves.csv")
    assert len(curves) == 6 * 3 * 2
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {"registration.json", "metadata.json", "summary.json", "spacing_codes.npy"} <= set(hashes)
    registered = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    assert registered["source_hashes"] == study.study_hashes(root, json.loads(config.read_text(encoding="utf-8")))
    for condition in study.FAMILIES:
        path = output / "seed_71" / condition
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
            for checkpoint, stage in ((initial, "baseline"), (final, "final")):
                with np.load(path / f"reference_{stage}.npz") as reference:
                    x = reference["calibration_inputs"]
                    kc = np.maximum(x @ checkpoint["encoder"].T + checkpoint["bias"], 0)
                    mb = np.zeros((len(x), 2, len(checkpoint["readout_weights"])))
                    # Scatter every anatomical contact directly; no collapsed
                    # candidate cache or study calibration helper is used here.
                    for edge, (pre, post) in enumerate(zip(checkpoint["pre"], checkpoint["post"])):
                        mb[..., post] += kc[..., pre] * checkpoint["internal_weights"][edge]
                    expected_scores = mb @ checkpoint["readout_weights"]
                    np.testing.assert_allclose(reference["calibration_scores"], expected_scores, rtol=0, atol=1e-13)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"


def test_external_configuration_path_is_recorded_as_reproducible_absolute_path(tmp_path):
    root, original, _ = prepare_linearity_fixture(tmp_path)
    external = tmp_path / "outside config.json"
    external.write_bytes(original.read_bytes())
    registration = tmp_path / "outside registration.json"
    study.register(root, external, registration)
    output = tmp_path / "external-run"
    study.run(root, external, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    args = metadata["reproduce_argv"]
    assert args[args.index("--config") + 1] == external.resolve().as_posix()
