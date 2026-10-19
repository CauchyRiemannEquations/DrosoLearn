"""Repair tests use synthetic trajectories; scientific keys are inspected only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn.human_shell import produce, summarize
from drosolearn.human_shell_models import ADAPTIVE
from drosolearn.human_shell_rng import action_entropy, produce_corrected, validate_action_addresses
from test_human_shell import synthetic_config, synthetic_rows
from test_human_shell_models import small_graph


def test_registered_tuple_addresses_are_unique_without_drawing_random_values():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/human_shell_v1.json").read_text(encoding="utf8"))
    old = [seed + 300000 + int(pid[1:]) for seed in cfg["test_seeds"] for pid in cfg["test_ids"]]
    new = validate_action_addresses(cfg)
    assert len(old) == len(new) == 145
    assert len(set(old)) == 88 and len(set(new)) == 145
    assert all(key[0] == 20261019 for key in new)


def test_compositional_address_separates_diagonal_collision():
    assert 41 + 300000 + 9002 == 42 + 300000 + 9001
    assert action_entropy(41, "A9002") != action_entropy(42, "A9001")


@pytest.mark.parametrize("seed,pid", [(-1, "A9001"), (1.5, "A9001"), (1, "B9001"), (1, "A"),
                                    (1, "A123"), (1, "A12345"), (1, "A１２３４")])
def test_invalid_address_rejected(seed, pid):
    with pytest.raises(ValueError):
        action_entropy(seed, pid)


def test_ambiguous_numeric_person_identity_rejected():
    with pytest.raises(ValueError):
        validate_action_addresses({"test_seeds": [1], "test_ids": ["A0001", "A1"]})


@pytest.mark.parametrize("ids,seeds", [([], [1]), (["A9001"], []), (["A9001", "A9001"], [1]), (["A9001"], [1, 1])])
def test_empty_or_duplicate_address_axes_rejected(ids, seeds):
    with pytest.raises(ValueError):
        validate_action_addresses({"test_ids": ids, "test_seeds": seeds})


def test_repair_executes_only_autonomous_and_preserves_parent_inputs(monkeypatch):
    import drosolearn.human_shell_rng as repair
    cfg = synthetic_config()
    cfg["test_seeds"] = [41, 42]
    cfg["test_ids"] = ["A9001", "A9002"]
    records = {pid: synthetic_rows() for pid in cfg["test_ids"]}
    selection = {"rates": {c: .1 for c in ADAPTIVE}, "calibration_bias": .4}
    old_arrays, old_rows, old_witness = produce(records, cfg, small_graph(), selection)
    original_summary = summarize(records, cfg, old_arrays, old_rows)
    before_arrays = {k: v.tobytes() for k, v in old_arrays.items()}
    before_witness = {k: v.tobytes() for k, v in old_witness.items()}
    before_rows, before_selection = copy.deepcopy(old_rows), copy.deepcopy(selection)
    native = repair._trajectory
    executions = []

    def autonomous_only(model, rows, autonomous=False, uniforms=None):
        assert autonomous is True
        executions.append(len(rows))
        return native(model, rows, autonomous, uniforms)

    monkeypatch.setattr(repair, "_trajectory", autonomous_only)
    arrays, rows, witnesses = produce_corrected(old_arrays, old_rows, old_witness, records, cfg, small_graph(), selection)
    assert len(executions) == 11 * 2 * 2 and sum(executions) == 11 * 2 * 2 * 216
    assert selection == before_selection and old_rows == before_rows
    assert all(old_arrays[k].tobytes() == before_arrays[k] for k in old_arrays)
    assert all(old_witness[k].tobytes() == before_witness[k] for k in old_witness)
    assert all(arrays[k][0].tobytes() == old_arrays[k][0].tobytes() for k in arrays)
    assert [r for r in rows if r["mode"] == "conditional"] == [r for r in old_rows if r["mode"] == "conditional"]
    for key in old_witness:
        if key.startswith("conditional_"):
            assert witnesses[key].tobytes() == old_witness[key].tobytes()
    assert np.any(arrays["actions"][1] != old_arrays["actions"][1])
    summary = summarize(records, cfg, arrays, rows)
    for key in ("H_prediction", "conditional_family_mean_nll", "conditional_family_mean_brier"):
        assert summary[key] == original_summary[key]
    repeated = produce_corrected(old_arrays, old_rows, old_witness, records, cfg, small_graph(), selection)
    assert all(repeated[0][k].tobytes() == arrays[k].tobytes() for k in arrays)
    assert repeated[1] == rows
    assert all(repeated[2][k].tobytes() == witnesses[k].tobytes() for k in witnesses)
