"""Independent factor arithmetic on invented saved tensors only."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gated_factors_independent_tests", ROOT / "scripts/verify_gated_factors.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_all_eight_corners_use_the_declared_bits_and_exact_sigmoid():
    zc, zg = np.array([.1, -.2, .3, -.4]), np.array([-.3, .2, -.1, .4])
    gc, gg, fc, fg = -.03, -.02, np.array([.2, .3, -.1, .5]), np.array([.4, -.1, .2, .3])
    actual = audit.independent_bridge(zc, zg, np.asarray(gc), np.asarray(gg), fc, fg)
    for code in range(8):
        z, g, f = (zg if code & 4 else zc), (gg if code & 2 else gc), (fg if code & 1 else fc)
        assert np.allclose(actual[code], audit.SIGNS * (audit.sigmoid(z + g * f) - audit.sigmoid(z)))


def test_shapley_is_symmetric_for_a_three_way_interaction_and_preserves_efficiency():
    value = np.zeros((8, 4), int)
    value[7] = np.array([3, -3, 6, -6])
    result = audit.independent_shapley(value)
    assert np.allclose(result, np.array([[1, -1, 2, -2]] * 3))
    assert np.allclose(result.sum(0), value[7] - value[0])


@pytest.mark.parametrize("bit,index", [(4, 0), (2, 1), (1, 2)])
def test_a_single_factor_effect_is_assigned_only_to_that_factor(bit, index):
    value = np.array([[1., -2., 3., -4.] if code & bit else [0., 0., 0., 0.] for code in range(8)])
    result = audit.independent_shapley(value)
    assert np.allclose(result[index], value[bit])
    assert np.allclose(np.delete(result, index, axis=0), 0.)


def test_state_saturation_changes_finite_probability_effect_without_changing_margin_step():
    cells = audit.independent_bridge(np.zeros(4), np.full(4, 8.), np.asarray(.1), np.asarray(.1), np.ones(4), np.ones(4))
    assert abs(cells[4, 0]) < abs(cells[0, 0])
    contribution = audit.independent_shapley(cells)
    assert contribution[0, 0] < 0.
    assert np.allclose(contribution[1:], 0.)


@pytest.mark.parametrize("bad", [np.full(4, np.nan), np.ones(4, complex), np.zeros(3)])
def test_bad_corner_state_is_rejected(bad):
    with pytest.raises(ValueError):
        audit.independent_bridge(bad, np.zeros(4), np.asarray(.1), np.asarray(.1), np.ones(4), np.ones(4))
