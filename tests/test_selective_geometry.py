"""Invented small matrices only; never load graph or scientific task seeds."""
import numpy as np
import pytest

from drosolearn.selective_geometry import selective_direction, selective_geometry


def rows():
    return np.array([[1., 0., 0., 0., 0.],
                     [.6, .8, 0., 0., 0.],
                     [.3, .4, np.sqrt(.75), 0., 0.],
                     [-.4, .1, .2, np.sqrt(.79), 0.]])


def test_label_free_constructor_is_qualified_and_has_unit_endpoints():
    info = selective_geometry(rows())
    assert info["numeric_selective_qualified"]
    assert info["retained_rank"] == 4
    assert info["full_row_rank"]
    assert not info["direction_uses_task_labels"]
    assert info["observed_sign_memory_is_separate_learning_resource"]
    for key in ("shared_directions", "isolated_directions"):
        np.testing.assert_allclose(np.linalg.norm(info[key], axis=1), 1.)
    isolated = info["isolated_directions"] @ info["unit_rows"].T
    np.testing.assert_allclose(isolated - np.diag(np.diag(isolated)), 0., atol=1e-12)


def test_only_known_harmful_destinations_are_suppressed():
    info = selective_geometry(rows())
    result = selective_direction(info, 0, [1, 1, -1, 0])
    np.testing.assert_array_equal(result["harm_mask"], [False, False, True, False])
    np.testing.assert_allclose(result["desired_influence"], [1., .6, 0., -.4])
    np.testing.assert_allclose(result["raw_influence"], result["desired_influence"], atol=1e-12)
    assert result["raw_norm"] >= 1.
    assert result["normalization_scale"] < 1.
    np.testing.assert_allclose(result["influence"],
                               result["desired_influence"] / result["raw_norm"], atol=1e-12)
    assert result["influence"][1] < .6
    assert result["influence"][0] < 1.
    assert not result["original_helpful_magnitudes_retained_after_normalization"]


def test_negative_cross_score_can_help_opposite_targets():
    info = selective_geometry(rows())
    result = selective_direction(info, 0, [1, 1, 1, -1])
    assert not result["harm_mask"].any()
    np.testing.assert_array_equal(result["direction"], info["unit_rows"][0])
    assert result["influence"][3] < 0.


def test_negative_cross_score_can_harm_same_targets():
    info = selective_geometry(rows())
    result = selective_direction(info, 0, [1, 1, 1, 1])
    np.testing.assert_array_equal(result["harm_mask"], [False, False, False, True])
    assert abs(result["influence"][3]) < 1e-12
    assert result["influence"][1] > 0.


@pytest.mark.parametrize("signs", ([0, 0, 0, 0], [0, 1, -1, 1], [1, 0, 0, 0]))
def test_unknown_source_or_unknown_destinations_do_not_infer_targets(signs):
    info = selective_geometry(rows())
    result = selective_direction(info, 0, signs)
    assert not result["harm_mask"].any()
    np.testing.assert_array_equal(result["direction"], info["unit_rows"][0])


def test_no_harm_case_uses_no_new_svd(monkeypatch):
    info = selective_geometry(rows())
    def forbidden(*args, **kwargs):
        raise AssertionError("No direction-time SVD permitted")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    for signs in ([0, 0, 0, 0], [1, 1, -1, -1], [-1, 1, -1, 0]):
        for source in range(4):
            assert selective_direction(info, source, signs)["numeric_direction_qualified"]


def test_constructor_reuses_three_label_free_svd_calls(monkeypatch):
    original = np.linalg.svd
    calls = []
    def tracked(*args, **kwargs):
        calls.append(np.asarray(args[0]).shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", tracked)
    selective_geometry(rows())
    assert calls == [(4, 5), (4, 5), (4, 5)]


def test_direction_correction_is_closest_under_raw_effect_constraints():
    info = selective_geometry(rows())
    result = selective_direction(info, 0, [1, -1, 1, 1])
    null = np.array([0., 0., 0., 0., 1.])
    original = info["unit_rows"][0]
    raw = result["raw_direction"]
    competitor = raw + 2. * null
    np.testing.assert_allclose(info["unit_rows"] @ competitor,
                               result["desired_influence"], atol=1e-12)
    assert np.linalg.norm(raw - original) < np.linalg.norm(competitor - original)


def test_column_rotation_preserves_every_effect_and_cost():
    initial = rows()
    q, _ = np.linalg.qr(np.array([[1., 2., 3., 4., 5.], [0., 1., 0., 2., 0.],
                                 [1., 0., 2., 0., 0.], [0., 1., 3., 0., 1.],
                                 [2., 0., 0., 1., 1.]]))
    a = selective_geometry(initial)
    b = selective_geometry(initial @ q)
    first = selective_direction(a, 2, [1, -1, 1, -1])
    second = selective_direction(b, 2, [1, -1, 1, -1])
    np.testing.assert_allclose(first["influence"], second["influence"], atol=1e-12)
    np.testing.assert_allclose(first["raw_norm"], second["raw_norm"], atol=1e-12)
    np.testing.assert_allclose(first["direction"] @ q, second["direction"], atol=1e-12)


def test_feature_row_scale_does_not_supply_label_information():
    a = selective_geometry(rows())
    b = selective_geometry(rows() * np.array([2., 5., .1, 3.])[:, None])
    np.testing.assert_allclose(a["unit_rows"], b["unit_rows"], atol=1e-12)
    for source in range(4):
        first = selective_direction(a, source, [1, 1, -1, -1])
        second = selective_direction(b, source, [1, 1, -1, -1])
        np.testing.assert_allclose(first["direction"], second["direction"], atol=1e-12)


def test_arguments_are_copied_and_never_modified():
    value = rows()
    before = value.copy()
    info = selective_geometry(value)
    signs = np.array([1, 1, -1, 0])
    prior_signs = signs.copy()
    prior_unit = info["unit_rows"].copy()
    result = selective_direction(info, np.int64(0), signs)
    result["observed_signs"][0] = -1
    np.testing.assert_array_equal(value, before)
    np.testing.assert_array_equal(signs, prior_signs)
    np.testing.assert_array_equal(info["unit_rows"], prior_unit)


def test_signed_harm_threshold_keeps_tiny_effect_without_zeroing():
    epsilon = 5e-9
    value = np.eye(4)
    value[1, 0] = -epsilon
    info = selective_geometry(value)
    result = selective_direction(info, 0, [1, 1, 0, 0])
    assert not result["harm_mask"].any()
    assert result["influence"][1] < 0.


def test_signed_harm_beyond_threshold_is_removed():
    value = np.eye(4)
    value[1, 0] = -2e-8
    info = selective_geometry(value)
    result = selective_direction(info, 0, [1, 1, 0, 0])
    assert result["harm_mask"][1]
    assert abs(result["influence"][1]) < 1e-12


@pytest.mark.parametrize("source", (-1, 4, 0.0, True, False, "0", None))
def test_invalid_source_rejected(source):
    with pytest.raises(ValueError, match="Source"):
        selective_direction(selective_geometry(rows()), source, [1, 1, -1, -1])


@pytest.mark.parametrize("signs", ([1, 1, -1], [[1, 1, -1, -1]], [1, 2, -1, 0],
                                   [1, .5, -1, 0], [1, np.nan, -1, 0],
                                   [1, np.inf, -1, 0], ["1", "1", "-1", "0"],
                                   [True, True, False, False]))
def test_invalid_observation_record_rejected(signs):
    with pytest.raises(ValueError, match="Observed signs"):
        selective_direction(selective_geometry(rows()), 0, signs)


@pytest.mark.parametrize("value", (np.ones((4, 5)), np.eye(3), np.zeros((4, 5)),
                                   np.full((4, 5), np.nan), np.ones(4)))
def test_invalid_or_rank_deficient_geometry_fails_closed(value):
    with pytest.raises(ValueError):
        selective_geometry(value)


def test_full_rank_but_poor_condition_fails_closed():
    value = np.eye(4)
    value[3] = value[0] + 1e-7 * value[3]
    with pytest.raises(ValueError, match="qualified full-rank"):
        selective_geometry(value)


def test_unqualified_geometry_rejected_at_direction_call():
    info = selective_geometry(rows())
    info["numeric_selective_qualified"] = False
    with pytest.raises(ValueError, match="Unqualified"):
        selective_direction(info, 0, [1, 1, -1, -1])


def test_modified_inverse_fails_numeric_checks():
    info = selective_geometry(rows())
    info["pseudoinverse"] = np.zeros_like(info["pseudoinverse"])
    with pytest.raises(ValueError, match="numeric checks"):
        selective_direction(info, 0, [1, -1, 1, 1])


def test_all_synthetic_observation_states_have_unit_directions():
    import itertools
    info = selective_geometry(rows())
    for state in itertools.product((-1, 0, 1), repeat=4):
        for source in range(4):
            result = selective_direction(info, source, state)
            assert result["numeric_direction_qualified"]
            assert abs(np.linalg.norm(result["direction"]) - 1.) < 1e-12
            np.testing.assert_allclose(result["raw_influence"],
                                       result["desired_influence"], atol=1e-12)
