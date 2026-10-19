"""Saved-path accounting checks on invented values; no study data or training."""
from copy import deepcopy

import numpy as np
import pytest

from drosolearn.normalization_loss import (
    CATEGORIES, SCIENCE, category_masks, row_metrics, stats, validate_config,
)


@pytest.fixture
def scores():
    baseline = np.array([[2., 1., -3., -4.], [0., 0., 0., 0.],
                         [4., -2., 2., -4.]])
    desired = np.array([[2., 1., 0., -4.], [0., 0., 0., 0.],
                        [4., 0., 2., -4.]])
    scales = np.array([.25, 1., .5])
    actual = desired * scales[:, None]
    return baseline, desired, actual, scales


def measure(scores, mask=None, phase=None, nonzero=None):
    B, P, A, scales = scores
    return row_metrics(B, P, A,
        np.ones(B.shape, bool) if mask is None else mask,
        np.ones(len(B), bool) if phase is None else phase,
        scales, 1. / scales,
        np.array([True, False, True]) if nonzero is None else nonzero)


def test_three_stage_signed_losses_have_separate_meanings(scores):
    row = measure(scores)
    assert row['baseline_helpful'] == 9.
    assert row['baseline_harmful'] == 13.
    assert row['baseline_net'] == -4.
    assert row['desired_helpful'] == 9.
    assert row['desired_harmful'] == 8.
    assert row['desired_net'] == 1.
    assert row['actual_helpful'] == 3.75
    assert row['actual_harmful'] == 3.
    assert row['actual_net'] == .75
    assert row['suppression_net'] == -5.
    assert row['normalization_net'] == .25
    assert row['baseline_net'] - row['actual_net'] == (
        row['suppression_net'] + row['normalization_net'])


def test_normalization_reduces_retained_help_and_retained_harm(scores):
    row = measure(scores)
    assert row['helpful_loss'] == 5.25
    assert row['harmful_reduction'] == 5.
    assert row['helpful_retention'] == pytest.approx(3.75 / 9.)
    assert row['harmful_retention'] == pytest.approx(3. / 8.)


def test_nogo_direction_mean_does_not_describe_realized_update_mean(scores):
    row = measure(scores)
    assert row['trial_count'] == 3 and row['updated_trials'] == 2
    assert row['cells'] == 12 and row['updated_cells'] == 8
    assert row['scale_mean_all'] == pytest.approx(7. / 12.)
    assert row['scale_mean_updated'] == .375
    assert row['raw_norm_mean'] == pytest.approx(7. / 3.)
    assert row['scale_min'] == .25 and row['scale_max'] == 1.
    assert row['helpful_retention'] != row['scale_mean_all']
    assert row['helpful_retention'] != row['scale_mean_updated']


def test_direction_mean_is_not_weighted_by_destination_cell_count(scores):
    mask = np.array([[True, False, False, False], [True, True, True, True],
                     [True, True, False, False]])
    row = measure(scores, mask=mask)
    assert row['cells'] == 7 and row['updated_cells'] == 3
    assert row['scale_mean_all'] == pytest.approx((.25 + 1. + .5) / 3.)


def test_empty_category_still_reports_phase_direction_statistics(scores):
    row = measure(scores, mask=np.zeros((3, 4), bool))
    assert row['trial_count'] == 3 and row['cells'] == 0
    assert row['scale_mean_all'] == pytest.approx(7. / 12.)
    assert row['helpful_retention'] is None
    assert row['harmful_retention'] is None


def test_phase_selects_trials_before_destination_mass(scores):
    row = measure(scores, phase=np.array([True, True, False]))
    assert row['trial_count'] == 2 and row['updated_trials'] == 1
    assert row['scale_mean_all'] == .625
    assert row['scale_mean_updated'] == .25
    assert row['desired_helpful'] == 3.
    assert row['actual_helpful'] == .75


def test_empty_phase_does_not_fill_undefined_values_with_zero(scores):
    row = measure(scores, phase=np.zeros(3, bool))
    for key in ['trial_count', 'updated_trials', 'cells', 'updated_cells',
                'baseline_helpful', 'desired_harmful', 'actual_net']:
        assert row[key] == 0
    for key in ['scale_mean_all', 'scale_mean_updated', 'raw_norm_mean',
                'scale_min', 'scale_max', 'helpful_retention', 'harmful_retention']:
        assert row[key] is None


def test_no_updates_keeps_direction_values_but_no_realized_retention():
    zero = np.zeros((2, 4))
    row = row_metrics(zero, zero, zero, np.ones((2, 4), bool),
        np.ones(2, bool), np.array([.1, 1.]), np.array([10., 1.]),
        np.zeros(2, bool))
    assert row['scale_mean_all'] == .55
    assert row['scale_mean_updated'] is None
    assert row['helpful_retention'] is None
    assert row['harmful_retention'] is None


def test_cross_removed_before_normalization_has_undefined_retention():
    B = np.array([[1., .8, -.6, -.2]])
    P = np.array([[1., 0., 0., 0.]])
    A = P * .2
    cross = np.array([[False, True, True, True]])
    row = row_metrics(B, P, A, cross, np.ones(1, bool),
        np.array([.2]), np.array([5.]), np.ones(1, bool))
    assert row['desired_helpful'] == row['actual_helpful'] == 0.
    assert row['helpful_loss'] == 0.
    assert row['helpful_retention'] is None
    assert row['harmful_retention'] is None
    assert row['baseline_helpful'] == .8


def test_numerical_neutral_score_mass_is_not_deleted():
    B = np.array([[0., 4e-10, -2e-10, 0.]])
    row = row_metrics(B, B, B * .5, np.ones((1, 4), bool),
        np.ones(1, bool), np.array([.5]), np.array([2.]), np.ones(1, bool))
    assert row['desired_helpful'] == 4e-10
    assert row['desired_harmful'] == 2e-10
    assert row['helpful_retention'] == row['harmful_retention'] == .5


def test_helpers_leave_all_input_arrays_unchanged(scores):
    saved = [value.copy() for value in scores]
    measure(scores)
    assert all(np.array_equal(before, after) for before, after in zip(saved, scores))


@pytest.mark.parametrize('which', ['baseline', 'desired', 'actual', 'scales', 'norms'])
def test_nonfinite_row_inputs_fail(scores, which):
    B, P, A, scales = (value.copy() for value in scores)
    norms = 1. / scales
    {'baseline': B, 'desired': P, 'actual': A,
     'scales': scales, 'norms': norms}[which].flat[0] = np.nan
    with pytest.raises(ValueError):
        row_metrics(B, P, A, np.ones((3, 4), bool), np.ones(3, bool),
            scales, norms, np.ones(3, bool))


@pytest.mark.parametrize('which', ['mask', 'phase', 'nonzero'])
def test_boolean_row_masks_are_required(scores, which):
    B, P, A, scales = scores
    args = {'mask': np.ones((3, 4), bool), 'phase': np.ones(3, bool),
            'nonzero': np.ones(3, bool)}
    args[which] = args[which].astype(int)
    with pytest.raises(ValueError):
        row_metrics(B, P, A, args['mask'], args['phase'], scales,
            1. / scales, args['nonzero'])


@pytest.mark.parametrize('badshape', [(3, 3), (12,), (3, 4, 1)])
def test_destination_axis_cannot_be_silently_transposed_or_broadcast(scores, badshape):
    B, P, A, scales = scores
    with pytest.raises(ValueError):
        row_metrics(B, P, A, np.ones(badshape, bool), np.ones(3, bool),
            scales, 1. / scales, np.ones(3, bool))


@pytest.mark.parametrize('badvalue', [0., -1., np.inf])
def test_normalization_lengths_must_be_finite_positive(scores, badvalue):
    B, P, A, scales = scores
    norms = 1. / scales
    norms[0] = badvalue
    with pytest.raises(ValueError):
        row_metrics(B, P, A, np.ones((3, 4), bool), np.ones(3, bool),
            scales, norms, np.ones(3, bool))


def test_category_help_depends_on_geometric_sign_not_same_target():
    sources = np.array([0])
    labels = np.array([1, 1, -1, -1])
    gram = np.array([[1., -.3, -.4, .2]])
    known = np.array([[1, 0, -1, 0]])
    masks = category_masks(sources, labels, gram, known, 1e-8)
    assert list(masks) == CATEGORIES
    assert masks['self'].tolist() == [[True, False, False, False]]
    assert masks['cross_same'].tolist() == [[False, True, False, False]]
    assert masks['cross_opposite'].tolist() == [[False, False, True, True]]
    assert masks['cross_helpful'].tolist() == [[False, False, True, False]]
    assert masks['cross_harmful'].tolist() == [[False, True, False, True]]
    assert masks['cross_unknown'].tolist() == [[False, True, False, True]]


def test_help_harm_and_target_category_partitions_are_complete():
    sources = np.array([0, 2])
    labels = np.array([1, 1, -1, -1])
    gram = np.array([[1., 1e-8, -.4, .2], [.1, -.7, 1., -.2]])
    known = np.array([[1, 0, -1, 0], [1, 1, -1, -1]])
    m = category_masks(sources, labels, gram, known, 1e-8)
    assert np.array_equal(m['self'].astype(int) + m['cross_all'], m['all'])
    assert np.array_equal(m['cross_helpful'].astype(int) + m['cross_harmful'] +
                          m['cross_neutral'], m['cross_all'])
    assert np.array_equal(m['cross_same'].astype(int) + m['cross_opposite'], m['cross_all'])
    assert m['cross_neutral'][0, 1]
    assert np.all(m['cross_unknown'] <= m['cross_all'])
    assert not m['cross_unknown'][1].any()


def test_unknown_is_an_overlapping_information_group():
    m = category_masks(np.array([0]), np.array([1, 1, -1, -1]),
        np.array([[1., .4, .2, 0.]]), np.array([[1, 0, 0, 0]]), 1e-8)
    assert (m['cross_unknown'] & m['cross_helpful']).sum() == 1
    assert (m['cross_unknown'] & m['cross_harmful']).sum() == 1
    assert (m['cross_unknown'] & m['cross_neutral']).sum() == 1


@pytest.mark.parametrize('source', [np.array([-1]), np.array([4]), np.array([0.]),
                                    np.array([[0]]), np.array([False])])
def test_category_source_index_contract(source):
    with pytest.raises(ValueError):
        category_masks(source, np.array([1, 1, -1, -1]),
            np.ones((1, 4)), np.zeros((1, 4)), 1e-8)


@pytest.mark.parametrize('which', ['labels', 'gram', 'known', 'tolerance'])
def test_category_nonfinite_or_invalid_information_contract(which):
    labels = np.array([1, 1, -1, -1])
    gram = np.ones((1, 4))
    known = np.zeros((1, 4))
    tol = 1e-8
    if which == 'labels': labels[0] = 0
    elif which == 'gram': gram[0, 0] = np.nan
    elif which == 'known': known[0, 0] = 2
    else: tol = 0.
    with pytest.raises(ValueError):
        category_masks(np.array([0]), labels, gram, known, tol)


def test_statistics_keep_undefined_seed_counts_explicit():
    assert stats([1., None, 3.]) == {
        'mean': 2., 'sd': pytest.approx(np.sqrt(2.)), 'valid_count': 2, 'total_count': 3}
    assert stats([None, None]) == {
        'mean': None, 'sd': None, 'valid_count': 0, 'total_count': 2}
    assert stats([None, .5]) == {
        'mean': .5, 'sd': 0., 'valid_count': 1, 'total_count': 2}


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_statistics_reject_nonfinite_valid_values(value):
    with pytest.raises(ValueError):
        stats([None, value])


def test_seed_ratio_mean_is_not_implicitly_pooled_mass_ratio():
    seed_ratios = [.2, .8]
    desired_masses = [9., 1.]
    pooled = sum(r * m for r, m in zip(seed_ratios, desired_masses)) / sum(desired_masses)
    assert stats(seed_ratios)['mean'] == .5
    assert pooled == pytest.approx(.26)


@pytest.mark.parametrize('key,value', [
    ('neutral_tolerance', 1e-6), ('head_lr', .2), ('labels', [1, -1, 1, -1]),
    ('model_seeds', [17401]), ('methods', ['feedback_selective']),
    ('training_trials', 800), ('categories', ['self']), ('phases', ['all']),
])
def test_fixed_saved_study_cannot_be_redefined_after_results(key, value):
    cfg = deepcopy(SCIENCE)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def fixture_config():
    cfg = deepcopy(SCIENCE)
    cfg.update(scope='integration_test', model_seeds=[90001],
               training_trials=4, checkpoints=[0, 4])
    return cfg


def test_bounded_invented_config_does_not_use_actual_study_seeds():
    validate_config(fixture_config())


@pytest.mark.parametrize('key,value', [
    ('model_seeds', [17401]), ('model_seeds', [90001, 90001]),
    ('model_seeds', [90101]), ('training_trials', 20), ('training_trials', 6),
    ('checkpoints', [0, 0, 4]), ('checkpoints', [1, 4]),
    ('checkpoints', []), ('checkpoints', None), ('checkpoints', (0, 4)),
])
def test_invented_fixture_scope_is_bounded(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
