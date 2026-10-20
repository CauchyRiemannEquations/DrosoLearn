"""Independent factor arithmetic on invented saved tensors only."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("self_blend_factors_independent_tests", ROOT / "scripts/verify_self_blend_factors.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
FIXTURE_SPEC = importlib.util.spec_from_file_location("self_blend_factors_handwritten_fixture", ROOT / "tests/test_self_blend_factors.py")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


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


@pytest.fixture(scope="module")
def independent_fixture():
    cfg, parent, accounting = fixtures.invented_saved_paths()
    arrays = audit.expected_artifacts(cfg, parent, accounting)
    return cfg, parent, accounting, arrays, audit.expected_rows(cfg, arrays)


def test_full_current_factor_producer_matches_independent_saved_tensor_calculations(independent_fixture):
    from drosolearn.self_blend_factors import analyze, summarize_arrays
    cfg, parent, accounting, wanted, wanted_rows = independent_fixture
    arrays, rows = analyze(cfg, parent, accounting)
    audit.check_artifacts(arrays, wanted, rows, wanted_rows)
    summary = summarize_arrays(cfg, arrays, rows, parent, accounting)
    receipt = audit.verify_science(cfg, parent, accounting, arrays, rows, summary, allow_fixture=True)
    assert len(arrays) == 14 and len(rows) == 360 and len(rows[0]) == 15
    assert len(summary["families"]) == 30
    assert receipt["parent_trial_records_checked"] == 432 and receipt["destination_effects_checked"] == 1728


def test_only_post_both_actual_go_records_are_crossed_and_unpaired_effects_are_retained(independent_fixture, monkeypatch):
    cfg, parent, accounting, arrays, rows = independent_fixture
    assert arrays["pair_action_cases"][0, 0, 0, 0, 4:8].tolist() == [2, 1, 3, 0]
    post = arrays["reference_post"]
    assert np.array_equal(arrays["bridge_eligible"], post[:, :, None] & (arrays["pair_action_cases"] == 3))
    for name in ("bridge_q_cells", "bridge_q_shapley", "bridge_positive_shapley", "bridge_negative_shapley", "factual_pair_diagnostics"):
        assert not arrays[name][~arrays["bridge_eligible"]].any()
    original, calls = audit.independent_bridge, []
    def tracked(zc, zg, gc, gg, fc, fg):
        calls.append(len(gc))
        return original(zc, zg, gc, gg, fc, fg)
    monkeypatch.setattr(audit, "independent_bridge", tracked)
    audit.expected_artifacts(cfg, parent, accounting)
    assert sum(calls) == int(arrays["bridge_eligible"].sum())
    assert sum(calls) < arrays["bridge_eligible"].size
    selected = [row for row in rows if (row["topology"], row["representation"], row["control"], row["destination_group"])
        == ("real", "raw", "isolated", "own")]
    assert len(selected) == 6
    assert next(row for row in selected if row["component"] == "control_only_Go")["probability_signed_sum"] < 0.
    assert next(row for row in selected if row["component"] == "self_blend_only_Go")["probability_signed_sum"] > 0.
    assert next(row for row in selected if row["component"] == "neither_Go")["probability_signed_sum"] == 0.


def test_each_group_reconstructs_signed_positive_and_negative_actual_post_difference(independent_fixture):
    cfg, parent, accounting, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent, accounting)
    for family in summary["families"]:
        for record in family["seeds"]:
            for name, factual in (("component_endpoint_contributions", "factual_post_endpoint_difference"),
                ("component_positive_endpoint_allocations", "factual_post_positive_endpoint_difference"),
                ("component_negative_endpoint_allocations", "factual_post_negative_endpoint_difference")):
                assert np.allclose(np.asarray(record[name]).sum(axis=-1), record[factual], atol=1e-12)
            assert sum(record["case_counts"]) == record["reference_post_trials"]
            assert record["both_methods_full_before_on_eligible_trials"] + record["control_partial_before_on_eligible_trials"] == record["both_Go_trials"]


def test_positive_mass_allocations_can_be_negative_without_becoming_harm():
    cells = np.array([[1. - .2 * bool(code & 1)] * 4 for code in range(8)])
    positive = audit.independent_shapley(np.maximum(cells, 0.))
    negative = audit.independent_shapley(np.maximum(-cells, 0.))
    signed = audit.independent_shapley(cells)
    assert np.all(positive[2] < 0.) and not negative.any()
    assert np.allclose(signed, positive - negative)


def test_factual_descriptors_use_current_source_signed_scalar_and_real_direction_norm(independent_fixture):
    _, parent, _, arrays, _ = independent_fixture
    eligible = arrays["bridge_eligible"][0, 0, 0, 0]
    positions = np.flatnonzero(eligible)
    values = arrays["factual_pair_diagnostics"][0, 0, 0, 0][eligible]
    assert positions.size > 0
    for j, method in enumerate((4, 5)):
        norms = np.linalg.norm(parent["trial_directions"][0, 0, method, 0][eligible], axis=-1)
        g = parent["trial_update_factors"][0, 0, method, 0][eligible]
        assert np.allclose(values[:, j, 3], np.abs(g))
        assert np.allclose(values[:, j, 4], np.abs(g) * norms)
        assert np.allclose(values[:, j, 7], norms)
    assert np.allclose(values[:, :, 3],values[:, :, 4])
    assert np.allclose(values[:, :, 7],1.)
    assert np.allclose(values[:, :, 2], values[:, :, 1] * (1. - values[:, :, 1]))


def test_no_shared_go_and_never_opened_gate_leave_null_diagnostics_and_zero_padding(independent_fixture):
    from drosolearn.self_blend_factors import analyze, summarize_arrays
    cfg, parent, accounting, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent, accounting)
    shared = [family for family in summary["families"] if family["control"] == "shared"]
    assert all(family["means"]["diagnostic_means"] == [[None] * 8, [None] * 8] for family in shared)
    cfg, parent, accounting = fixtures.invented_saved_paths(gate_never=True)
    arrays, rows = analyze(cfg, parent, accounting)
    summary = summarize_arrays(cfg, arrays, rows, parent, accounting)
    audit.verify_science(cfg, parent, accounting, arrays, rows, summary, allow_fixture=True)
    assert np.all(arrays["first_post_trials"] == 13)
    assert not arrays["bridge_eligible"].any() and not arrays["bridge_q_cells"].any()
    assert all(family["means"]["diagnostic_means"] == [[None] * 8, [None] * 8] for family in summary["families"])


def test_control_knowledge_is_not_silently_completed_from_reference_gate(independent_fixture):
    cfg, parent, accounting, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent, accounting)
    partial = next(family for family in summary["families"] if family["control"] == "feedback_selective")
    assert partial["seeds"][0]["control_partial_before_on_eligible_trials"] > 0


@pytest.mark.parametrize("field", audit.FIELDS)
def test_all_fourteen_derived_arrays_are_independently_bound(independent_fixture, field):
    _, _, _, original, rows = independent_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, rows, rows)


@pytest.mark.parametrize("field", ["trial_update_factors", "trial_directions", "trial_correct_probability_delta",
    "trial_aligned_margin_delta", "checkpoint_heads", "checkpoint_probabilities", "first_blend_mode_trials",
    "model_seeds", "targets", "schedules"])
def test_parent_factual_binding_corruption_is_rejected(independent_fixture, field):
    cfg, original, accounting, _, _ = independent_fixture
    parent = copy.deepcopy(original)
    parent[field].reshape(-1)[-1] += 1 if parent[field].dtype.kind in "iu" else .03
    with pytest.raises(ValueError): audit.expected_artifacts(cfg, parent, accounting)


@pytest.mark.parametrize("field", ["own_endpoint_contribution_by_seed", "other_endpoint_contribution_by_seed"])
def test_original_displayed_factual_accounting_cannot_be_replaced(independent_fixture, field):
    cfg, parent, original, _, _ = independent_fixture
    prior_plot = copy.deepcopy(original)
    prior_plot["factual_post_pathways"][0][field][0] += .03
    with pytest.raises(ValueError): audit.expected_artifacts(cfg, parent, prior_plot)


@pytest.mark.parametrize("field", ["reference_post_trials", "both_Go_trials", "component_trials", "probability_signed_sum",
    "positive_mass_allocation", "negative_mass_allocation", "endpoint_contribution", "positive_endpoint_allocation", "negative_endpoint_allocation"])
def test_csv_component_counts_and_allocations_cannot_be_changed(independent_fixture, field):
    _, _, _, arrays, original = independent_fixture
    rows = copy.deepcopy(original)
    rows[0][field] += 1
    with pytest.raises(ValueError): audit.check_artifacts(arrays, arrays, rows, original)


@pytest.mark.parametrize("key,value", [("bridge_eligibility", "any_actual_Go"), ("bridge_bits", ["direction", "scalar", "state"]),
    ("bridge_allocation", "linearized"), ("contrast_controls", ["isolated"]), ("numeric_tolerance", .001),
    ("new_confirmatory_hypothesis", True), ("phase_definition", "each_method_own_gate")])
def test_fixed_descriptive_factor_choices_cannot_be_relaxed(key, value):
    cfg, _, _ = fixtures.invented_saved_paths()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_cannot_enter_actual_registered_analysis():
    cfg, _, _ = fixtures.invented_saved_paths()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17701]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


@pytest.mark.parametrize("target", ["parent_verdict", "diagnostic", "component_gain", "new_training"])
def test_summary_cannot_hide_old_failures_or_change_allocations(independent_fixture, target):
    cfg, parent, accounting, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent, accounting)
    if target == "parent_verdict": summary["interpretation"]["parent_H_gain"] = True
    elif target == "diagnostic": summary["families"][0]["means"]["diagnostic_means"][0][0] += .1
    elif target == "component_gain": summary["families"][0]["means"]["component_endpoint_contributions"][0][0] += .1
    else: summary["counts"]["new_training_trials"] = 1
    with pytest.raises(ValueError): audit.verify_science(cfg, parent, accounting, arrays, rows, summary, allow_fixture=True)


def test_analysis_and_verification_do_not_call_rng_or_optimization(independent_fixture, monkeypatch):
    from drosolearn.self_blend_factors import analyze, summarize_arrays
    def forbidden(*args, **kwargs): raise AssertionError("Saved one-step accounting cannot draw, construct or optimize")
    for name in ("default_rng", "SeedSequence", "random", "rand"): monkeypatch.setattr(np.random, name, forbidden)
    for name in ("svd", "pinv", "lstsq"): monkeypatch.setattr(np.linalg, name, forbidden)
    cfg, parent, accounting, _, _ = independent_fixture
    arrays, rows = analyze(cfg, parent, accounting)
    summary = summarize_arrays(cfg, arrays, rows, parent, accounting)
    audit.verify_science(cfg, parent, accounting, arrays, rows, summary, allow_fixture=True)


def test_geometry_uses_normalized_coefficients_on_the_same_actual_old_gated_Go_events(independent_fixture):
    cfg,parent,_,arrays,_=independent_fixture
    geometry=audit.mixture_geometry(cfg,parent,arrays)
    assert len(geometry)==6
    for family in geometry:
        ti=audit.TOPS.index(family['topology']);ri=audit.REPS.index(family['representation'])
        record=family['seeds'][0];mask=arrays['bridge_eligible'][ti,ri,0,0]
        g=(ti,ri,5,0);c=(ti,ri,4,0)
        norm=parent['trial_blend_raw_norm'][g][mask];coefficient=.25/norm
        assert record['eligible_both_Go_trials']==int(mask.sum())
        assert record['has_both_Go_evidence'] is True
        assert record['raw_norm_mean']==pytest.approx(norm.mean())
        assert record['balanced_weight_after_normalization_mean']==pytest.approx(coefficient.mean())
        assert record['balanced_weight_after_normalization_min']==pytest.approx(coefficient.min())
        assert record['balanced_weight_after_normalization_max']==pytest.approx(coefficient.max())
        assert record['self_weight_after_normalization_mean']==pytest.approx((.75/norm).mean())
        assert np.all(coefficient>.25) and np.all(coefficient<.5)
        np.testing.assert_allclose(parent['trial_blend_balanced_directions'][g][mask],parent['trial_directions'][c][mask],atol=1e-14)
        assert max(record[key] for key in audit.MIX_RESIDUALS)<1e-12


def test_never_opened_gate_has_undefined_geometry_means_and_zero_residuals():
    cfg,parent,plot=fixtures.invented_saved_paths(gate_never=True)
    arrays=audit.expected_artifacts(cfg,parent,plot)
    for family in audit.mixture_geometry(cfg,parent,arrays):
        r=family['seeds'][0]
        assert r['eligible_both_Go_trials']==0 and r['has_both_Go_evidence'] is False
        assert all(r[key] is None for key in ('raw_norm_mean','balanced_weight_after_normalization_mean',
            'balanced_weight_after_normalization_min','balanced_weight_after_normalization_max','self_weight_after_normalization_mean'))
        assert all(r[key]==0. for key in audit.MIX_RESIDUALS)


@pytest.mark.parametrize('field',['trial_blend_isolated_directions','trial_blend_balanced_directions',
    'trial_blend_raw_norm','trial_blend_applicable'])
def test_corrupt_recorded_mixture_ingredient_or_activation_fails_geometry(independent_fixture,field):
    cfg,original,_,arrays,_=independent_fixture
    parent=copy.deepcopy(original)
    t=int(np.flatnonzero(arrays['bridge_eligible'][0,0,0,0])[0]);at=(0,0,5,0,t)
    if field=='trial_blend_applicable':parent[field][at]=False
    elif field=='trial_blend_raw_norm':parent[field][at]+=.02
    else:parent[field][at][0]+=.02
    with pytest.raises(ValueError):audit.mixture_geometry(cfg,parent,arrays)


@pytest.mark.parametrize('norm',[0.,-.1,np.nan,np.inf])
def test_nonpositive_or_nonfinite_saved_norm_is_not_silently_normalized(independent_fixture,norm):
    cfg,original,_,arrays,_=independent_fixture;parent=copy.deepcopy(original)
    t=int(np.flatnonzero(arrays['bridge_eligible'][0,0,0,0])[0]);parent['trial_blend_raw_norm'][0,0,5,0,t]=norm
    with pytest.raises(ValueError):audit.mixture_geometry(cfg,parent,arrays)


def test_geometry_does_not_borrow_unpaired_or_pre_gate_new_directions(independent_fixture):
    cfg,original,_,arrays,_=independent_fixture;parent=copy.deepcopy(original)
    wanted=audit.mixture_geometry(cfg,parent,arrays)
    for ti in range(3):
        for ri in range(2):
            outside=~arrays['bridge_eligible'][ti,ri,0,0]
            parent['trial_blend_raw_norm'][ti,ri,5,0][outside]=np.nan
            parent['trial_blend_isolated_directions'][ti,ri,5,0][outside]=np.nan
            parent['trial_blend_balanced_directions'][ti,ri,5,0][outside]=np.nan
    # The immutable parent gate remains responsible for original bytes. This
    # helper describes only its registered paired-Go subset.
    audit.compare_tree(audit.mixture_geometry(cfg,parent,arrays),wanted)


def test_unit_other_coefficient_is_not_an_actual_probability_retention_promise():
    coefficient=.27
    cells=audit.independent_bridge(np.zeros(4),np.full(4,7.),np.asarray(.1),np.asarray(.01),
        np.ones(4),np.full(4,coefficient))
    assert 0.<abs(cells[7,0])<coefficient*abs(cells[0,0])*.01


@pytest.mark.parametrize('mutation',['missing_method','duplicate_method','wrong_order','missing_field','wrong_seed_count','nonfinite','bool','not_record'])
def test_original_six_method_post_ledger_has_exact_complete_finite_seed_structure(independent_fixture,mutation):
    cfg,parent,original,_,_=independent_fixture;plot=copy.deepcopy(original)
    rows=plot['factual_post_pathways']
    if mutation=='missing_method':rows.pop()
    elif mutation=='duplicate_method':rows[1]['method']=rows[0]['method']
    elif mutation=='wrong_order':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='missing_field':rows[0].pop('own_endpoint_contribution_by_seed')
    elif mutation=='wrong_seed_count':rows[0]['own_endpoint_contribution_by_seed']=[]
    elif mutation=='nonfinite':rows[0]['own_endpoint_contribution_by_seed']=[np.nan]
    elif mutation=='bool':rows[0]['own_endpoint_contribution_by_seed']=[False]
    else:rows[0]=None
    with pytest.raises(ValueError):audit.expected_artifacts(cfg,parent,plot)


@pytest.mark.parametrize('field',['raw_norm_mean','balanced_weight_after_normalization_mean',
    'balanced_weight_after_normalization_min','balanced_weight_after_normalization_max',
    'self_weight_after_normalization_mean',*audit.MIX_RESIDUALS,'eligible_both_Go_trials','has_both_Go_evidence'])
def test_saved_mixture_summary_cannot_hide_norm_coefficients_or_equation_failure(independent_fixture,field):
    cfg,parent,plot,arrays,rows=independent_fixture
    summary=audit.expected_summary(cfg,arrays,rows,parent,plot)
    record=summary['mixture_geometry'][0]['seeds'][0];value=record[field]
    record[field]=not value if isinstance(value,bool) else value+1
    with pytest.raises(ValueError):audit.verify_science(cfg,parent,plot,arrays,rows,summary,allow_fixture=True)


def test_parent_side_old_gated_is_primary_control_and_all_five_controls_are_retained(independent_fixture):
    cfg,parent,plot,arrays,rows=independent_fixture
    summary=audit.expected_summary(cfg,arrays,rows,parent,plot)
    assert summary['control_order']==['gated_balanced','isolated','balanced','feedback_selective','shared']
    assert summary['families'][0]['control']=='gated_balanced' and summary['families'][0]['role']=='primary'
    assert len(summary['families'])==30
    assert summary['counts']['reference_gate_paths']==6
    assert summary['counts']['comparison_paths']==30
    assert summary['counts']['paired_trial_records']==360
    assert summary['counts']['analyzed_parent_trial_records']==432
    assert summary['counts']['analyzed_destination_effects']==1728
    assert tuple(summary['validation']['residuals'])[:2]==('prior_primary_post_bookkeeping','saved_unit_mixture_geometry')
