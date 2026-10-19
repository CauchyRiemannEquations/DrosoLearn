"""Exact saved-path accounting and crossed-update algebra on invented data."""
from copy import deepcopy
import numpy as np
import pytest


def module():
    import drosolearn.balanced_pathways as study
    return study


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


@pytest.fixture
def rows():
    raw = np.array([[1., 0., 0., 0., .2, .3], [0., 1., 0., 0., .2, .3],
        [0., 0., 1., 0., .2, .3], [0., 0., 0., 1., .2, .3]])
    return raw / np.linalg.norm(raw, axis=1)[:, None]


def test_probability_effects_are_exact_before_after_and_do_not_mutate_inputs(rows):
    head = np.array([.2, -.5, .6, .1, -.2, .3])
    direction = np.array([.3, .1, -.3, .2, .1, -.1])
    labels = np.array([1, 1, -1, -1])
    originals = [value.copy() for value in [rows, head, direction, labels]]
    probability, margin = module().probability_effects(rows, head, -.04, direction, labels)
    old, new = rows @ head, rows @ (head - .04 * direction)
    assert np.allclose(probability, labels * (sigmoid(new) - sigmoid(old)), atol=1e-14)
    assert np.allclose(margin, labels * (new - old), atol=1e-14)
    for actual, original in zip([rows, head, direction, labels], originals):
        assert np.array_equal(actual, original)


def test_probability_effects_batched_last_axes_match_handwritten_scalar_calculations(rows):
    head = np.array([[.1, .2, -.3, .4, .2, .1], [-.2, .3, .1, -.4, .1, .2]])
    direction = np.array([[.3, .1, .2, -.2, .1, .1], [.1, -.3, .2, .2, .1, .2]])
    factor = np.array([.03, -.04]); signs = np.array([1, 1, -1, -1])
    probability, margin = module().probability_effects(rows, head, factor, direction, signs)
    assert probability.shape == margin.shape == (2, 4)
    for i in range(2):
        old, new = rows @ head[i], rows @ (head[i] + factor[i] * direction[i])
        assert np.allclose(probability[i], signs * (sigmoid(new) - sigmoid(old)), atol=1e-14)
        assert np.allclose(margin[i], signs * (new - old), atol=1e-14)


@pytest.mark.parametrize("factor", [0., .02, -.02])
def test_zero_scalar_or_zero_direction_has_zero_effect(rows, factor):
    head = np.array([.1, .2, .3, .4, .5, .6])
    for g,w in [(0., rows[0]), (factor, np.zeros(6))]:
        probability, margin = module().probability_effects(rows, head, g, w, np.array([1, 1, -1, -1]))
        assert not probability.any() and not margin.any()


def test_probability_change_is_not_assumed_equal_to_logit_change():
    rows = np.eye(4); head = np.array([0., 3., -3., 1.])
    probability, margin = module().probability_effects(rows, head, .05, np.ones(4), np.ones(4, int))
    assert np.allclose(margin, np.full(4, .05))
    assert probability[0] > probability[1] > 0.
    assert not np.allclose(probability, margin * .25)


def test_parent_sigmoid_clipping_is_preserved_in_extreme_saved_states():
    rows = np.eye(4); head = np.array([50., -50., 0., 1.])
    signs = np.array([1, -1, 1, -1])
    probability, margin = module().probability_effects(rows, head, .1, np.ones(4), signs)
    assert probability[0] == probability[1] == 0.
    assert np.allclose(margin, signs * .1)


@pytest.mark.parametrize("source", range(4))
def test_categories_partition_destinations_once_and_never_omit_unknown_source(source):
    known = np.array([1, 0, -1, 0], np.int8)
    category = module().decompose_categories(known, np.asarray(source))
    assert category.shape == (4,) and category[source] == 0
    for j in range(4):
        expected = 0 if j == source else (1 if known[j] else 2)
        assert category[j] == expected
    assert sum(np.count_nonzero(category == c) for c in range(3)) == 4


def test_batched_category_masks_use_each_actual_learner_memory():
    known = np.array([[0, 0, 0, 0], [1, 1, -1, -1], [0, 1, -1, 0]], np.int8)
    sources = np.array([2, 0, 3])
    category = module().decompose_categories(known, sources)
    assert category.tolist() == [[2, 2, 0, 2], [0, 1, 1, 1], [2, 1, 1, 0]]


def test_help_and_harm_can_be_reported_separately_without_changing_signed_balance():
    delta = np.array([.03, -.01, .02, -.04])
    categories = module().decompose_categories(np.array([1, 1, 0, 0]), np.asarray(0))
    signed = positive = negative = 0.
    for category in range(3):
        part = delta[categories == category]
        plus, minus = np.maximum(part, 0.).sum(), np.maximum(-part, 0.).sum()
        assert plus - minus == pytest.approx(part.sum())
        positive += plus; negative += minus; signed += part.sum()
    assert signed == pytest.approx(delta.sum())
    assert positive == pytest.approx(.05) and negative == pytest.approx(.05)


@pytest.fixture
def bridge_arguments():
    return (np.array([.1, -.3, .4, -.2]), np.array([.4, -.1, .2, -.5]),
        .04, .03, np.array([1., .2, -.1, .3]), np.array([.8, .3, -.2, .1]), np.array([1, 1, -1, -1]))


def test_bridge_bits_are_state_four_scalar_two_direction_one(bridge_arguments):
    z_s,z_b,g_s,g_b,f_s,f_b,signs = bridge_arguments
    cells = module().bridge_cells(*bridge_arguments)
    assert cells.shape == (8, 4)
    for index in range(8):
        z = z_b if index & 4 else z_s
        g = g_b if index & 2 else g_s
        f = f_b if index & 1 else f_s
        expected = signs * (sigmoid(z + g * f) - sigmoid(z))
        assert np.allclose(cells[index], expected, atol=1e-14)


def test_bridge_all_batched_corners_match_scalar_formula(bridge_arguments):
    z_s,z_b,g_s,g_b,f_s,f_b,signs = bridge_arguments
    batch = module().bridge_cells(np.stack([z_s,z_s+.1]), np.stack([z_b,z_b-.1]),
        np.array([g_s,-g_s]), np.array([g_b,-g_b]), np.stack([f_s,-f_s]), np.stack([f_b,-f_b]), signs)
    assert batch.shape == (2, 8, 4)
    assert np.allclose(batch[0],module().bridge_cells(*bridge_arguments),atol=1e-14)
    second=module().bridge_cells(z_s+.1,z_b-.1,-g_s,-g_b,-f_s,-f_b,signs)
    assert np.allclose(batch[1],second,atol=1e-14)


def test_crossed_single_step_does_not_mutate_or_advance_a_trajectory(bridge_arguments):
    originals = [value.copy() if isinstance(value,np.ndarray) else value for value in bridge_arguments]
    module().bridge_cells(*bridge_arguments)
    for actual,original in zip(bridge_arguments,originals):
        assert np.array_equal(actual,original)


def test_shapley_additive_factors_recover_each_explicit_contribution():
    effects = np.array([[.01, .02, -.03, .04], [-.02, .03, .01, -.01], [.03, -.01, .02, .01]])
    baseline = np.array([.1, -.1, .2, -.2])
    cells = np.array([baseline + sum((effects[j] for j,bit in enumerate([4,2,1]) if index&bit),start=np.zeros(4))
        for index in range(8)])
    actual = module().shapley3(cells)
    assert np.allclose(actual,effects,atol=1e-14)
    assert np.allclose(actual.sum(0),cells[7]-cells[0],atol=1e-14)


def test_three_way_interaction_is_split_equally_over_three_factors():
    cells=np.zeros((8,4));cells[7]=[.03,.06,-.09,.12]
    actual=module().shapley3(cells)
    assert np.allclose(actual,np.broadcast_to(cells[7]/3.,(3,4)),atol=1e-14)


def test_two_way_interaction_does_not_leak_into_uninvolved_direction_factor():
    cells=np.zeros((8,4));cells[[6,7]]=[.02,-.04,.06,-.08]
    actual=module().shapley3(cells)
    assert np.allclose(actual[0],cells[7]/2.) and np.allclose(actual[1],cells[7]/2.)
    assert np.allclose(actual[2],0.,atol=1e-14)


def test_shapley_preserves_batch_axes_and_exact_factual_difference(bridge_arguments):
    cells=module().bridge_cells(*bridge_arguments)
    batch=np.stack([cells,2*cells,cells[::-1]])
    actual=module().shapley3(batch)
    assert actual.shape==(3,3,4)
    assert np.allclose(actual.sum(-2),batch[:,7]-batch[:,0],atol=1e-14)
    assert np.allclose(actual[1],2*actual[0],atol=1e-14)
    assert np.allclose(actual[2],-actual[0],atol=1e-14)


def test_unchanged_direction_has_zero_direction_shapley(bridge_arguments):
    z_s,z_b,g_s,g_b,f_s,_,signs=bridge_arguments
    cells=module().bridge_cells(z_s,z_b,g_s,g_b,f_s,f_s,signs)
    actual=module().shapley3(cells)
    assert np.allclose(actual[2],0.,atol=1e-14)


def test_negative_part_is_applied_to_each_corner_before_harm_shapley():
    cells=np.zeros((8,4));cells[0]=[-.02,.03,0.,0.];cells[7]=[.01,-.01,0.,0.]
    damage=module().shapley3(np.maximum(-cells,0.))
    expected=np.maximum(-cells[7],0.)-np.maximum(-cells[0],0.)
    assert np.allclose(damage.sum(0),expected,atol=1e-14)
    assert np.any(damage<0.)
    assert not np.allclose(damage,np.maximum(-module().shapley3(cells),0.))


def test_factual_steps_telescope_to_final_probability_and_small_paired_gap(rows):
    signs=np.array([1,1,-1,-1]); g_s=np.array([.03,-.02,.04,-.01]);g_b=np.array([.04,-.01,.02,-.03])
    w_s=rows.copy();w_b=np.roll(rows,1,axis=1)
    h_s=np.zeros(6);h_b=np.zeros(6);s_total=np.zeros(4);b_total=np.zeros(4);bridge_total=np.zeros(4)
    for t in range(4):
        cells=module().bridge_cells(rows@h_s,rows@h_b,g_s[t],g_b[t],rows@w_s[t],rows@w_b[t],signs)
        s_q,_=module().probability_effects(rows,h_s,g_s[t],w_s[t],signs)
        b_q,_=module().probability_effects(rows,h_b,g_b[t],w_b[t],signs)
        assert np.allclose(cells[0],s_q,atol=1e-14) and np.allclose(cells[7],b_q,atol=1e-14)
        s_total+=s_q;b_total+=b_q;bridge_total+=module().shapley3(cells).sum(0)
        h_s+=g_s[t]*w_s[t];h_b+=g_b[t]*w_b[t]
    assert np.allclose(s_total,signs*(sigmoid(rows@h_s)-.5),atol=1e-14)
    assert np.allclose(b_total,signs*(sigmoid(rows@h_b)-.5),atol=1e-14)
    assert np.allclose(bridge_total,b_total-s_total,atol=1e-14)


@pytest.mark.parametrize("key,value", [("model_seeds", [17501]), ("training_trials", 800),
    ("numeric_tolerance", .1), ("bridge_eligibility", "all_trials"),
    ("phase_definition", "known_after_count_equals_four"),
    ("new_confirmatory_hypothesis", True), ("baseline_commit", "0"*40),
    ("methods", ["balanced"]), ("bridge_bits", ["direction","scalar","state"]),
    ("objective_eligibility", "any_Go")])
def test_descriptive_config_cannot_change_scope_or_select_favorable_records(key,value):
    cfg=deepcopy(module().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):module().validate_config(cfg)


def saved_fixture(pattern="mixed"):
    """Hand-written eight-step path; no graph, RNG, actor or cone call."""
    from test_balanced_pathways_verify import invented_parent
    parent=invented_parent()
    cfg=deepcopy(module().SCIENCE)
    cfg.update(scope="integration_test",model_seeds=[90087],training_trials=8)
    shape=parent['trial_actions'].shape
    parent['trial_cumulative_head_path_l2']=np.zeros(shape)
    signs=np.array([1,1,-1,-1])
    for ti in range(3):
        for ri in range(2):
            for mi in range(4):
                head=np.zeros(4);known=np.zeros(4,np.int8);path=0.
                for t,source in enumerate(parent['schedules'][0]):
                    at=(ti,ri,mi,0,t)
                    action=1
                    if mi in (2,3) and pattern=="mixed":
                        action=int(t not in ([0,2] if mi==2 else [0,1]))
                    if pattern=="selective_never_Go" and mi==2:action=0
                    before=head.copy();known_before=known.copy()
                    p=float(sigmoid(head[source]));reward=int(signs[source]) if action else 0
                    direction=np.eye(4)[source]*(.7 if mi==2 else 1.)
                    if mi==3 and action==0:direction[:]=0.
                    factor=.1*(action-p)*reward
                    head+=factor*direction
                    if action:known[source]=reward
                    path+=float(np.linalg.norm(factor*direction))
                    parent['trial_actions'][at]=action;parent['trial_rewards'][at]=reward
                    parent['trial_probabilities'][at]=p;parent['trial_update_factors'][at]=factor
                    parent['trial_directions'][at]=direction;parent['trial_actual_influence'][at]=direction
                    parent['trial_correct_probability_delta'][at]=signs*(sigmoid(head)-sigmoid(before))
                    parent['trial_aligned_margin_delta'][at]=signs*(head-before)
                    parent['trial_known_before'][at]=known_before;parent['trial_known_after'][at]=known
                    other=np.arange(4)!=source
                    parent['trial_unknown_other_mask_before'][at]=(known_before==0)&other
                    parent['trial_unknown_other_mask_after'][at]=(known==0)&other
                    parent['trial_cumulative_head_path_l2'][at]=path
                    if t+1 in (4,8):
                        ki=(4,8).index(t+1)+1
                        ix=(ti,ri,mi,0,ki)
                        parent['checkpoint_heads'][ix]=head
                        parent['checkpoint_logits'][ix]=head
                        parent['checkpoint_probabilities'][ix]=sigmoid(head)
    return cfg,parent


def test_invented_analysis_has_exact_partitions_and_discovery_trial_remains_partial():
    cfg,parent=saved_fixture();arrays,rows=module().analyze(cfg,parent)
    assert len(arrays)==19 and len(rows)==144
    assert arrays['pair_action_cases'][0,0,0].tolist()==[0,1,2,3,3,3,3,3]
    assert arrays['first_full_trials'][0,0,2,0]==7
    assert arrays['first_full_trials'][0,0,3,0]==6
    assert arrays['phase_before'][0,0,2,0].tolist()==[0,0,0,0,0,0,0,1]
    assert arrays['phase_before'][0,0,3,0].tolist()==[0,0,0,0,0,0,1,1]
    assert arrays['objective_eligible'][0,0,0].tolist()==[False]*7+[True]
    category=arrays['effect_categories'];full=arrays['phase_before']==1
    assert not np.any((category==2)&full[...,None])
    assert sum(row['exposures'] for row in rows)==3*2*4*8*4
    for row in rows:
        assert row['probability_positive_mass']-row['probability_negative_mass']==pytest.approx(row['probability_signed_sum'])


def test_nonpaired_go_cases_keep_factual_difference_but_do_not_get_crossed_cells():
    cfg,parent=saved_fixture();arrays,rows=module().analyze(cfg,parent)
    eligible=arrays['bridge_eligible']
    assert np.array_equal(eligible,arrays['pair_action_cases']==3)
    assert not arrays['bridge_q_cells'][~eligible].any()
    assert not arrays['bridge_q_shapley'][~eligible].any()
    assert not arrays['bridge_harm_shapley'][~eligible].any()
    summary=module().summarize_arrays(cfg,arrays,rows,parent)
    seed=summary['contrasts'][0]['seeds'][0]
    assert seed['action_case_counts']==[1,1,1,5]
    assert seed['action_case_endpoint_difference'][1]!=0.
    assert seed['action_case_endpoint_difference'][2]!=0.
    assert sum(seed['action_case_endpoint_difference'])==pytest.approx(seed['endpoint_difference'])
    assert sum(seed['both_Go_probability_shapley'])==pytest.approx(seed['both_Go_probability_difference'])
    assert sum(seed['unknown_membership_damage_difference'])==pytest.approx(seed['unknown_damage_difference'])


def test_never_full_or_never_both_go_has_explicit_sentinel_padding_and_null_objective():
    cfg,parent=saved_fixture("selective_never_Go");arrays,rows=module().analyze(cfg,parent)
    assert np.all(arrays['first_full_trials'][:,:,2]==9)
    assert not arrays['bridge_eligible'].any() and not arrays['objective_eligible'].any()
    assert not arrays['bridge_q_cells'].any()
    assert not arrays['objective_values'].any()
    summary=module().summarize_arrays(cfg,arrays,rows,parent)
    seed=summary['contrasts'][0]['seeds'][0]
    assert seed['objective_gain_mean'] is None
    assert seed['objective_matched_probability_gain_mean'] is None
    assert seed['full_known_objective_trials']==0
    assert summary['counts']['bridge_evaluated_corner_destination_effects']==0


def test_analysis_uses_no_random_generator_or_new_optimizer(monkeypatch):
    cfg,parent=saved_fixture()
    def forbidden(*args,**kwargs):raise AssertionError("Saved-path arithmetic must not create random or optimized directions")
    monkeypatch.setattr(np.random,"default_rng",forbidden)
    monkeypatch.setattr(np.linalg,"svd",forbidden)
    arrays,rows=module().analyze(cfg,parent)
    summary=module().summarize_arrays(cfg,arrays,rows,parent)
    for name in ['new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls',
            'new_random_draws','fresh_model_constructions','new_optimization_calls']:
        assert summary['counts'][name]==0
    assert summary['validation']['all_exact_accounting_gates_passed']
    assert all(value<=1e-8 for value in summary['validation']['residuals'].values())


@pytest.mark.parametrize("mutation", ["wrong_rows", "mismatched_direction", "mismatched_scalar", "invalid_sign",
    "nan_head", "infinite_factor", "complex_direction", "string_rows"])
def test_factual_effects_reject_bad_axes_or_nonfinite_real_inputs(rows,mutation):
    values=[rows.copy(),np.zeros(6),np.asarray(.04),np.ones(6),np.array([1,1,-1,-1])]
    if mutation=="wrong_rows":values[0]=values[0][:3]
    elif mutation=="mismatched_direction":values[3]=values[3][:5]
    elif mutation=="mismatched_scalar":values[2]=np.array([.04])
    elif mutation=="invalid_sign":values[4][2]=0
    elif mutation=="nan_head":values[1][0]=np.nan
    elif mutation=="infinite_factor":values[2]=np.asarray(np.inf)
    elif mutation=="complex_direction":values[3]=values[3].astype(complex)
    else:values[0]=values[0].astype(str)
    with pytest.raises(ValueError):module().probability_effects(*values)


@pytest.mark.parametrize("mutation", ["source_shape", "direction_shape", "scalar_shape", "invalid_sign",
    "nan_state", "infinite_direction", "complex_state", "string_signs"])
def test_bridge_rejects_nonfinite_axes_and_invalid_target_signs(bridge_arguments,mutation):
    values=[value.copy() if isinstance(value,np.ndarray) else np.asarray(value) for value in bridge_arguments]
    if mutation=="source_shape":values[1]=values[1][:3]
    elif mutation=="direction_shape":values[5]=values[5][:3]
    elif mutation=="scalar_shape":values[2]=np.array([.04])
    elif mutation=="invalid_sign":values[6][2]=0
    elif mutation=="nan_state":values[0][0]=np.nan
    elif mutation=="infinite_direction":values[5][0]=np.inf
    elif mutation=="complex_state":values[1]=values[1].astype(complex)
    else:values[6]=values[6].astype(str)
    with pytest.raises(ValueError):module().bridge_cells(*values)


@pytest.mark.parametrize("values", [np.zeros(8),np.zeros((7,4)),np.full((8,4),np.nan),
    np.full((8,4),np.inf),np.zeros((8,4),complex),np.full((8,4),'0')])
def test_shapley_rejects_incomplete_or_nonfinite_cells(values):
    with pytest.raises(ValueError):module().shapley3(values)


def test_integer_valued_corner_table_is_safe_float_arithmetic():
    cells=np.zeros((8,4),dtype=int);cells[7]=[3,6,-9,12]
    result=module().shapley3(cells)
    assert np.allclose(result,np.broadcast_to(cells[7]/3.,(3,4)),atol=1e-14)


@pytest.mark.parametrize("known,sources", [(np.asarray(0),np.asarray(0)),
    (np.zeros(3,dtype=int),np.asarray(0)),(np.array([1,2,0,0]),np.asarray(0)),
    (np.array([1.,0.,0.,0.]),np.asarray(0)),(np.zeros(4,dtype=int),np.asarray(1.)),
    (np.zeros(4,dtype=int),np.asarray(4)),(np.zeros((2,4),dtype=int),np.asarray(1))])
def test_categories_reject_invalid_memory_or_noninteger_source_axes(known,sources):
    with pytest.raises(ValueError):module().decompose_categories(known,sources)
