"""Invented-only autonomous replay and objective-learning verification."""
import copy,importlib.util,json
from pathlib import Path
import numpy as np
import pytest

R=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('independent_objective_tests',R/'scripts/verify_objective_learning.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


@pytest.mark.parametrize('method,late',[('gated_uniform','uniform'),('gated_probability','probability')])
def test_fourth_discovery_is_still_isolated_and_next_full_before_can_use_objective(method,late):
    partial=np.array([1,1,-1,0],np.int8);full=np.array([1,1,-1,-1],np.int8)
    assert audit.independent_gate(partial,method)['mode']=='isolated'
    assert audit.independent_gate(full,method)['mode']==late


@pytest.mark.parametrize('kind',['uniform','probability'])
@pytest.mark.parametrize('source',range(4))
def test_four_input_objective_uses_signed_own_observations_and_state_only(kind,source):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.2,-.3,-.4])
    problem=audit.independent_objective_problem(d,source,k,z,kind);a=k[source]*k[:,None]*d
    p=audit.sigmoid(z);expected=a.sum(0)/4. if kind=='uniform' else (p*(1-p))@a/4.
    np.testing.assert_allclose(problem['constraints'],a,atol=1e-14)
    np.testing.assert_allclose(problem['objective'],expected,atol=1e-14)


@pytest.mark.parametrize('kind',['uniform','probability'])
@pytest.mark.parametrize('bad',[np.zeros(4,np.int8),np.array([1,1,-1,0],np.int8),np.array([1.,1.,-1.,-1.])])
def test_objectives_cannot_silently_fill_partial_or_cast_float_memory(kind,bad):
    with pytest.raises(ValueError):audit.independent_objective_problem(np.eye(4),0,bad,np.zeros(4),kind)


@pytest.mark.parametrize('boundary',[-41.,-40.,40.,41.])
def test_probability_gradient_fails_closed_at_clipping_boundaries(boundary):
    z=np.array([boundary,0.,0.,0.]);k=np.array([1,1,-1,-1],np.int8)
    with pytest.raises(ValueError):audit.independent_objective_problem(np.eye(4),0,k,z,'probability')
    audit.independent_objective_problem(np.eye(4),0,k,z,'uniform')


def test_current_state_probability_objective_changes_but_uniform_is_cached_by_source():
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.2,-.3,-.4]);other=z*2
    u=audit.independent_objective_problem(d,0,k,z,'uniform')['objective']
    p=audit.independent_objective_problem(d,0,k,z,'probability')['objective']
    np.testing.assert_array_equal(u,audit.independent_objective_problem(d,0,k,other,'uniform')['objective'])
    assert not np.array_equal(p,audit.independent_objective_problem(d,0,k,other,'probability')['objective'])


@pytest.mark.parametrize('kind',['uniform','probability'])
def test_closed_form_positive_orthant_support_certifies_actual_unit_objective(kind):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.7,-1.,-.4])
    c=audit.independent_objective_problem(d,2,k,z,kind)['objective'];w=c/np.linalg.norm(c)
    record=audit.certify_objective(d,2,k,z,kind,w,c,np.zeros(4))
    assert record['optimum']==pytest.approx(np.linalg.norm(c)) and np.max(record['certificate'][4:])<1e-14


@pytest.mark.parametrize('tamper',['direction','projection','multipliers'])
def test_direction_projection_and_dual_witness_must_certify_actual_current_objective(tamper):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.7,-1.,-.4])
    c=audit.independent_objective_problem(d,2,k,z,'probability')['objective'];w=c/np.linalg.norm(c);v=c.copy();mu=np.zeros(4)
    if tamper=='direction':w[0]+=.01
    elif tamper=='projection':v[0]+=.01
    else:mu[0]=-1e-16
    with pytest.raises(ValueError):audit.certify_objective(d,2,k,z,'probability',w,v,mu)


def fixture_config():
    from drosolearn.objective_learning import SCIENCE
    cfg=copy.deepcopy(SCIENCE);cfg.update(scope='integration_test',model_seeds=[90085,90086],training_trials=16,
        checkpoints=[0,4,8,16],population_pass_count=2,positive_seed_count=2,protection_nonworse_seed_count=2,
        state_positive_seed_count=2)
    return cfg


def tiny_graph():
    n=16
    return {'pre':np.arange(n),'post':np.arange(n),'counts':np.ones(n),'n_kc':n,'n_mbon':n}


@pytest.fixture(scope='module')
def producer_fixture(tmp_path_factory):
    from drosolearn import objective_learning as producer
    from drosolearn.graph import Projection
    cfg,raw=fixture_config(),tiny_graph();root=tmp_path_factory.mktemp('invented_objective_replay');n=raw['n_kc']
    graph=Projection(raw['pre'],raw['post'],raw['counts'],n,n,[str(i) for i in range(n)],[str(i) for i in range(n)],{})
    catalog=root/audit.INPUT_CATALOG;catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'study_id':'representation_update_v1','representations':list(audit.REPRESENTATIONS),
        'catalog':{key:value.tolist() for key,value in audit.prior.independent_catalog_arrays().items()}}),encoding='utf8')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer,'load_projection',lambda _:graph)
        arrays,rows=producer.produce(cfg,root)
    expected,wanted_rows=audit.expected_artifacts(cfg,raw,arrays)
    return cfg,raw,arrays,rows,expected,wanted_rows


def test_invented_eight_policy_actions_rewards_states_directions_and_objective_proofs_replay(producer_fixture):
    cfg,_,arrays,rows,expected,wanted_rows=producer_fixture
    audit.validate_science(cfg,allow_fixture=True);audit.check_artifacts(arrays,expected,rows,wanted_rows)
    assert len(arrays)==215 and len(rows)==1536 and len(rows[0])==96
    assert arrays['trial_actions'].shape==(3,2,8,2,16)
    assert arrays['objective_case_keys'].shape[1]==10
    assert all(audit.information_checks(cfg,arrays).values()) and all(audit.gate_checks(cfg,arrays).values())


def test_all_eight_policy_families_and_four_hypothesis_groups_match_independent_summary(producer_fixture):
    from drosolearn.objective_learning import summarize
    cfg,graph,a,rows,_,_=producer_fixture;summary=summarize(cfg,a,rows)
    checks=audit.verify_against_graph(cfg,graph,a,rows,summary,allow_fixture=True)
    assert checks['independently_replayed_training_actions']==1536 and checks['conditions']==48
    assert len(summary['families'])==48
    assert checks['unique_probability_states_independently_reoptimized']==summary['counts']['probability_direction_requests']
    assert checks['unique_uniform_states_independently_reoptimized']==summary['counts']['uniform_unique_solver_cases']


@pytest.mark.parametrize('field',['inputs','targets','schedules','uniforms','trial_actions','trial_rewards',
    'trial_probabilities','trial_known_before','trial_known_after','trial_routing_signs','trial_harm_mask',
    'trial_directions','trial_actual_influence','trial_desired_influence','trial_update_factors',
    'trial_direction_residuals','trial_direction_residuals_applicable','trial_case_indices','trial_direction_modes',
    'trial_gate_known_counts_before','trial_gate_full_before','trial_gate_switch_events',
    'trial_unknown_other_mask_before','trial_unknown_other_mask_after','trial_correct_probability_delta',
    'trial_aligned_margin_delta','trial_unknown_other_negative_correct_probability_sum','trial_memory_changed',
    'checkpoint_heads','checkpoint_known','checkpoint_probabilities','first_discovery_trials',
    'g_real_raw_unit_rows','g_real_raw_preflight_direction_count','w_real_raw_weights','finalfixed_ann_identity_probe_mbon',
    'trial_objective_applicable','trial_objective_logits_before','trial_objective_vectors','trial_objective_kind',
    'trial_objective_case_indices','first_uniform_mode_trials','first_probability_mode_trials'])
def test_every_new_field_and_frozen_control_state_is_bound_to_complete_replay(producer_fixture,field):
    _,_,original,rows,expected,wanted_rows=producer_fixture;a=copy.deepcopy(original)
    flat=a[field].reshape(-1);flat[-1]=not flat[-1] if flat.dtype.kind=='b' else flat[-1]+(1 if flat.dtype.kind in 'iu' else .03)
    with pytest.raises(ValueError):audit.check_artifacts(a,expected,rows,wanted_rows)


@pytest.mark.parametrize('field',['case_keys','case_directions','case_projections','case_multipliers','case_certificates',
    'objective_case_keys','objective_case_objectives','objective_case_logits','objective_case_directions',
    'objective_case_projections','objective_case_multipliers','objective_case_certificates'])
def test_actual_case_witnesses_must_reoptimize_the_actual_current_objective(producer_fixture,field):
    cfg,graph,original,_,_,_=producer_fixture;a=copy.deepcopy(original)
    a[field].reshape(-1)[-1]+=1 if a[field].dtype.kind in 'iu' else .1
    with pytest.raises(ValueError):audit.expected_artifacts(cfg,graph,a)


def test_uniform_source_cache_and_probability_trial_identity_are_not_interchangeable(producer_fixture):
    _,_,a,_,_,_=producer_fixture;keys=a['objective_case_keys']
    assert set(keys[:,3])=={6,7} and np.isin(keys[:,6:],[-1,1]).all()
    u=keys[keys[:,3]==6];p=keys[keys[:,3]==7]
    assert np.array_equal(u[:,4],u[:,5])
    assert len({tuple(key) for key in keys})==len(keys)
    assert len(p)==np.count_nonzero(a['trial_objective_applicable'][:,:,7])
    for si,ti,ri,mi,t,source,*known in p:
        at=(ti,ri,mi,si,t)
        assert a['trial_actions'][at]==1 and a['trial_gate_full_before'][at]
        assert a['schedules'][si,t]==source
        np.testing.assert_array_equal(a['trial_known_before'][at],known)


def test_trial_objective_z_is_each_learners_actual_head_prefix_not_first_cached_state(producer_fixture):
    cfg,_,a,_,_,_=producer_fixture
    for ti,top in enumerate(audit.TOPOLOGIES):
        for ri,rep in enumerate(audit.REPRESENTATIONS):
            for si in range(2):
                d=a['g_'+top+'_'+rep+'_unit_rows'][si]
                for mi in (6,7):
                    g=a['trial_update_factors'][ti,ri,mi,si];w=a['trial_directions'][ti,ri,mi,si]
                    heads=np.concatenate((np.zeros((1,16)),np.cumsum(g[:,None]*w,axis=0)))
                    active=a['trial_objective_applicable'][ti,ri,mi,si]
                    np.testing.assert_allclose(a['trial_objective_logits_before'][ti,ri,mi,si][active],(heads[:-1]@d.T)[active],atol=1e-12)


def test_partial_and_nogo_records_have_no_objective_or_solver_but_late_modes_remain(producer_fixture):
    _,_,a,_,_,_=producer_fixture
    active=a['trial_objective_applicable'];expected=np.zeros_like(active)
    expected[:,:,6:]=a['trial_gate_full_before'][:,:,6:]&(a['trial_actions'][:,:,6:]==1)
    assert np.array_equal(active,expected)
    for name in ['logits_before','vectors']:assert not a['trial_objective_'+name][~active].any()
    assert np.all(a['trial_objective_kind'][~active]==-1) and np.all(a['trial_objective_case_indices'][~active]==-1)
    assert np.all(a['trial_case_indices'][:,:,6:]==-1)
    for mi,mode in [(6,5),(7,6)]:
        no=a['trial_gate_full_before'][:,:,mi]&(a['trial_actions'][:,:,mi]==0)
        assert np.all(a['trial_direction_modes'][:,:,mi][no]==mode)
        assert not a['trial_directions'][:,:,mi][no].any()


@pytest.mark.parametrize('method,firstfield,mode',[(6,'first_uniform_mode_trials',5),(7,'first_probability_mode_trials',6)])
def test_new_methods_match_isolation_through_fourth_discovery_inclusive(producer_fixture,method,firstfield,mode):
    cfg,_,a,_,_,_=producer_fixture;n=cfg['training_trials']
    assert np.all(a[firstfield][:,:,:method]==n+1)
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                at=(ti,ri,method,si);discovery=int(a['first_discovery_trials'][at].max());end=min(n,discovery)
                for field in ['trial_actions','trial_rewards','trial_directions','trial_known_before','trial_known_after','trial_update_factors']:
                    assert a[field][at][:end].tobytes()==a[field][ti,ri,1,si][:end].tobytes()
                if discovery<=n:assert a['trial_direction_modes'][at+(discovery-1,)]==1
                if discovery<n:assert a['trial_direction_modes'][at+(discovery,)]==mode


@pytest.mark.parametrize('field,value',[('model_seeds',[17801]),('rng_namespace',202610203),('head_lr',.2),
    ('state_effect_threshold',0.),('state_positive_seed_count',1),('gain_controls',['isolated']),
    ('gate_switch_rule','known_after'),('training_trials',400),('probability_gradient_domain','clip_then_differentiate')])
def test_fixed_new_cohort_objective_and_state_hypothesis_choices_cannot_drift(producer_fixture,field,value):
    cfg=copy.deepcopy(producer_fixture[0]);cfg[field]=value
    with pytest.raises(ValueError):audit.validate_science(cfg,allow_fixture=True)


def test_new_verifier_does_not_import_the_old_or_new_head_or_scientific_producer():
    import ast
    tree=ast.parse((R/'scripts/verify_objective_learning.py').read_text(encoding='utf8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
        elif isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
    assert not any(name in imports for name in ['drosolearn.objective_learning','drosolearn.self_blend_learning','drosolearn.cone_geometry'])
    assert 'verify_self_blend_learning.py' not in [node.args[0].value for node in ast.walk(tree)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_load']


def isolated_endpoint_fixture(producer_fixture,monkeypatch):
    """Formal endpoint tests only; these altered paths never pass full replay."""
    cfg,_,original,rows,_,_=producer_fixture
    info=audit.information_checks(cfg,original);gates=audit.gate_checks(cfg,original)
    monkeypatch.setattr(audit,'information_checks',lambda *_:info)
    monkeypatch.setattr(audit,'gate_checks',lambda *_:gates)
    a=copy.deepcopy(original);a['checkpoint_probabilities'][:]=.5
    a['trial_unknown_other_negative_correct_probability_sum'][:]=0.
    return cfg,a,rows


def set_final(a,method,value,seed=slice(None)):
    a['checkpoint_probabilities'][0,0,method,seed,-1]=np.where(audit.TARGETS,value,1.-np.asarray(value))


@pytest.mark.parametrize('method',[6,7])
def test_gain_and_acquisition_are_separate_for_both_new_policies(producer_fixture,monkeypatch,method):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch);name=audit.METHODS[method]
    set_final(a,method,.7);s=audit.expected_summary(cfg,a,rows)
    assert s['H_gain'][name]['supported'] and not s['H_acquisition'][name]['supported']
    for control in (4,1,5):set_final(a,control,.90)
    set_final(a,method,.94);s=audit.expected_summary(cfg,a,rows)
    assert s['H_acquisition'][name]['supported'] and not s['H_gain'][name]['supported']


@pytest.mark.parametrize('method',[6,7])
@pytest.mark.parametrize('control',[4,1,5])
def test_each_policy_gain_requires_every_prespecified_comparator(producer_fixture,monkeypatch,method,control):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch)
    set_final(a,method,.7);set_final(a,control,.69)
    assert not audit.expected_summary(cfg,a,rows)['H_gain'][audit.METHODS[method]]['supported']


def test_state_effect_is_a_separate_prospective_P_minus_U_gate(producer_fixture,monkeypatch):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch)
    set_final(a,7,.7)
    for control in (4,1,5):set_final(a,control,.9)
    s=audit.expected_summary(cfg,a,rows)
    assert s['H_state']['supported'] and not s['H_gain']['gated_probability']['supported']
    assert not s['H_acquisition']['gated_probability']['supported']
    set_final(a,6,.90);set_final(a,7,.94)
    for control in (4,1,5):set_final(a,control,.5)
    s=audit.expected_summary(cfg,a,rows)
    assert not s['H_state']['supported'] and s['H_gain']['gated_probability']['supported']


def test_state_effect_requires_positive_seed_count_not_only_mean_advantage(producer_fixture,monkeypatch):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch)
    set_final(a,7,.7,seed=0);set_final(a,7,.5,seed=1)
    s=audit.expected_summary(cfg,a,rows)
    assert s['H_state']['mean_effect']>.05 and s['H_state']['positive_seeds']==1 and not s['H_state']['supported']


@pytest.mark.parametrize('method',[6,7])
def test_acquisition_requires_both_classes_and_training_gain(producer_fixture,monkeypatch,method):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch);name=audit.METHODS[method]
    # Mean .82 but one class .65 fails the prespecified class floor.
    a['checkpoint_probabilities'][0,0,method,:,-1]=[.99,.99,.35,.35]
    assert not audit.expected_summary(cfg,a,rows)['H_acquisition'][name]['supported']
    set_final(a,method,.9)
    a['checkpoint_probabilities'][0,0,method,:,0]=np.where(audit.TARGETS,.8,.2)
    assert not audit.expected_summary(cfg,a,rows)['H_acquisition'][name]['supported']


@pytest.mark.parametrize('method',[6,7])
def test_protection_requires_maximum_seed_harm_and_all_seed_nonworsening(producer_fixture,monkeypatch,method):
    cfg,a,rows=isolated_endpoint_fixture(producer_fixture,monkeypatch);name=audit.METHODS[method]
    assert audit.expected_summary(cfg,a,rows)['H_protection'][name]['supported']
    a['trial_unknown_other_negative_correct_probability_sum'][0,0,method,0,0]=1.1e-8
    s=audit.expected_summary(cfg,a,rows)['H_protection'][name]
    assert s['maximum_primary_seed_unknown_damage']>1e-8 and not s['supported']
    assert not s['comparisons']['isolated']['all_seeds_nonworsening']


@pytest.mark.parametrize('group',['H_gain','H_acquisition','H_protection','H_state','objective_validation','gate_validation','counts','resources','interpretation'])
def test_all_summary_hypotheses_controls_resource_counts_and_claims_are_bound(producer_fixture,group):
    cfg,_,a,rows,_,_=producer_fixture;expected=audit.expected_summary(cfg,a,rows);actual=copy.deepcopy(expected)
    if group in ['H_gain','H_acquisition','H_protection']:
        actual[group]['gated_probability']['supported']=not actual[group]['gated_probability']['supported']
    elif group=='H_state':actual[group]['mean_effect']+=.01
    elif group=='objective_validation':actual[group]['uniform_source_cache_probability_each_actual_trial']=False
    elif group=='gate_validation':actual[group]['first_probability_mode_follows_full_discovery']=False
    elif group=='counts':actual[group]['probability_unique_solver_cases']-=1
    elif group=='resources':actual[group]['processing_cost_equalized']=True
    else:actual[group]['generalization_tested']=True
    with pytest.raises(ValueError):audit.compare_tree(actual,expected)


@pytest.mark.parametrize('field',['method','trial','objective_applicable','objective_kind','objective_case_index',
    'objective_vector_l2','objective_certificate_residual_max','objective_logit_before_0'])
def test_objective_csv_columns_are_checked_and_keep_exact_order(producer_fixture,field):
    _,_,a,original,expected,wanted_rows=producer_fixture;rows=copy.deepcopy(original)
    value=rows[-1][field];rows[-1][field]='corrupt' if isinstance(value,str) else value+.1
    with pytest.raises(ValueError):audit.check_artifacts(a,expected,rows,wanted_rows)


def test_schema_does_not_accept_an_extra_unregistered_array(producer_fixture):
    _,_,original,rows,expected,wanted_rows=producer_fixture;a=copy.deepcopy(original);a['extra_objective_statistic']=np.zeros(1)
    with pytest.raises(ValueError):audit.check_artifacts(a,expected,rows,wanted_rows)


def test_lossless_gzip_csv_roundtrip_preserves_all_rows_columns_signs_and_numbers(producer_fixture,tmp_path):
    import csv,gzip,io
    _,_,a,rows,expected,wanted_rows=producer_fixture;stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=stream.getvalue().encode('utf8');require_crlf=b'\r\n' in raw
    assert require_crlf and len(rows)==1536 and len(rows[0])==96
    path=tmp_path/'measurements.csv.gz';path.write_bytes(gzip.compress(raw,mtime=0))
    assert gzip.decompress(path.read_bytes())==raw
    audit.check_artifacts(a,expected,audit.load_rows(path),wanted_rows)
    path.write_bytes(gzip.compress(b'trial,trial\r\n1,1\r\n',mtime=0))
    with pytest.raises(ValueError):audit.load_rows(path)


def test_four_trials_never_allow_post_full_objective_calls_and_preserve_empty_proof_inventory(tmp_path,monkeypatch):
    from drosolearn import objective_learning as producer
    from drosolearn.graph import Projection
    cfg=fixture_config();cfg.update(model_seeds=[90085],training_trials=4,checkpoints=[0,4],
        population_pass_count=1,positive_seed_count=1,protection_nonworse_seed_count=1,state_positive_seed_count=1)
    raw=tiny_graph();n=16
    graph=Projection(raw['pre'],raw['post'],raw['counts'],n,n,[str(i) for i in range(n)],[str(i) for i in range(n)],{})
    catalog=tmp_path/audit.INPUT_CATALOG;catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'study_id':'representation_update_v1','representations':list(audit.REPRESENTATIONS),
        'catalog':{key:value.tolist() for key,value in audit.prior.independent_catalog_arrays().items()}}),encoding='utf8')
    monkeypatch.setattr(producer,'load_projection',lambda _:graph)
    a,rows=producer.produce(cfg,tmp_path);summary=producer.summarize(cfg,a,rows)
    assert a['objective_case_keys'].shape==(0,10) and not a['trial_objective_applicable'].any()
    def forbidden(*args,**kwargs):raise AssertionError('No full-before trial can request a new objective support')
    monkeypatch.setattr(audit,'certify_objective',forbidden)
    receipt=audit.verify_against_graph(cfg,raw,a,rows,summary,allow_fixture=True)
    assert receipt['unique_objective_states_independently_reoptimized']==0
    assert np.all(a['first_uniform_mode_trials']==5) and np.all(a['first_probability_mode_trials']==5)


@pytest.mark.parametrize('method',[6,7])
def test_gate_audit_rejects_switching_on_the_fourth_discovery_itself(producer_fixture,method):
    cfg,_,original,_,_,_=producer_fixture;a=copy.deepcopy(original)
    for ti in range(3):
        for ri in range(2):
            for si in range(2):
                at=(ti,ri,method,si);discovery=int(a['first_discovery_trials'][at].max())
                if discovery<=cfg['training_trials']:
                    a['trial_direction_modes'][at+(discovery-1,)]=method-1
                    flags=audit.gate_checks(cfg,a)
                    assert not flags['fourth_discovery_update_is_isolated'] and not flags['mode_uses_known_before_only']
                    return
    raise AssertionError('Invented fixture must contain an actual fourth discovery')
