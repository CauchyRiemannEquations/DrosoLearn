"""Independent algebra and invented-only saved-step alignment verification."""
import copy,importlib.util
from pathlib import Path
import numpy as np
import pytest

R=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('probability_alignment_independent_tests',R/'scripts/verify_probability_alignment.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


def test_uniform_and_probability_objectives_use_all_four_obtained_signs():
    d=np.eye(4);known=np.array([1,1,-1,-1],np.int8);p=np.array([.5,.7,.2,.9])
    result=audit.independent_objectives(d,2,known,p)
    a=known[2]*known[:,None]*d
    np.testing.assert_array_equal(result['constraints'],a)
    np.testing.assert_allclose(result['uniform'],a.sum(0)/4)
    np.testing.assert_allclose(result['probability'],(p*(1-p))@a/4)
    np.testing.assert_array_equal(result['balanced'],d[2])
    assert not result['helpful_indices'].size


def test_legacy_helpful_subset_is_distinct_from_uniform_target_alignment():
    d=np.array([[1.,0.],[.6,.8],[-.6,.8],[0.,1.]])
    k=np.array([1,1,-1,-1],np.int8)
    result=audit.independent_objectives(d,0,k,np.full(4,.5))
    assert result['helpful_indices'].tolist()==[1,2]
    np.testing.assert_allclose(result['balanced'],.5*d[0]+.5*np.mean((k[0]*k[:,None]*d)[[1,2]],0))
    assert not np.allclose(result['balanced'],result['uniform'])


def test_probability_alignment_has_no_state_effect_at_equal_probability_slopes():
    result=audit.independent_objectives(np.eye(4),0,np.array([1,1,-1,-1]),np.array([.2,.8,.2,.8]))
    np.testing.assert_allclose(result['probability'],.16*result['uniform'],atol=1e-16)


@pytest.mark.parametrize('memory',[np.array([1,1,-1,0]),np.ones(4,float),np.ones(3,int),np.array([1,1,-1,2])])
def test_unobtained_or_malformed_signs_cannot_be_completed_for_the_objective(memory):
    with pytest.raises(ValueError):audit.independent_objectives(np.eye(4),0,memory,np.full(4,.5))


@pytest.mark.parametrize('p',[np.array([.5,.5,.5,np.nan]),np.full(4,1.1),np.ones(3),np.ones(4,complex)])
def test_probability_weights_require_finite_current_probabilities(p):
    with pytest.raises(ValueError):audit.independent_objectives(np.eye(4),0,np.array([1,1,-1,-1]),p)


def test_independent_dual_certificate_matches_a_closed_form_positive_orthant_optimum():
    a=np.eye(4);c=np.array([.1,.3,.2,.4]);direction=c/np.linalg.norm(c)
    cert=audit.certify_case(c,a,direction,c,np.zeros(4))
    np.testing.assert_allclose(cert[:3],np.linalg.norm(c),atol=1e-14)
    assert cert[3]==pytest.approx(1.) and np.max(cert[4:])<1e-14


def test_a_zero_objective_has_a_zero_finite_certificate_without_normalization():
    cert=audit.certify_case(np.zeros(4),np.eye(4),np.zeros(4),np.zeros(4),np.zeros(4))
    np.testing.assert_array_equal(cert,np.zeros(8))


def test_exact_negative_dual_multiplier_is_rejected_even_when_tiny():
    c=np.array([1.,0.,0.,0.]);mu=np.array([0.,-1e-16,0.,0.])
    with pytest.raises(ValueError):audit.certify_case(c,np.eye(4),c,c,mu)


def test_nonoptimal_feasible_direction_is_not_certified_by_its_own_objective_value():
    c=np.ones(4)
    with pytest.raises(ValueError):audit.certify_case(c,np.eye(4),np.array([1.,0.,0.,0.]),c,np.zeros(4))


def test_same_state_exact_effect_uses_the_saved_negative_scalar_and_four_output_probabilities():
    z=np.array([.2,-.4,.5,-.6]);d=np.eye(4);y=np.array([1,1,-1,-1]);g=-.04
    directions=np.array([[0.,0.,1.,0.],[0.,0.,.8,.6]])
    actual=audit.same_state_effects(d,z,g,directions,y)
    np.testing.assert_allclose(actual['probability_changes'],y*(audit.sigmoid(z+g*directions)-audit.sigmoid(z)))
    np.testing.assert_allclose(actual['margin_changes'],y*g*directions)
    np.testing.assert_allclose(actual['step_lengths'],[.04,.04])
    assert actual['probability_changes'][0,2]>0


def test_tangent_optimum_is_not_defined_as_a_finite_probability_success():
    # A valid mathematical counterexample: saturation and a finite positive
    # step can reverse the order of two first-order values.
    z=np.array([.02,-.021,0.,0.]);d=np.eye(4);y=np.ones(4)
    w=np.eye(4)[:2]
    effect=audit.same_state_effects(d,z,.1,w,y)
    tangent=.1*(audit.sigmoid(z)*(1-audit.sigmoid(z)))@w.T
    assert tangent[0]>tangent[1]
    assert effect['probability_changes'][0].sum()<effect['probability_changes'][1].sum()
    assert np.allclose(np.linalg.norm(w,axis=1),1.)


def test_global_probability_tangent_optimum_can_have_lower_finite_q_than_uniform_control():
    # Invented mathematically valid full-rank unit rows, not a recorded learner.
    eps=.01;a=np.array([[1.,0.,0.,0.],[np.cos(eps),np.sin(eps),0.,0.],
        [np.cos(eps),0.,np.sin(eps),0.],[0.,0.,0.,1.]])
    k=np.array([1,1,-1,-1],np.int8);d=k[:,None]*a;z=k*np.array([.98,.98,.98,1.])
    objectives=audit.independent_objectives(d,0,k,audit.sigmoid(z));u=objectives['uniform'];p=objectives['probability']
    U=u/np.linalg.norm(u);P=p/np.linalg.norm(p)
    audit.certify_case(u,a,U,u,np.zeros(4));audit.certify_case(p,a,P,p,np.zeros(4))
    g=.1*(1-audit.sigmoid(z[0]));effect=audit.same_state_effects(d,z,g,np.stack([U,P]),k)
    assert g*(p@P)>g*(p@U)
    difference=np.mean(effect['probability_changes'][1]-effect['probability_changes'][0])
    assert difference < -1e-8


def test_exact_zero_saved_scalar_keeps_all_candidate_finite_and_tangent_effects_zero():
    d=np.eye(4);k=np.array([1,1,-1,-1]);z=np.array([38.,.2,-.5,-.2]);w=np.stack([d[0],k/2.])
    effect=audit.same_state_effects(d,z,0.,w,k)
    assert not any(np.any(effect[key]) for key in ('probability_changes','margin_changes','step_lengths'))
    assert np.all(np.isfinite(effect['influences']))


def test_independent_integration_preserves_actual_go_with_rounded_zero_scalar():
    from test_probability_alignment import invented_parent,study
    cfg,parent=invented_parent();y=np.array([1,1,-1,-1])
    # Numerical edge only: the large invented early direction cannot occur in
    # the original step-capped learning cohort. It exercises g=0 eligibility.
    for ti in range(3):
        for ri in range(2):
            old=(ti,ri,4,0);head=np.zeros(4)
            for t,source in enumerate(parent['schedules'][0]):
                at=old+(t,);before=head.copy();action=parent['trial_actions'][at]
                direction=parent['trial_directions'][at].copy()
                if t==0:direction=np.array([760.,0.,0.,0.])
                probability=audit.sigmoid(head[source]);reward=parent['trial_rewards'][at]
                g=.1*(action-probability)*reward;head+=g*direction
                parent['trial_directions'][at]=direction;parent['trial_update_factors'][at]=g
                parent['trial_correct_probability_delta'][at]=y*(audit.sigmoid(head)-audit.sigmoid(before))
                if t+1 in (4,8,12):parent['checkpoint_heads'][old+((4,8,12).index(t+1)+1,)]=head
    producer=study();a,rows=producer.analyze(cfg,parent);summary=producer.summarize_arrays(cfg,a,rows,parent)
    checks=audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)
    assert a['eligible'][0,0,0,4] and a['factual_update_factors'][0,0,0,4]==0.
    assert not a['candidate_q_changes'][0,0,0,4].any()
    assert checks['eligible_actual_Go_records_checked']==36 and checks['independent_ambient_support_reoptimizations']==60


def fixture(never_full=False):
    # Only the test process imports the new producer, on explicit invented paths.
    from test_probability_alignment import invented_parent,study
    cfg,parent=invented_parent(never_full);producer=study()
    arrays,rows=producer.analyze(cfg,parent)
    return cfg,parent,arrays,rows,producer.summarize_arrays(cfg,arrays,rows,parent)


@pytest.fixture(scope='module')
def invented():return fixture()


def test_full_synthetic_integration_independently_checks_all_arrays_rows_summary_and_solver_counts(invented):
    cfg,parent,a,rows,summary=invented
    checks=audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)
    assert checks['numeric_arrays_checked']==27 and checks['measurement_records_checked']==90
    assert checks['eligible_actual_Go_records_checked']==36
    assert checks['candidate_destination_effects_checked']==720
    assert checks['independent_ambient_support_reoptimizations']==60
    assert checks['scientific_producer_imported'] is False and checks['new_cone_solver_imported'] is False


def test_empty_eligible_subset_has_no_optimization_null_means_and_complete_ordered_zero_rows():
    cfg,parent,a,rows,summary=fixture(True)
    checks=audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)
    assert checks['eligible_actual_Go_records_checked']==0 and checks['independent_ambient_support_reoptimizations']==0
    assert summary['finite_alignment_summary']['mean_exact_P_minus_U_per_update'] is None
    assert all(row['exact_Q_mean'] is None and row['eligible_trials']==0 for row in rows)


@pytest.mark.parametrize('field',audit.FIELDS)
def test_every_saved_array_is_independently_rebuilt_or_its_global_certificate_checked(invented,field):
    cfg,parent,original,rows,summary=invented;a=copy.deepcopy(original)
    if a[field].dtype.kind=='b':a[field].flat[0]=not a[field].flat[0]
    elif a[field].dtype.kind in 'iu':a[field].flat[0]+=1
    else:a[field].flat[0]+=.01
    with pytest.raises((ValueError,IndexError)):
        audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)


@pytest.mark.parametrize('field',['trial_known_before','trial_rewards','trial_update_factors','trial_directions',
    'trial_correct_probability_delta','checkpoint_heads','g_real_raw_isolated_directions','g_real_raw_unit_rows'])
def test_parent_memory_scalar_head_effect_and_geometries_are_checked(invented,field):
    cfg,original,a,rows,summary=invented;parent=copy.deepcopy(original)
    if field.startswith('g_'):parent[field].flat[0]+=.01
    elif field=='checkpoint_heads':parent[field][0,0,4,0,1,0]+=.01
    else:parent[field][0,0,4,0].flat[0]+=.01 if parent[field].dtype.kind=='f' else 1
    with pytest.raises(ValueError):audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)


@pytest.mark.parametrize('mutation',['extra','missing','order','identity','count','arithmetic','null'])
def test_complete_ordered_csv_and_exact_counts_are_bound(invented,mutation):
    cfg,parent,a,original,summary=invented;rows=copy.deepcopy(original)
    if mutation=='extra':rows.append(rows[-1])
    elif mutation=='missing':rows.pop()
    elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='identity':rows[0]['candidate']='state_probability'
    elif mutation=='count':rows[0]['eligible_trials']+=1
    elif mutation=='arithmetic':rows[0]['exact_Q_sum']+=.01
    else:rows[0]['exact_Q_mean']=None
    with pytest.raises((ValueError,TypeError)):audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)


@pytest.mark.parametrize('mutation',['support','finite_result','counts','seed_sign_count','claim','missing'])
def test_summary_signs_optimizer_work_failures_and_interpretation_are_independent(invented,mutation):
    cfg,parent,a,rows,original=invented;summary=copy.deepcopy(original)
    if mutation=='support':summary['validation']['residuals']['uniform_global_support']=.01
    elif mutation=='finite_result':summary['finite_alignment_summary']['mean_exact_P_minus_U_per_update']+=.01
    elif mutation=='counts':summary['counts']['new_optimization_calls']=0
    elif mutation=='seed_sign_count':summary['families'][-1]['seeds'][0]['positive_vs_U_trials']+=1
    elif mutation=='claim':summary['interpretation']['finite_probability_global_optimum_claimed']=True
    else:summary.pop('validation')
    with pytest.raises(ValueError):audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)


@pytest.mark.parametrize('field,value',[('reference_method_index',5),('numeric_tolerance',.01),('head_lr',.2),
    ('primary_comparison','P_minus_B'),('blend_balanced_weight',.5),('new_confirmatory_hypothesis',True),
    ('model_seeds',[17701]),('training_trials',400)])
def test_fixed_scientific_choices_and_fixture_seed_limits_are_independent(invented,field,value):
    cfg=copy.deepcopy(invented[0]);cfg[field]=value
    with pytest.raises(ValueError):audit.validate_science(cfg,allow_fixture=True)


def test_fixture_scope_is_never_silently_accepted_as_actual_science(invented):
    with pytest.raises(ValueError):audit.validate_science(invented[0])


def test_noneligible_records_do_not_request_independent_reoptimization(invented,monkeypatch):
    cfg,parent,a,rows,summary=fixture(True)
    def forbidden(*args,**kwargs):raise AssertionError('An excluded record cannot run a cone audit')
    monkeypatch.setattr(audit,'certify_case',forbidden)
    audit.verify_science(cfg,parent,a,rows,summary,allow_fixture=True)


def test_independent_runtime_does_not_import_the_new_scientific_producer_or_solver():
    import ast
    tree=ast.parse((R/'scripts/verify_probability_alignment.py').read_text(encoding='utf8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
        elif isinstance(node,ast.Import):imports.extend(alias.name for alias in node.names)
    assert 'drosolearn.probability_alignment' not in imports and 'drosolearn.cone_geometry' not in imports
    assert not any(name and 'verify_' in name for name in imports)


@pytest.mark.parametrize('empty',[False,True])
def test_csv_roundtrip_accepts_null_and_numeric_strings_but_not_duplicate_fields(invented,tmp_path,empty):
    import csv
    cfg,parent,a,rows,summary=fixture(True) if empty else invented;path=tmp_path/'rows.csv'
    with path.open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit.verify_science(cfg,parent,a,audit.load_rows(path),summary,allow_fixture=True)
    path.write_text('seed,seed\n1,1\n',encoding='utf8')
    with pytest.raises(ValueError):audit.load_rows(path)
