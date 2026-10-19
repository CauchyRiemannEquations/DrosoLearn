"""Independent contextual arithmetic; all executable examples are synthetic."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("independent_order", Path(__file__).resolve().parents[1]/"scripts/verify_interference_order.py")
audit = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(audit)


def real_config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/interference_order_v1.json").read_text(encoding="utf8"))


def fixture_config():
    cfg = real_config()
    cfg.update(scope="integration_test",model_seeds=[91001,91002],stage_trials=6,joint_trials=12,
               checkpoint_every=6,joint_checkpoint_every=12,population_pass_count=2,bootstrap_repeats=16,
               bootstrap_entropy=[202610192,99001,2],graph="synthetic-unused")
    return cfg


def small_graph():
    return {"pre":np.arange(6),"post":np.array([0,1,2,0,1,2]),"counts":np.ones(6),"n_kc":6,"n_mbon":3}


def write_catalog(root):
    path = root/audit.INPUT_CATALOG; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"study_id":"interference_order_v1","set_order":["A","B"],
                                "catalog":{k:v.tolist() for k,v in audit.catalog().items()}}),encoding="utf8")
    return path


def test_scientific_validation_never_uses_rng(monkeypatch):
    monkeypatch.setattr(np.random,"default_rng",lambda *args: (_ for _ in ()).throw(AssertionError("RNG")))
    audit.validate_science(real_config()); audit.validate_science(fixture_config(),True)


@pytest.mark.parametrize("key,value",[("model_seeds",[15201]),("stage_trials",720),("joint_trials",1440),
                                      ("conflict_threshold",.5),("head_lr",.3),("class_threshold",.6),
                                      ("bootstrap_repeats",10000),("order_margin",.01),("population_pass_count",1),
                                      ("joint_order","posthoc"),("idle","decay"),("normalization","all_after_training")])
def test_registered_scope_cannot_relax_or_extend(key,value):
    cfg=real_config(); cfg[key]=value
    with pytest.raises(ValueError): audit.validate_science(cfg)


@pytest.mark.parametrize("change",[{"model_seeds":[15201]},{"stage_trials":360},{"bootstrap_entropy":[202610192,15301,2]},
                                   {"model_seeds":[91001,91001]},{"population_pass_count":1}])
def test_fixtures_cannot_use_registered_science(change):
    cfg=fixture_config(); cfg.update(change)
    with pytest.raises(ValueError): audit.validate_science(cfg,True)


@pytest.mark.parametrize("args",[(True,0,0,0),(-1,0,0,0),(91001,2,0,0),(91001,0,2,0),(91001,0,0,2)])
def test_rng_tuple_domain(args):
    with pytest.raises(ValueError): audit.address(*args)


def test_rng_and_paired_curricula_match_declared_task_tapes():
    cfg=fixture_config(); plan=audit.curriculum(cfg,91001)
    np.testing.assert_array_equal(audit.address(91001,1,0,1).random(6),
                                  np.random.default_rng(np.random.SeedSequence([202610192,91001,1,0,1])).random(6))
    for schedule in plan["schedules"].reshape(-1,6): assert sorted(schedule)==list(range(6))
    for left,right in ((0,4),(1,2)):
        np.testing.assert_array_equal(plan["stream_stimulus_indices"][left,:6],plan["stream_stimulus_indices"][right,:6])
        np.testing.assert_array_equal(plan["stream_uniforms"][left,:6],plan["stream_uniforms"][right,:6])
    np.testing.assert_array_equal(plan["stream_stimulus_indices"][6,::2],plan["schedules"][0,0])
    np.testing.assert_array_equal(plan["stream_stimulus_indices"][6,1::2],plan["schedules"][1,0])
    assert plan["trial_valid"].sum()==48


def test_catalog_conflicts_are_supplied_context_not_missing_input(tmp_path):
    data=audit.catalog(); x,y=data["probe_inputs"],data["probe_targets"]
    assert x.shape==(2,6,7) and len(set(map(tuple,x.reshape(12,7))))==12
    np.testing.assert_array_equal(y[0]!=y[1],data["conflicting_ids"])
    path=write_catalog(tmp_path); audit.verify_input_catalog(fixture_config(),tmp_path)
    raw=json.loads(path.read_bytes()); raw["catalog"]["conflicting_ids"]=[1,1,0,0,0,0]
    path.write_text(json.dumps(raw),encoding="utf8")
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(),tmp_path)


def test_duplicate_or_nonfinite_catalog_json_rejected(tmp_path):
    path=write_catalog(tmp_path)
    path.write_text('{"study_id":"interference_order_v1","study_id":"interference_order_v1"}',encoding="utf8")
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(),tmp_path)
    path.write_text('{"value":NaN}',encoding="utf8")
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(),tmp_path)


def test_checkpoint_branches_reference_acquired_prefix_not_previous_branch():
    cp=audit.checkpoint_metadata(fixture_config())
    np.testing.assert_array_equal(cp["stream_start_checkpoint"],[0,0,1,1,2,2,0])
    np.testing.assert_array_equal(cp["stream_end_checkpoint"],[1,2,3,4,5,6,7])
    np.testing.assert_array_equal(cp["checkpoint_total_trial"],[0,6,6,12,12,12,12,12])


def test_context_q_has_twelve_disjoint_values_and_no_hidden_nogo_updates():
    probes=audit.catalog()["probe_inputs"].reshape(12,7)
    model=audit.make_independent_actor("context_q",91001,small_graph(),probes,{})
    assert model.values.shape==(12,2)
    model.update(probes[0],0,0); assert not np.any(model.values)
    model.update(probes[0],1,1)
    assert model.values[0,1]==pytest.approx(.1) and model.probability(probes[6])==.5
    with pytest.raises(ValueError): model.update(probes[0],0,-1)


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    # Producer use exists only on the test side of cross-implementation checks.
    import drosolearn.interference_order as producer
    from drosolearn.graph import Projection
    cfg=fixture_config(); root=tmp_path_factory.mktemp("contextual-fixture"); write_catalog(root)
    g=small_graph(); projection=Projection(g["pre"],g["post"],g["counts"],6,3,list(map(str,range(6))),list(map(str,range(3))),{"synthetic":True})
    original=producer.load_projection
    producer.load_projection=lambda unused: projection
    try: arrays,rows=producer.produce(cfg,root)
    finally: producer.load_projection=original
    return cfg,g,arrays,rows,producer.summarize(cfg,arrays,rows)


def test_all_synthetic_feedback_states_and_claims_independently_agree(produced):
    cfg,g,a,rows,summary=produced
    result=audit.verify_against_graph(cfg,g,a,rows,summary,True)
    assert result["independently_replayed_feedback_trials"]==864
    assert result["checkpoint_static_probes"]==1728


def test_csv_row_fields_and_padding_are_checked(produced):
    cfg,g,a,rows,summary=produced
    converted=[{k:str(v) for k,v in row.items()} for row in rows]
    audit.verify_against_graph(cfg,g,a,converted,summary,True)
    assert np.all(a["trial_actions"][:,:,~a["trial_valid"]]==-1)


@pytest.mark.parametrize("field",["trial_actions","trial_probabilities","trial_rewards","trial_head_steps",
                                   "trial_internal_call_increments","trial_head_update_calls","stream_start_head_calls",
                                   "checkpoint_logits","checkpoint_probabilities","schedules","stream_uniforms",
                                   "w_plastic_head","w_plastic_weights","w_context_q_value_table",
                                   "idle_w_plastic_weights","idle_probabilities"])
def test_corrupted_state_or_ledger_rejected(produced,field):
    cfg,g,original,rows,summary=produced; a=copy.deepcopy(original); a[field].flat[0]+=1
    with pytest.raises(ValueError): audit.verify_against_graph(cfg,g,a,rows,summary,True)


def test_corrupted_row_missing_array_or_false_gate_rejected(produced):
    cfg,g,a,rows,summary=produced
    changed=copy.deepcopy(rows); changed[0]["head_call_increment"]+=1
    with pytest.raises(ValueError): audit.verify_against_graph(cfg,g,a,changed,summary,True)
    missing=dict(a); missing.pop("idle_logits")
    with pytest.raises(ValueError): audit.verify_against_graph(cfg,g,missing,rows,summary,True)
    false=copy.deepcopy(summary); false["qualification_supported"]=not false["qualification_supported"]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg,g,a,rows,false,True)


def test_context_blind_five_sixths_is_not_qualified(produced):
    cfg,_,original,_,_=produced; a=copy.deepcopy(original)
    correct=np.ones(a["checkpoint_probabilities"].shape); correct[:,:,:,:,:2]=.5; correct[:,:,0]=.5
    a["checkpoint_probabilities"]=np.where(a["probe_targets"][None,None,None],correct,1-correct)
    summary=audit.expected_summary(cfg,a)
    assert summary["analytic_context_blind_witness"]["overall_accuracy"]==pytest.approx(5/6)
    assert summary["qualification_supported"] is False
    assert summary["H_interference"]["status"]==summary["H_order"]["status"]=="not_assessable"


def test_synthetic_qualified_loss_and_practice_and_order_are_distinct(produced):
    cfg,_,original,_,_=produced; a=copy.deepcopy(original); end=a["stream_end_checkpoint"]
    correct=np.full(a["checkpoint_probabilities"].shape,.9); correct[:,:,0]=.5
    correct[:,:,end[2],0]=.6; correct[:,:,end[4],1]=.8; correct[:,:,end[3],0]=.95
    a["checkpoint_probabilities"]=np.where(a["probe_targets"][None,None,None],correct,1-correct)
    summary=audit.expected_summary(cfg,a); first=summary["families"][0]["seeds"][0]
    assert summary["qualification_supported"] and summary["H_interference"]["supported"] and summary["H_order"]["supported"]
    assert first["tasks"][0]["retention_loss"]==pytest.approx(.3)
    assert first["tasks"][0]["continued_practice_gain"]==pytest.approx(.05)
    assert first["tasks"][0]["matched_budget_terminal_cost"]==pytest.approx(.35)
    assert first["order_delta"]==pytest.approx(-.1)
    assert summary["max_decomposition_residual"]<1e-12


def test_context_table_order_commutes_and_idle_has_no_new_updates(produced):
    cfg,_,a,_,summary=produced; end=a["stream_end_checkpoint"]
    np.testing.assert_array_equal(a["w_context_q_value_table"][:,end[2]],a["w_context_q_value_table"][:,end[4]])
    np.testing.assert_array_equal(a["w_context_q_value_table"][:,end[2]],a["w_context_q_value_table"][:,end[6]])
    for kind in ("head","internal","value"):
        np.testing.assert_array_equal(a[f"stream_start_{kind}_calls"],a[f"checkpoint_{kind}_calls"][:,:,a["stream_start_checkpoint"]])
    assert summary["counts"]["idle_new_probability_evaluations"]==summary["counts"]["idle_feedback_trials"]==0
    assert summary["counts"]["feedback_trials"]==864
    assert summary["counts"]["logical_active_history_trials"]==1080
