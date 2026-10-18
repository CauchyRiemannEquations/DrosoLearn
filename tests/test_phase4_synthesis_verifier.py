"""Handwritten evidence maps and corruption checks; no human raw data or producer imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_phase4_synthesis.py'
spec = importlib.util.spec_from_file_location('independent_phase4_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def hashed(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_counts(n):
    return {'denominator':n, 'stay':n//2, 'switch':n-n//2,
            'stay_rate':None if not n else (n//2)/n,
            'switch_rate':None if not n else (n-n//2)/n}


def handcrafted_streams():
    streams = []
    supports = {}
    for group, count in (('human',25), ('deterministic',80), ('probabilistic',80), ('uninformative',80)):
        is_human = group == 'human'
        support = dict(streams=count, strata=count*(8 if is_human else 4),
                       both_feedback=194 if is_human else 0 if group=='deterministic' else 320,
                       feedback_0_only=6 if is_human else 160 if group=='deterministic' else 0,
                       feedback_1_only=160 if group=='deterministic' else 0, neither=0,
                       feedback_0_denominator=0, feedback_1_denominator=0, eligible=0)
        for record in range(count):
            cells = ([{'phase':phase,'state':state,'choice':choice}
                      for phase in ('initial','transfer') for state in ('1','2') for choice in (0,1)]
                     if is_human else
                     [{'phase':phase,'state':state,'choice':choice}
                      for phase,state in (('phase1','larger'),('phase2','smaller')) for choice in (0,1)])
            target = (890 if record==24 else 860) if is_human else 798
            allowed = []
            for index, cell in enumerate(cells):
                feedbacks = ([0] if is_human and record==0 and index<6 else
                             [index%2] if group=='deterministic' else [0,1])
                allowed.extend((index, f) for f in feedbacks)
                cell['by_feedback'] = {'0':cell_counts(0),'1':cell_counts(0)}
            q, r = divmod(target, len(allowed))
            for position, (index, f) in enumerate(allowed):
                cells[index]['by_feedback'][str(f)] = cell_counts(q+(position<r))
            for index, cell in enumerate(cells):
                zero_only = (is_human and record==0 and index<6) or (group=='deterministic' and index%2==0)
                one_only = group=='deterministic' and index%2==1
                cell.update(common_support=not(zero_only or one_only),
                            support_status='feedback_0_only' if zero_only else 'feedback_1_only' if one_only else 'both_feedback')
            by_feedback = {f:{key:sum(c['by_feedback'][f][key] for c in cells)
                              for key in ('denominator','stay','switch')} for f in ('0','1')}
            for f in ('0','1'):
                support['feedback_'+f+'_denominator'] += by_feedback[f]['denominator']
            support['eligible'] += target
            streams.append(dict(stream_id='human/main/'+str(record) if is_human else 'model/'+str(record)+'/fixture/'+group,
                kind='human' if is_human else 'model', trials=1000 if is_human else 800,
                candidates=999 if is_human else 799, eligible=target,
                excluded=(999 if is_human else 799)-target,
                by_feedback=by_feedback, strata=cells))
        supports[group] = support
    return streams, supports


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    cfg = dict(study_id='phase4_synthesis_v1', scope=verifier.SCOPE,
               features=[dict(id=name, metric='fixture measurement', unit='fixture observation unit',
                    human=dict(honore='descriptive_only',markovic='source_coded_only',openneuro='semantic_gate_closed'),
                    model='fixture model', difference='fixture difference', unlock='fixture gate evidence')
                    for name in verifier.FEATURES],
               **{name:False for name in verifier.FLAGS}, source_files=list(verifier.SOURCES),
               evidence_files=list(verifier.EVIDENCE))
    for name in verifier.SOURCES:
        file = root/name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('handcrafted source fixture\n',encoding='utf8')
    (root/'scripts/verify_phase4_synthesis.py').write_bytes(SCRIPT.read_bytes())
    for name in verifier.EVIDENCE:
        dump(root/name, {'status':'passed'})
    dump(root/'configs/phase4_synthesis_v1.json', cfg)
    gate = dict(source_coded_descriptive_metric_allowed=True,
                exact_historical_task_execution_version_verified=False,
                direct_human_model_similarity_allowed=False)
    dump(root/'results/human_task_structure_v1/audit.json', {'gate':gate})
    origin = dict(status='passed', source_identity_gate_passed=True, mismatches_verified=0,
                  raw_rows_replayed=25, unmatched_rows_preserved=49, values_replayed=75000)
    dump(root/'results/human_osf_origin_v1/verification.json', origin)
    dump(root/'results/human_semantics_v1/verification.json', {'status':'passed','scope':verifier.SEMANTIC_SCOPE})
    schema = {'datasets':[{'sheets':[{'sheet':s,'participants':56} for s in ('T1','T2')]} for _ in range(2)]}
    dump(root/'data/human/honore_noel_2016/schema-audit.json', schema)
    streams, support = handcrafted_streams()
    metrics = dict(study_id='conditional_transition_v1', streams=streams,
                   pool_group_or_species_estimates=False,averaged_stratum_estimates=False,causal_effect_estimated=False)
    metric_file = root/'results/conditional_transition_v1/metrics.json'
    dump(metric_file,metrics)
    old_artifact = root/'results/conditional_transition_v1/artifact_hashes.json'
    dump(old_artifact, {'metrics.json':hashed(metric_file)})
    counts = dict(streams=265,input_trials=217000,candidates=216735,eligible=213050,excluded=3685,
                  strata=1160,feedback_cells=2320,human_records=25,model_streams=240)
    dump(root/'results/conditional_transition_v1/verification.json',
         {'status':'passed','counts':counts,'metrics_sha256':hashed(metric_file),'artifact_hashes_sha256':hashed(old_artifact)})
    for study, field, artifact in (
            ('human_task_structure_v1','result_sha256','audit.json'),
            ('stochastic_feedback_network_v1','summary_sha256','summary.json'),
            ('learning_ablation_v1','summary_sha256','summary.json'),
            ('time_mixture_v1','metrics_sha256','metrics.json')):
        dump(root/'results'/study/'verification.json',{'status':'passed',field:hashed(root/'results'/study/artifact)})
    for args in (['git','init','--quiet'], ['git','config','core.autocrlf','false'], ['git','add','.'],
                 ['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--quiet','-m','handwritten frozen fixture']):
        subprocess.run(args,cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    config = root/'configs/phase4_synthesis_v1.json'
    reg = dict(study_id=cfg['study_id'],registered_at_utc='2000-01-01T00:00:00+00:00',
               source_git_commit=commit,config_sha256=hashed(config),config=cfg,
               source_files={n:hashed(root/n) for n in verifier.SOURCES},
               evidence_files={n:hashed(root/n) for n in verifier.EVIDENCE})
    registration = root/'registration_fixture.json'
    dump(registration,reg)
    result = root/'result'
    result.mkdir()
    (result/'registration.json').write_bytes(registration.read_bytes())
    synthesis = dict(study_id=cfg['study_id'],bounded_phase4_deliverables_completed=True,
        **{k:False for k in verifier.FLAGS},direct_level4_verdict='untested_not_testable_with_current_pairing',
        feature_count=6,human_source_cells=18,features=cfg['features'],
        source_coded_descriptive_metric_allowed=True,historical_task_execution_version_verified=False,
        openneuro_semantic_mapping_verified=False,verified_original_human_records=25,
        unverified_original_human_records_retained=49,support_counts=support,support_is_only_denominator_presence=True,
        additional_learning_updates=0,rng_draws=0,new_human_participants=0,new_human_effect_estimates=0,
        scope='evidence map and integer recount of prior verified measurement; no new rate pooling, significance, similarity or biological claim')
    dump(result/'synthesis.json',synthesis)
    dump(result/'metadata.json',dict(study_id=cfg['study_id'],status='completed',
        source_git_commit=commit,registration_sha256=hashed(registration),config_sha256=hashed(config),
        started_at_utc='2000-01-01T00:00:01+00:00',completed_at_utc='2000-01-01T00:00:02+00:00'))
    fixture = dict(root=root,config=config,registration=registration,result_dir=result)
    refresh_artifacts(fixture)
    return fixture


def refresh_artifacts(fixture):
    result = fixture['result_dir']
    dump(result/'artifact_hashes.json',{n:hashed(result/n) for n in verifier.ARTIFACTS})


def mutate(file, operation):
    value = json.loads(file.read_text(encoding='utf8'))
    operation(value)
    dump(file,value)


def run(fixture):
    return verifier.verify(fixture['root'],fixture['config'],fixture['registration'],fixture['result_dir'])


def assert_failed(fixture):
    report = run(fixture)
    assert report['passed'] is False and report['status']=='failed' and report['errors']


def test_handwritten_full_scope_passes(frozen):
    report = run(frozen)
    assert report['passed'] and report['counts']['strata']==1160
    assert report['counts']['human_source_cells']==18


@pytest.mark.parametrize('flag',verifier.FLAGS)
def test_forbidden_claim_even_with_new_artifact_hash(frozen,flag):
    mutate(frozen['result_dir']/'synthesis.json',lambda v:v.__setitem__(flag,True))
    refresh_artifacts(frozen)
    assert_failed(frozen)


@pytest.mark.parametrize('key,value',[
    ('verified_original_human_records',74),('unverified_original_human_records_retained',0),
    ('openneuro_semantic_mapping_verified',True),('historical_task_execution_version_verified',True),
    ('direct_level4_verdict','confirmed'),('new_human_effect_estimates',1),
    ('additional_learning_updates',False),('support_is_only_denominator_presence',False),
    ('human_source_cells',17),('feature_count',5)])
def test_synthesis_contract(frozen,key,value):
    mutate(frozen['result_dir']/'synthesis.json',lambda v:v.__setitem__(key,value))
    refresh_artifacts(frozen)
    assert_failed(frozen)


def test_support_recount_is_independent_of_claimed_counts(frozen):
    def change(value):
        value['support_counts']['human']['both_feedback'] += 1
        value['support_counts']['human']['feedback_0_only'] -= 1
    mutate(frozen['result_dir']/'synthesis.json',change)
    refresh_artifacts(frozen)
    assert_failed(frozen)


def test_feature_cell_cannot_be_silently_rewritten(frozen):
    mutate(frozen['result_dir']/'synthesis.json',lambda v:v['features'][0]['human'].__setitem__('honore','direct_similarity'))
    refresh_artifacts(frozen)
    assert_failed(frozen)


@pytest.mark.parametrize('name', ['synthesis.json','metadata.json','registration.json'])
def test_artifact_corruption(frozen,name):
    with (frozen['result_dir']/name).open('ab') as stream:
        stream.write(b' ')
    assert_failed(frozen)


@pytest.mark.parametrize('name', ['source_files','evidence_files'])
def test_frozen_inventory_is_exact(frozen,name):
    mutate(frozen['registration'],lambda v:v[name].pop(next(iter(v[name]))))
    assert_failed(frozen)


@pytest.mark.parametrize('source', [verifier.SOURCES[1],verifier.EVIDENCE[0]])
def test_frozen_source_or_evidence_bytes(frozen,source):
    (frozen['root']/source).write_bytes(b'changed')
    assert_failed(frozen)


def test_refreshed_hash_does_not_replace_source_git_blob(frozen):
    name = verifier.SOURCES[1]
    (frozen['root']/name).write_bytes(b'new uncommitted source')
    mutate(frozen['registration'],lambda v:v['source_files'].__setitem__(name,hashed(frozen['root']/name)))
    (frozen['result_dir']/'registration.json').write_bytes(frozen['registration'].read_bytes())
    refresh_artifacts(frozen)
    assert_failed(frozen)


@pytest.mark.parametrize('stamp', ['1999-01-01T00:00:00+00:00','2999-01-01T00:00:00+00:00','2000-01-01T00:00:01'])
def test_timestamp_order_and_timezone(frozen,stamp):
    mutate(frozen['result_dir']/'metadata.json',lambda v:v.__setitem__('started_at_utc',stamp))
    refresh_artifacts(frozen)
    assert_failed(frozen)


@pytest.mark.parametrize('mutation',[
    lambda m:m['streams'].append(copy.deepcopy(m['streams'][0])),
    lambda m:m['streams'][0]['strata'].pop(),
    lambda m:m['streams'][0]['strata'].__setitem__(0,copy.deepcopy(m['streams'][0]['strata'][1])),
    lambda m:m['streams'][0]['strata'][0]['by_feedback']['0'].__setitem__('denominator',True),
    lambda m:m['streams'][0]['strata'][0]['by_feedback']['0'].__setitem__('stay',-1),
    lambda m:m['streams'][0]['strata'][0].__setitem__('support_status','both_feedback'),
    lambda m:m['streams'][0]['strata'][0].__setitem__('common_support',True),
    lambda m:m['streams'][0]['by_feedback']['0'].__setitem__('stay',0),
    lambda m:m['streams'][0].__setitem__('eligible',0),
    lambda m:m['streams'][0].__setitem__('excluded',0),
    lambda m:m['streams'][25].__setitem__('stream_id','model/0/fixture/unknown'),
    lambda m:m.__setitem__('averaged_stratum_estimates',True),
])
def test_saved_integer_partition_corruptions(mutation):
    streams,_ = handcrafted_streams()
    metrics = dict(study_id='conditional_transition_v1',streams=streams,
                   pool_group_or_species_estimates=False,averaged_stratum_estimates=False,causal_effect_estimated=False)
    prior = {'counts':dict(streams=265,input_trials=217000,candidates=216735,eligible=213050,
                         excluded=3685,strata=1160,feedback_cells=2320,human_records=25,model_streams=240)}
    mutation(metrics)
    with pytest.raises((ValueError,KeyError,TypeError)):
        verifier.inspect_support(metrics,prior)


@pytest.mark.parametrize('relative', ['../outside','/absolute','docs\\wrong.md',''])
def test_paths_cannot_escape(tmp_path,relative):
    with pytest.raises(ValueError):
        verifier.evidence_path(tmp_path,relative)


def test_duplicate_json_keys_and_nonfinite_constants(tmp_path):
    file = tmp_path/'bad.json'
    for raw in ('{"a":1,"a":2}','{"a":NaN}'):
        file.write_text(raw,encoding='utf8')
        with pytest.raises(ValueError):
            verifier.read(file)


def test_cli_refuses_existing_output(frozen):
    out = frozen['result_dir']/'verification.json'
    out.write_text('preserve this previous result',encoding='utf8')
    command = [sys.executable,str(SCRIPT),'--root',str(frozen['root']),'--config',str(frozen['config']),
               '--registration',str(frozen['registration']),'--result-dir',str(frozen['result_dir']),'--out',str(out)]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0 and out.read_text()=='preserve this previous result'


def rebind_evidence(fixture, name, operation):
    """Make internally consistent fresh fixture hashes; scientific gates must still hold."""
    mutate(fixture['root']/name,operation)
    mutate(fixture['registration'],lambda v:v['evidence_files'].__setitem__(name,hashed(fixture['root']/name)))
    (fixture['result_dir']/'registration.json').write_bytes(fixture['registration'].read_bytes())
    mutate(fixture['result_dir']/'metadata.json',lambda v:v.__setitem__('registration_sha256',hashed(fixture['registration'])))
    refresh_artifacts(fixture)


@pytest.mark.parametrize('study',[
    'human_task_structure_v1','human_osf_origin_v1','human_semantics_v1',
    'conditional_transition_v1','stochastic_feedback_network_v1','learning_ablation_v1','time_mixture_v1'])
def test_required_previous_verification_cannot_fail(frozen,study):
    rebind_evidence(frozen,'results/'+study+'/verification.json',lambda v:v.__setitem__('status','failed'))
    assert_failed(frozen)


@pytest.mark.parametrize('key,value',[
    ('raw_rows_replayed',74),('unmatched_rows_preserved',0),('values_replayed',74000),
    ('mismatches_verified',1),('source_identity_gate_passed',False),('mismatches_verified',False)])
def test_source_identity_and_unverified_rows_remain_separate(frozen,key,value):
    rebind_evidence(frozen,'results/human_osf_origin_v1/verification.json',lambda v:v.__setitem__(key,value))
    assert_failed(frozen)


@pytest.mark.parametrize('key,value',[
    ('source_coded_descriptive_metric_allowed',False),('exact_historical_task_execution_version_verified',True),
    ('direct_human_model_similarity_allowed',True),('direct_human_model_similarity_allowed',0)])
def test_original_task_gate_stays_closed(frozen,key,value):
    rebind_evidence(frozen,'results/human_task_structure_v1/audit.json',lambda v:v['gate'].__setitem__(key,value))
    assert_failed(frozen)


@pytest.mark.parametrize('operation',[
    lambda v:v['datasets'][0]['sheets'][0].__setitem__('participants',55),
    lambda v:v['datasets'][0]['sheets'][0].__setitem__('sheet','T2'),
    lambda v:v['datasets'].pop(),
])
def test_aggregate_schema_does_not_turn_into_trials(frozen,operation):
    rebind_evidence(frozen,'data/human/honore_noel_2016/schema-audit.json',operation)
    assert_failed(frozen)


def test_semantics_count_pass_is_not_mapping_pass(frozen):
    rebind_evidence(frozen,'results/human_semantics_v1/verification.json',lambda v:v.__setitem__('scope','mapping_verified'))
    assert_failed(frozen)


@pytest.mark.parametrize('key',['metrics_sha256','artifact_hashes_sha256'])
def test_previous_verifier_must_bind_exact_metric_inventory(frozen,key):
    rebind_evidence(frozen,'results/conditional_transition_v1/verification.json',lambda v:v.__setitem__(key,'0'*64))
    assert_failed(frozen)


@pytest.mark.parametrize('key',['bounded_phase4_deliverables_completed','source_coded_descriptive_metric_allowed',
                                'historical_task_execution_version_verified','openneuro_semantic_mapping_verified'])
def test_gate_numbers_cannot_impersonate_boolean_claims(frozen,key):
    mutate(frozen['result_dir']/'synthesis.json',lambda v:v.__setitem__(key,int(v[key])))
    refresh_artifacts(frozen)
    assert_failed(frozen)


def test_source_commit_must_exist_and_be_an_ancestor(frozen):
    mutate(frozen['registration'],lambda v:v.__setitem__('source_git_commit','0'*40))
    (frozen['result_dir']/'registration.json').write_bytes(frozen['registration'].read_bytes())
    mutate(frozen['result_dir']/'metadata.json',lambda v:v.__setitem__('registration_sha256',hashed(frozen['registration'])))
    refresh_artifacts(frozen)
    assert_failed(frozen)


@pytest.mark.parametrize('study,field',[
    ('human_task_structure_v1','result_sha256'),('stochastic_feedback_network_v1','summary_sha256'),
    ('learning_ablation_v1','summary_sha256'),('time_mixture_v1','metrics_sha256')])
def test_previous_independent_result_binding(frozen,study,field):
    rebind_evidence(frozen,'results/'+study+'/verification.json',lambda v:v.__setitem__(field,'0'*64))
    assert_failed(frozen)


@pytest.mark.parametrize('name',['extra.json','raw_trials.csv','unexpected_directory'])
def test_unregistered_output_artifact_is_rejected(frozen,name):
    file = frozen['result_dir']/name
    if name.endswith('directory'):
        file.mkdir()
    else:
        file.write_text('unexpected output',encoding='utf8')
    assert_failed(frozen)


def test_prior_verification_report_may_remain_for_readonly_rerun(frozen):
    dump(frozen['result_dir']/'verification.json',{'status':'prior independent report'})
    assert run(frozen)['passed']
