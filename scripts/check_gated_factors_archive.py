"""Seal post-gate factor evidence without altering any completed study."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.gated_factors_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,committed,read,sha,utc,write_new,verify_envelope


def main():
    out=ROOT/'results/gated_factors_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/gated_factors_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['allowed_evolving_indexes']==sorted(EVOLVING_INDEXES)
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/gated_factors_v1';figures=ROOT/'results/gated_factors_figures_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/gated_factors_v1.json',ROOT/'results/registration_gated_factors_v1.json',
        folder,ROOT/'results/gated_factors_publication_proof.json')
    assert len(reg['source_files'])==20 and len(reg['parent_files'])==12
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['status']=='passed' and verified['study_id']=='gated_factors_v1'
    assert verified['scientific_producer_imported'] is False and verified['new_helper_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==sha(ROOT/'results/registration_gated_factors_v1.json')
    assert verified['counts']['arrays']==14 and verified['counts']['parent_trial_records_checked']==120000
    assert verified['counts']['destination_effects_checked']==480000 and verified['counts']['measurement_records_checked']==2880
    assert all(verified['counts'][key] is True for key in ('factual_head_prefixes_and_prior_accounting_checked',
        'only_post_both_actual_Go_eight_corners_checked','state_scalar_direction_symmetric_allocations_checked',
        'signed_positive_and_negative_mass_allocations_checked','own_other_and_all_unpaired_actual_actions_accounted',
        'factual_diagnostics_and_nullable_seed_means_checked','partial_control_information_not_silently_equalized',
        'no_rng_model_graph_solver_or_future_policy_calls'))
    assert utc(meta['completed_at_utc'])<=utc(verified['verified_at_utc'])<=datetime.now(timezone.utc)
    assert summary['validation']['all_exact_accounting_gates_passed'] is True
    counts=summary['counts'];assert counts['numeric_arrays']==14 and counts['measurement_records']==2880
    assert counts['corner_destination_effects']==counts['eligible_both_Go_trials']*32
    assert summary['interpretation']['parent_H_protection'] is True
    assert summary['interpretation']['parent_H_gain'] is summary['interpretation']['parent_H_acquisition'] is False
    tests=read(ROOT/'results/gated_factors_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files'] and tests['parent_byte_gate']==reg['gate']
    assert tests['actual_new_saved_factor_aggregation_before_freeze']==0
    assert tests['unique_passed_tests']==269 and tests['new_unique_passed_tests']==158 and tests['unchanged_related_regression_passed_tests']==111
    assert len(tests['test_receipts'])==1 and tests['test_receipts'][0]['path']=='results/gated_factors_prefreeze_tests/gated_factors_prefreeze_final.xml'
    prepaths={'results/gated_factors_prefreeze_validation.json','results/gated_factors_baseline_preservation.json',
        'results/gated_factors_prefreeze_evidence_exception.json'}
    for item in tests['test_receipts']+tests['development_failure_artifacts']:
        assert sha(ROOT/item['path'])==item['sha256'];prepaths.add(item['path'])
    failed_paths={'results/gated_factors_development/gated_factors_verify_development_failure_01.xml',
        'results/gated_factors_development/gated_factors_verify_failure_01.py',
        'results/gated_factors_development/gated_factors_verify_tests_failure_01.py'}
    assert len(tests['development_failure_artifacts'])==3 and {r['path'] for r in tests['development_failure_artifacts']}==failed_paths
    committed(ROOT,reg['source_git_commit'],{name:sha(ROOT/name) for name in sorted(prepaths)})
    cases=ET.parse(ROOT/tests['test_receipts'][0]['path']).findall('.//testcase')
    assert len(cases)==len({(c.get('classname'),c.get('name')) for c in cases})==269
    assert all(not any(c.find(k) is not None for k in ('failure','error','skipped')) for c in cases)
    newclasses={'tests.test_gated_factors','tests.test_gated_factors_envelope','tests.test_gated_factors_verify'}
    assert sum(c.get('classname') in newclasses for c in cases)==158
    assert sum(c.get('classname')=='tests.test_balanced_pathways' for c in cases)==67
    assert sum(c.get('classname')=='tests.test_gated_pathways' for c in cases)==44
    note=read(ROOT/'results/gated_factors_prefreeze_evidence_exception.json')
    raw='results/gated_factors_development/gated_factors_verify_development_failure_01.xml'
    assert note['status']=='passed' and note['original_failure_xml']==raw and note['original_failure_xml_sha256']==sha(ROOT/raw)
    assert note['scientific_sources_configuration_thresholds_or_parent_inputs_changed'] is False
    assert note['actual_factor_aggregation_started'] is False and note['prefreeze_validation_sha256']==sha(ROOT/'results/gated_factors_prefreeze_validation.json')
    failed=ET.parse(ROOT/raw).findall('.//testcase');assert len(failed)==9 and sum(c.find('failure') is not None for c in failed)==1
    assert all(c.find(k) is None for c in failed for k in ('error','skipped'))
    assert sum(c.find('failure') is None for c in failed)==8
    publication=read(ROOT/'results/gated_factors_publication_proof.json')
    chronology=[baseline['verified_at_utc'],tests['checked_at_utc'],note['checked_at_utc'],reg['registered_at_utc'],
        publication['checked_at_utc'],meta['started_at_utc'],meta['completed_at_utc'],verified['verified_at_utc']]
    times=list(map(utc,chronology));assert times==sorted(times) and times[3]<times[4]<times[5]
    proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    assert proof['arrays_sha256']==sha(folder/'arrays.npz') and proof['plotted_values_sha256']==sha(figures/'plotted_values.json')
    assert proof['render_script_sha256']==sha(ROOT/'scripts/render_gated_factors.py')
    expected_figures={'all-contrasts.png','factor-accounting.png','paired-diagnostics.png'};assert set(proof['figure_sha256'])==expected_figures
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    status=read(ROOT/'docs/repository-status.json')
    assert status['gated_factors_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/gated-factors-findings.md'
    assert status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False
    presentation=read(ROOT/'results/gated_factors_presentation_validation.json');assert presentation['status']=='passed'
    required_docs={'docs/gated-factors-findings.md','docs/gated-factors-reproduction.md'}|set(EVOLVING_INDEXES)
    assert required_docs<=set(presentation['document_sha256'])|set(presentation['additional_file_sha256'])
    required_additional={'scripts/render_gated_factors.py','results/gated_factors_figures_v1/plotted_values.json',
        'results/gated_factors_figures_v1/provenance.json'}|{'results/gated_factors_figures_v1/'+name for name in expected_figures}
    assert required_additional<=set(presentation['additional_file_sha256'])
    for key in ('document_sha256','additional_file_sha256'):
        for name,digest in presentation[key].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(EVOLVING_INDEXES)|prepaths|{
        'results/registration_gated_factors_v1.json','results/gated_factors_publication_proof.json',
        'results/gated_factors_presentation_validation.json','docs/gated-factors-findings.md','docs/gated-factors-reproduction.md',
        'scripts/check_gated_factors_archive.py','scripts/check_gated_factors_baseline.py','scripts/render_gated_factors.py'}
    for directory in (folder,figures):paths|={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/gated_factors_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Prior-tree raw-byte Git blob hash pass, later old-path diff, registered source/parent plus result/document/figure snapshots; not a new literal-byte whole-tree comparison.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':sha(ROOT/'results/registration_gated_factors_v1.json'),
        'prefreeze_receipts_committed_source_sha256':{name:sha(ROOT/name) for name in sorted(prepaths)},
        'prefreeze_publication_analysis_verification_utc_order':chronology,
        'original_development_failure_and_raw_whitespace_evidence_retained':True,
        'counts':counts,'scientific_verification_sha256':sha(folder/'verification.json'),
        'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Symmetric allocations on paired factual Go cells, not independent causal effects or a relearned policy. All one-sided actual actions retained; no new learning.'})
    print('Gated-factors archive passed: '+str(baseline['hash_compared_files'])+' old Git blobs preserved;all recorded factor arithmetic independently checked.')


if __name__=='__main__':main()
