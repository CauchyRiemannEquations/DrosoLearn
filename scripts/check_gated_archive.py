"""Seal prospective gated-learning evidence and prior Git-blob preservation."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.gated_envelope import (
    BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,ancestor,committed,read,sha,utc,write_new,verify_envelope,
)

FINAL_SOURCE_COMMIT='6c3f709263c8fc7bea8ad52efbafca96afc5d188'
PREMATURE_SOURCE_COMMIT='32de2f9d5165108d711f2c7d9ae3d431059d19ca'
PREFREEZE_RECEIPT='results/gated_learning_prefreeze_validation.json'
FINALIZATION_RECEIPT='results/gated_learning_prefreeze_finalization.json'
BASELINE_RECEIPT='results/gated_learning_baseline_preservation.json'
FINAL_JUNIT='results/gated_learning_prefreeze_tests/gated_prefreeze_tests_final.xml'
FINALIZED_PATHS=frozenset({PREFREEZE_RECEIPT,FINALIZATION_RECEIPT,FINAL_JUNIT,
                           'tests/test_gated_learning_verify.py'})


def main():
    out=ROOT/'results/gated_learning_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/gated_learning_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT and baseline['baseline_tag']==BASE_TAG
    assert baseline['allowed_evolving_indexes']==sorted(EVOLVING_INDEXES)
    assert type(baseline['new_learning_calls']) is int and baseline['new_learning_calls']==0
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/gated_learning_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/gated_learning_v1.json',ROOT/'results/registration_gated_learning_v1.json',
        folder,ROOT/'results/gated_learning_publication_proof.json')
    assert reg['source_git_commit']==FINAL_SOURCE_COMMIT and len(reg['source_files'])==39
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['study_id']=='gated_learning_v1' and verified['status']=='passed'
    assert verified['scientific_producer_imported'] is False and verified['new_helper_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==meta['registration_sha256']
    assert utc(meta['completed_at_utc'])<=utc(verified['verified_at_utc'])<=datetime.now(timezone.utc)
    counts=meta['counts'];assert counts['training_feedback_trials']==120000 and counts['static_probe_cases']==6000
    assert counts['internal_update_calls']==counts['value_update_calls']==0
    assert verified['counts']['arrays']==195 and verified['counts']['independently_replayed_training_actions']==120000
    assert verified['counts']['unique_balanced_states_independently_reoptimized']==counts['balanced_unique_solver_cases']==720
    assert summary['validation']['all_numeric_and_information_gates_passed'] is True
    assert all(summary['gate_validation'].values())
    tests=read(ROOT/PREFREEZE_RECEIPT)
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['baseline_commit']==BASE_COMMIT and tests['parent_byte_gate']==reg['gate']
    assert tests['unique_passed_tests']==tests['new_unique_passed_tests']+tests['unchanged_related_regression_passed_tests']==242
    assert tests['actual_new_cohort_graph_constructions_before_freeze']==0
    assert tests['actual_new_cohort_learning_or_aggregation_before_freeze']==0
    assert len(tests['test_receipts'])==1 and tests['test_receipts'][0]['path']==FINAL_JUNIT
    for item in tests['test_receipts']:
        assert sha(ROOT/item['path'])==item['sha256'] and item['passed_unique_tests']==242
    testcases=ET.parse(ROOT/FINAL_JUNIT).getroot().findall('.//testcase')
    identities={(item.get('classname'),item.get('name')) for item in testcases}
    assert len(testcases)==len(identities)==242
    assert all(item.get('classname') and item.get('name') and not any(
        item.find(tag) is not None for tag in ('failure','error','skipped')) for item in testcases)
    finalization=read(ROOT/FINALIZATION_RECEIPT)
    assert finalization['status']=='passed' and finalization['new_cohort_graph_or_learning_started'] is False
    assert finalization['final_prefreeze_receipt_sha256']==sha(ROOT/PREFREEZE_RECEIPT)
    assert finalization['all39_final_source_hashes_verified'] is True and finalization['passed_unique_tests']==242
    assert finalization['configuration_seeds_budget_thresholds_or_model_source_changed'] is False
    assert finalization['prefreeze_helper_final_stage_assertion_rejected_changed_HEAD'] is True
    assert finalization['registration_rejected_file']=='tests/test_gated_learning_verify.py'
    assert finalization['premature_unpublished_source_commit']==PREMATURE_SOURCE_COMMIT
    ancestor(ROOT,BASE_COMMIT,PREMATURE_SOURCE_COMMIT)
    ancestor(ROOT,PREMATURE_SOURCE_COMMIT,FINAL_SOURCE_COMMIT)
    finalized_changes=set(subprocess.check_output(['git','diff','--name-only',PREMATURE_SOURCE_COMMIT,
        FINAL_SOURCE_COMMIT],cwd=ROOT).decode().splitlines())
    assert finalized_changes==FINALIZED_PATHS
    # Only the verifier test snapshot changed within the 39-source closure.
    # Model, config, seed/budget criteria, and every other source were already
    # identical in the retained initial commit; no scientific run is repeated.
    committed(ROOT,PREMATURE_SOURCE_COMMIT,{name:digest for name,digest in reg['source_files'].items()
        if name!='tests/test_gated_learning_verify.py'})
    prefreeze_hashes={name:sha(ROOT/name) for name in
        (BASELINE_RECEIPT,PREFREEZE_RECEIPT,FINALIZATION_RECEIPT,FINAL_JUNIT)}
    committed(ROOT,FINAL_SOURCE_COMMIT,prefreeze_hashes)
    publication=read(ROOT/'results/gated_learning_publication_proof.json')
    chronology={
        'baseline_started':baseline['started_at_utc'],'baseline_verified':baseline['verified_at_utc'],
        'final_tests_checked':tests['checked_at_utc'],'finalization_checked':finalization['checked_at_utc'],
        'registered':reg['registered_at_utc'],'remote_bytes_checked':publication['checked_at_utc'],
        'learning_started':meta['started_at_utc'],'learning_completed':meta['completed_at_utc'],
        'independently_verified':verified['verified_at_utc'],
    }
    times=[utc(value) for value in chronology.values()]
    assert times==sorted(times) and times[-1]<=datetime.now(timezone.utc)
    assert times[3]<times[4]<times[5]<times[6]
    status=read(ROOT/'docs/repository-status.json')
    assert status['gated_learning_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/gated-learning-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/gated_learning_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    assert proof['arrays_sha256']==sha(folder/'arrays.npz') and proof['plotted_values_sha256']==sha(figures/'plotted_values.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/gated_learning_presentation_validation.json');assert presentation['status']=='passed'
    for name,digest in presentation['document_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in presentation.get('additional_file_sha256',{}).items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(reg['graph_files'])|set(reg['input_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_gated_learning_v1.json','results/gated_learning_publication_proof.json',
        'results/gated_learning_prefreeze_validation.json','results/gated_learning_baseline_preservation.json',
        'results/gated_learning_prefreeze_finalization.json',
        'results/gated_learning_presentation_validation.json','docs/gated-learning-findings.md','docs/gated-learning-reproduction.md',
        'scripts/check_gated_archive.py','scripts/check_gated_baseline.py','scripts/render_gated_learning.py'}
    for directory in [folder,figures,ROOT/'results/gated_learning_prefreeze_tests']:
        paths|={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/gated_learning_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Separate whole-old-tree Git blob hash pass plus later full old-path diff and frozen source/parent/data/result/figure hashes; no new literal-byte whole-tree comparison claimed.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],'counts':counts,
        'prefreeze_receipts_committed_source_sha256':prefreeze_hashes,
        'prefreeze_passed_unique_tests_independently_counted':len(identities),
        'prefreeze_finalization_retained_initial_commit':PREMATURE_SOURCE_COMMIT,
        'prefreeze_finalization_changed_paths':sorted(finalized_changes),
        'configuration_and_model_source_unchanged_during_finalization':True,
        'prefreeze_to_independent_verification_utc_order':chronology,
        'scientific_verification_sha256':sha(folder/'verification.json'),'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Autonomous output learning under common ceilings; actual movement/compute costs differ. Reward-unobserved other training inputs are not novel-input generalization.'})
    print('Gated learning archive passed: '+str(baseline['hash_compared_files'])+' old Git blobs preserved;120,000 actual actions independently replayed.')


if __name__=='__main__':main()
