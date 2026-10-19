"""Seal finite geometric analysis and old Git-blob preservation checks."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.cone_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,read,sha,write_new,verify_envelope


def main():
    out=ROOT/'results/cone_frontier_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/cone_frontier_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/cone_frontier_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/cone_frontier_v1.json',ROOT/'results/registration_cone_frontier_v1.json',
        folder,ROOT/'results/cone_frontier_publication_proof.json')
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['status']=='passed' and verified['scientific_producer_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==meta['registration_sha256']
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verified['verified_at_utc'])
    counts=meta['counts'];assert counts['analyzed_parent_trial_records']==24000 and counts['candidate_destination_effects']==672000
    assert all(counts[k]==0 for k in ['new_training_trials','new_head_update_calls','new_internal_update_calls',
        'new_value_update_calls','new_random_draws','fresh_model_constructions'])
    assert all(summary['validation'][k] is True for k in ['all_unit_caps_passed','all_observed_and_oracle_constraints_passed',
        'all_case_certificates_passed','all_phase_balances_passed'])
    tests=read(ROOT/'results/cone_frontier_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['actual_saved_cone_aggregations_before_freeze']==0
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    status=read(ROOT/'docs/repository-status.json')
    assert status['cone_frontier_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/cone-frontier-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/cone_frontier_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/cone_frontier_presentation_validation.json');assert presentation['status']=='passed'
    for name,digest in presentation['document_sha256'].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_cone_frontier_v1.json','results/cone_frontier_publication_proof.json',
        'results/cone_frontier_prefreeze_validation.json','results/cone_frontier_baseline_preservation.json',
        'results/cone_frontier_presentation_validation.json','docs/cone-frontier-findings.md','docs/cone-frontier-reproduction.md',
        'scripts/check_cone_archive.py','scripts/check_cone_baseline.py'}
    paths|={item['path'] for item in tests['test_receipts']}
    paths|={p.relative_to(ROOT).as_posix() for directory in [folder,figures] for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/cone_frontier_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Separate whole-old-tree Git blob hash pass plus later full old-path diff and frozen source/parent/result/figure hashes; prior literal receipt retained, not relabeled.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],'counts':counts,
        'scientific_verification_sha256':sha(folder/'verification.json'),'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Known-sign geometric optima and full-information safety references are not autonomous learning outcomes or universal common upper bounds.'})
    print('Cone archive passed: '+str(baseline['hash_compared_files'])+' old Git blobs preserved; no new learning.')


if __name__=='__main__':main()
