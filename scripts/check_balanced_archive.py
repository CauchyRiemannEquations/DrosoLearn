"""Seal prospective balanced-learning evidence and prior Git-blob preservation."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.balanced_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,read,sha,write_new,verify_envelope


def main():
    out=ROOT/'results/balanced_learning_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/balanced_learning_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/balanced_learning_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/balanced_learning_v1.json',ROOT/'results/registration_balanced_learning_v1.json',
        folder,ROOT/'results/balanced_learning_publication_proof.json')
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['status']=='passed' and verified['scientific_producer_imported'] is False and verified['new_helper_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==meta['registration_sha256']
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verified['verified_at_utc'])
    counts=meta['counts'];assert counts['training_feedback_trials']==96000 and counts['static_probe_cases']==4800
    assert counts['internal_update_calls']==counts['value_update_calls']==0
    assert verified['counts']['arrays']==190 and verified['counts']['independently_replayed_training_actions']==96000
    assert verified['counts']['unique_balanced_states_independently_reoptimized']==counts['balanced_unique_solver_cases']
    assert summary['validation']['all_numeric_and_information_gates_passed'] is True
    tests=read(ROOT/'results/balanced_learning_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['actual_new_cohort_graph_constructions_before_freeze']==0
    assert tests['actual_new_cohort_learning_or_aggregation_before_freeze']==0
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    cleanup=tests['nonsemantic_EOF_cleanup'];assert sha(ROOT/cleanup['retained_prior_receipt'])==cleanup['prior_receipt_sha256']
    assert cleanup['only_removed_extra_terminal_blank_line'] is True
    status=read(ROOT/'docs/repository-status.json')
    assert status['balanced_learning_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/balanced-learning-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/balanced_learning_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    assert proof['arrays_sha256']==sha(folder/'arrays.npz') and proof['plotted_values_sha256']==sha(figures/'plotted_values.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/balanced_learning_presentation_validation.json');assert presentation['status']=='passed'
    for name,digest in presentation['document_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in presentation.get('additional_file_sha256',{}).items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(reg['graph_files'])|set(reg['input_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_balanced_learning_v1.json','results/balanced_learning_publication_proof.json',
        'results/balanced_learning_prefreeze_validation.json','results/balanced_learning_baseline_preservation.json',
        'results/balanced_learning_presentation_validation.json','docs/balanced-learning-findings.md','docs/balanced-learning-reproduction.md',
        'scripts/check_balanced_archive.py','scripts/check_balanced_baseline.py','scripts/render_balanced_learning.py'}
    for directory in [folder,figures,ROOT/'results/balanced_learning_prefreeze_tests']:
        paths|={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/balanced_learning_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Separate whole-old-tree Git blob hash pass plus later full old-path diff and frozen source/parent/data/result/figure hashes; no new literal-byte whole-tree comparison claimed.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],'counts':counts,
        'scientific_verification_sha256':sha(folder/'verification.json'),'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Autonomous output learning under common ceilings; actual movement/compute costs differ. Reward-unobserved other training inputs are not novel-input generalization.'})
    print('Balanced learning archive passed: '+str(baseline['hash_compared_files'])+' old Git blobs preserved;96,000 actual actions independently replayed.')


if __name__=='__main__':main()
