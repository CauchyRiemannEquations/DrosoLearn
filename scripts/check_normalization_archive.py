"""Seal saved-path normalization accounting and preservation receipts."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from drosolearn.normalization_envelope import BASE_COMMIT,BASE_TAG,read,sha,write_new,verify_envelope,EVOLVING_INDEXES


def main():
    out=ROOT/'results/normalization_loss_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/normalization_loss_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_baseline_files_literal_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert (old&changed)<=EVOLVING_INDEXES
    folder=ROOT/'results/normalization_loss_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/normalization_loss_v1.json',
        ROOT/'results/registration_normalization_loss_v1.json',folder,ROOT/'results/normalization_loss_publication_proof.json')
    verified=read(folder/'verification.json')
    assert verified['status']=='passed' and verified['scientific_producer_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json')
    assert verified['registration_sha256']==meta['registration_sha256']
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verified['verified_at_utc'])
    counts=meta['counts']
    assert counts['analyzed_parent_trial_records']==96000 and counts['trial_destination_cells']==384000
    assert all(counts[key]==0 for key in ['new_training_trials','new_head_update_calls','new_internal_update_calls',
        'new_value_update_calls','new_random_draws','fresh_model_constructions'])
    tests=read(ROOT/'results/normalization_loss_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['actual_saved_normalization_aggregations_before_freeze']==0
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    status=read(ROOT/'docs/repository-status.json')
    assert status['normalization_loss_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/normalization-loss-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/normalization_loss_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json')
    assert proof['summary_sha256']==sha(folder/'summary.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/normalization_loss_presentation_validation.json');assert presentation['status']=='passed'
    for name,digest in presentation['document_sha256'].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_normalization_loss_v1.json','results/normalization_loss_publication_proof.json',
        'results/normalization_loss_prefreeze_validation.json','results/normalization_loss_baseline_preservation.json',
        'results/normalization_loss_presentation_validation.json','docs/normalization-loss-findings.md',
        'docs/normalization-loss-reproduction.md','scripts/check_normalization_archive.py','scripts/check_normalization_baseline.py'}
    paths|={item['path'] for item in tests['test_receipts']}
    paths|={p.relative_to(ROOT).as_posix() for directory in (folder,figures) for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'literal_baseline_receipt_sha256':sha(ROOT/'results/normalization_loss_baseline_preservation.json'),
        'literal_compared_files':baseline['literal_compared_files'],'literal_compared_bytes':baseline['literal_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_baseline_files_literal_preserved_except_three_indexes':True,
        'method':'Separate literal byte pass followed by full Git diff of old paths and current frozen source/parent/result checks; no learner rerun',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],
        'counts':counts,'scientific_verification_sha256':sha(folder/'verification.json'),
        'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Retrospective descriptive accounting only. Baseline byte comparison and final index snapshot have separate UTC times; no future tag/remote equality asserted.'})
    print('Normalization-loss archive passed: '+str(baseline['literal_compared_files'])+' old files preserved; no new learning or draws.')


if __name__=='__main__':main()
