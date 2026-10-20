"""Seal read-only pathway evidence, presentation and prior Git-blob hashes."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.pathways_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,read,sha,write_new,verify_envelope


def main():
    out=ROOT/'results/balanced_pathways_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/balanced_pathways_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/balanced_pathways_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/balanced_pathways_v1.json',ROOT/'results/registration_balanced_pathways_v1.json',
        folder,ROOT/'results/balanced_pathways_publication_proof.json')
    verified=read(folder/'verification.json');summary=read(folder/'summary.json');counts=summary['counts']
    assert verified['status']=='passed' and verified['scientific_producer_imported'] is False and verified['new_helper_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==meta['registration_sha256']
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verified['verified_at_utc'])
    assert counts['analyzed_parent_trial_records']==96000 and counts['analyzed_destination_effects']==384000
    assert verified['counts']['arrays']==19 and verified['counts']['measurement_records_checked']==1440
    assert all(counts[k]==0 for k in ('new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls',
        'new_random_draws','fresh_model_constructions','new_optimization_calls'))
    assert summary['validation']['all_exact_accounting_gates_passed'] is True
    tests=read(ROOT/'results/balanced_pathways_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['actual_saved_pathway_aggregations_before_freeze']==0
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    item=tests['development_failure_retained'];assert sha(ROOT/item['path'])==item['sha256']
    status=read(ROOT/'docs/repository-status.json')
    assert status['balanced_pathways_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/balanced-pathways-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/balanced_pathways_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    assert proof['arrays_sha256']==sha(folder/'arrays.npz') and proof['plotted_values_sha256']==sha(figures/'plotted_values.json')
    assert proof['render_script_sha256']==sha(ROOT/proof['render_script'])
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/balanced_pathways_presentation_validation.json');assert presentation['status']=='passed'
    for key in ('document_sha256','additional_file_sha256'):
        for name,digest in presentation[key].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_balanced_pathways_v1.json','results/balanced_pathways_publication_proof.json',
        'results/balanced_pathways_prefreeze_validation.json','results/balanced_pathways_baseline_preservation.json',
        'results/balanced_pathways_presentation_validation.json','docs/balanced-pathways-findings.md','docs/balanced-pathways-reproduction.md',
        'scripts/check_pathways_archive.py','scripts/check_pathways_baseline.py','scripts/render_balanced_pathways.py'}
    for directory in [folder,figures,ROOT/'results/balanced_pathways_prefreeze_tests']:
        paths|={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/balanced_pathways_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Whole prior-tree Git blob hash pass followed by old-path diff and frozen source/parent/result/document/figure hashes; not a new literal-byte whole-tree comparison.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],'counts':counts,
        'scientific_verification_sha256':sha(folder/'verification.json'),'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Retrospective exact recorded-path accounting, not unique causal attribution, alternative learning, new generalization or human mechanism.'})
    print('Pathways archive passed:'+str(baseline['hash_compared_files'])+' old Git blobs;384,000 effects; no new training.')


if __name__=='__main__':main()
