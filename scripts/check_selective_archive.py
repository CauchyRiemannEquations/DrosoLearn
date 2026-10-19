"""Seal finite selective-spillover science and preservation receipts."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.selective_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,read,sha,write_new,verify_envelope


def main():
    out=ROOT/'results/selective_spillover_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/selective_spillover_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_baseline_files_literal_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert (old&changed)<=EVOLVING_INDEXES
    folder=ROOT/'results/selective_spillover_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/selective_spillover_v1.json',
        ROOT/'results/registration_selective_spillover_v1.json',folder,ROOT/'results/selective_spillover_publication_proof.json')
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['status']=='passed' and verified['scientific_producer_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==meta['registration_sha256']
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verified['verified_at_utc'])
    counts=meta['counts'];assert counts['training_feedback_trials']==96000 and counts['static_probe_cases']==4800
    assert counts['preflight_direction_evaluations']==19440 and counts['lookup_static_cases']==4800
    assert counts['internal_update_calls']==counts['value_update_calls']==counts['lookup_autonomous_training_trials']==0
    assert all(summary['information_validation'].values()) and summary['budget_validation']['common_bounds_valid_all']
    assert not summary['H_selective']['oracle_can_replace_primary'] and not summary['H_acquisition']['oracle_can_replace_primary']
    tests=read(ROOT/'results/selective_spillover_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['actual_research_graph_or_seed_used_before_freeze'] is False
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    status=read(ROOT/'docs/repository-status.json')
    assert status['selective_spillover_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/selective-spillover-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/selective_spillover_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    presentation=read(ROOT/'results/selective_spillover_presentation_validation.json');assert presentation['status']=='passed'
    for name,digest in presentation['document_sha256'].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(reg['graph_files'])|set(reg['input_files'])|set(EVOLVING_INDEXES)|{
        'results/registration_selective_spillover_v1.json','results/selective_spillover_publication_proof.json',
        'results/selective_spillover_prefreeze_validation.json','results/selective_spillover_baseline_preservation.json',
        'results/selective_spillover_presentation_validation.json','docs/selective-spillover-findings.md',
        'docs/selective-spillover-reproduction.md','scripts/check_selective_archive.py','scripts/check_selective_baseline.py'}
    paths|={item['path'] for item in tests['test_receipts']}
    paths|={p.relative_to(ROOT).as_posix() for p in (ROOT/'results/selective_spillover_prefreeze_tests').iterdir() if p.is_file()}
    paths|={'results/selective_spillover_prefreeze_format_draft.json','results/selective_spillover_prefreeze_whitespace_correction.json'}
    paths|={p.relative_to(ROOT).as_posix() for directory in (folder,figures) for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,'literal_baseline_receipt_sha256':sha(ROOT/'results/selective_spillover_baseline_preservation.json'),
        'literal_compared_files':baseline['literal_compared_files'],'literal_compared_bytes':baseline['literal_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_baseline_files_literal_preserved_except_three_indexes':True,
        'method':'Separate literal byte pass, final full Git diff of old paths, frozen source/parent/data/run/figure receipts; no previous learner rerun',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],'counts':counts,
        'scientific_verification_sha256':sha(folder/'verification.json'),'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Observed reward memory is separate learning state. Oracle and static lookup cannot replace primary feedback evidence. Original whole-tree pass and final index snapshot have separate UTC times.'})
    print('Selective archive passed: '+str(baseline['literal_compared_files'])+' old files preserved; new feedback/oracle results and information gates checked.')


if __name__=='__main__':main()
