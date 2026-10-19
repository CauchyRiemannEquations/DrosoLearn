"""Finalize bounded-cost diagnostic receipts; no model simulation or new inference."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from drosolearn.spillover_envelope import BASE_COMMIT,BASE_TAG,read,sha,write_new,verify_envelope
ALLOWED={'README.md','docs/repository-status.json','docs/research-roadmap.md'}


def main():
    out=ROOT/'results/spillover_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/spillover_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['all_old_baseline_files_literal_preserved_except_three_indexes'] is True
    old_names=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert (changed & old_names) <= ALLOWED
    directory=ROOT/'results/spillover_tradeoff_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/spillover_tradeoff_v1.json',
        ROOT/'results/registration_spillover_tradeoff_v1.json',directory,ROOT/'results/spillover_tradeoff_publication_proof.json')
    verification=read(directory/'verification.json')
    assert verification['status']=='passed' and verification['scientific_producer_imported'] is False
    assert verification['summary_sha256']==sha(directory/'summary.json')
    assert verification['registration_sha256']==meta['registration_sha256']
    assert meta['counts']['training_feedback_trials']==120000 and meta['counts']['static_probe_cases']==6000
    assert datetime.fromisoformat(meta['completed_at_utc'])<=datetime.fromisoformat(verification['verified_at_utc'])
    tests=read(ROOT/'results/spillover_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files']
    assert tests['scientific_seed_draws_before_freeze']==0
    for item in tests['test_receipts']:assert sha(ROOT/item['path'])==item['sha256']
    status=read(ROOT/'docs/repository-status.json')
    assert status['spillover_tradeoff_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/spillover-tradeoff-findings.md'
    assert not status['level4_similarity_supported'] and not status['shared_neural_mechanism_verified']
    figures=ROOT/'results/spillover_figures_v1';proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(directory/'verification.json')
    assert proof['summary_sha256']==sha(directory/'summary.json')
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    paths=set(reg['source_files'])|set(reg['input_files'])|set(reg['parent_files'])|set(reg['graph_files'])|ALLOWED|{
        'results/registration_spillover_tradeoff_v1.json','results/spillover_tradeoff_publication_proof.json',
        'results/spillover_prefreeze_validation.json','results/spillover_baseline_preservation.json',
        'results/spillover_presentation_validation.json','results/spillover_prefreeze_whitespace_correction.json',
        'docs/spillover-tradeoff-findings.md','docs/spillover-tradeoff-reproduction.md',
        'scripts/check_spillover_archive.py','scripts/check_spillover_baseline.py'}
    paths|={p.relative_to(ROOT).as_posix() for folder in (directory,figures) for p in folder.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'literal_baseline_receipt_sha256':sha(ROOT/'results/spillover_baseline_preservation.json'),
        'literal_compared_files':baseline['literal_compared_files'],'literal_compared_bytes':baseline['literal_compared_bytes'],
        'subsequent_old_file_changes':sorted(changed & old_names),'allowed_evolving_indexes':sorted(ALLOWED),
        'all_old_baseline_files_literal_preserved_except_three_indexes':True,
        'method':'Literal parent-tree byte pass followed by Git full diff of old paths; source/parent/input/graph/result receipts rechecked; no old learner rerun',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':meta['registration_sha256'],
        'counts':meta['counts'],'scientific_verification_sha256':sha(directory/'verification.json'),
        'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'The literal baseline pass and final evolving-index snapshot have separate UTC times. No future completion-tag or remote equality is asserted here.'})
    print('Archive passed: '+str(baseline['literal_compared_files'])+' preserved old files, new frozen sources/results/figures checked.')


if __name__=='__main__':main()
