"""Seal new read-only accounting while preserving all earlier studies."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.gated_pathways_envelope import BASE_COMMIT,BASE_TAG,EVOLVING_INDEXES,committed,read,sha,utc,write_new,verify_envelope


def main():
    out=ROOT/'results/gated_pathways_archive_validation.json';assert not out.exists()
    baseline=read(ROOT/'results/gated_pathways_baseline_preservation.json')
    assert baseline['status']=='passed' and baseline['baseline_commit']==BASE_COMMIT
    assert baseline['allowed_evolving_indexes']==sorted(EVOLVING_INDEXES)
    assert baseline['all_old_files_Git_blob_hash_preserved_except_three_indexes'] is True
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',BASE_COMMIT],cwd=ROOT).decode().splitlines())
    assert old&changed<=EVOLVING_INDEXES
    folder=ROOT/'results/gated_pathways_v1';figures=ROOT/'results/gated_pathways_figures_v1'
    cfg,reg,meta=verify_envelope(ROOT,ROOT/'configs/gated_pathways_v1.json',ROOT/'results/registration_gated_pathways_v1.json',
        folder,ROOT/'results/gated_pathways_publication_proof.json')
    assert len(reg['source_files'])==19 and len(reg['parent_files'])==11
    verified=read(folder/'verification.json');summary=read(folder/'summary.json')
    assert verified['status']=='passed' and verified['study_id']=='gated_pathways_v1'
    assert verified['scientific_producer_imported'] is False and verified['new_helper_imported'] is False
    assert verified['summary_sha256']==sha(folder/'summary.json') and verified['registration_sha256']==sha(ROOT/'results/registration_gated_pathways_v1.json')
    assert verified['counts']['arrays']==12 and verified['counts']['parent_trial_records_checked']==120000
    assert verified['counts']['destination_effects_checked']==480000 and verified['counts']['measurement_records_checked']==1800
    assert all(verified['counts'][key] is True for key in (
        'all_factual_head_prefixes_and_parent_effects_reconstructed',
        'all_methods_use_same_reference_prefeedback_time_boundary',
        'own_method_observations_used_for_known_unknown_categories',
        'last_discovery_trial_pre_and_next_trial_post_checked',
        'positive_negative_masses_and_endpoint_contributions_separated',
        'costs_Go_counts_and_nullable_descriptive_ratios_checked',
        'all_four_controls_and_all_thirty_families_checked',
        'parent_full_autonomous_replay_and_byte_gate_reused_by_frozen_hashes',
        'no_counterfactual_direction_policy_rng_graph_or_solver_calls'))
    assert utc(meta['completed_at_utc'])<=utc(verified['verified_at_utc'])<=datetime.now(timezone.utc)
    assert summary['validation']['all_exact_accounting_gates_passed'] is True
    assert summary['counts']['analyzed_parent_trial_records']==120000 and summary['counts']['analyzed_destination_effects']==480000
    assert summary['counts']['numeric_arrays']==12 and summary['counts']['measurement_records']==1800
    assert summary['interpretation']['parent_H_protection'] is True
    assert summary['interpretation']['parent_H_gain'] is summary['interpretation']['parent_H_acquisition'] is False
    tests=read(ROOT/'results/gated_pathways_prefreeze_validation.json')
    assert tests['status']=='passed' and tests['frozen_source_hashes']==reg['source_files'] and tests['parent_byte_gate']==reg['gate']
    assert tests['actual_new_saved_path_aggregation_before_freeze']==0
    assert tests['unique_passed_tests']==256 and tests['new_unique_passed_tests']==142
    assert tests['unchanged_related_regression_passed_tests']==114
    publication=read(ROOT/'results/gated_pathways_publication_proof.json')
    chronology=[baseline['verified_at_utc'],tests['checked_at_utc'],reg['registered_at_utc'],
        publication['checked_at_utc'],meta['started_at_utc'],meta['completed_at_utc'],verified['verified_at_utc']]
    times=list(map(utc,chronology));assert times==sorted(times) and times[2]<times[3]<times[4]
    prepaths={'results/gated_pathways_prefreeze_validation.json','results/gated_pathways_baseline_preservation.json'}
    assert len(tests['test_receipts'])==1
    assert tests['test_receipts'][0]['path']=='results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml'
    for item in tests['test_receipts']:
        assert sha(ROOT/item['path'])==item['sha256'];prepaths.add(item['path'])
    committed(ROOT,reg['source_git_commit'],{name:sha(ROOT/name) for name in sorted(prepaths)})
    cases=ET.parse(ROOT/tests['test_receipts'][0]['path']).findall('.//testcase')
    assert len(cases)==len({(c.get('classname'),c.get('name')) for c in cases})==tests['unique_passed_tests']
    assert all(not any(c.find(k) is not None for k in ('failure','error','skipped')) for c in cases)
    newclasses={'tests.test_gated_pathways','tests.test_gated_pathways_envelope','tests.test_gated_pathways_verify'}
    assert sum(c.get('classname') in newclasses for c in cases)==142
    assert sum(c.get('classname') not in newclasses for c in cases)==114
    proof=read(figures/'provenance.json')
    assert proof['verification_sha256']==sha(folder/'verification.json') and proof['summary_sha256']==sha(folder/'summary.json')
    assert proof['arrays_sha256']==sha(folder/'arrays.npz') and proof['plotted_values_sha256']==sha(figures/'plotted_values.json')
    assert proof['render_script_sha256']==sha(ROOT/'scripts/render_gated_pathways.py')
    expected_figures={'all-contrasts.png','help-and-harm.png','phase-accounting.png'}
    assert set(proof['figure_sha256'])==expected_figures
    for name,digest in proof['figure_sha256'].items():assert sha(figures/name)==digest
    status=read(ROOT/'docs/repository-status.json')
    assert status['gated_pathways_completed'] is True and status['continuation_automation']['status']=='PAUSED'
    assert status['current_findings_document']=='docs/gated-pathways-findings.md'
    presentation=read(ROOT/'results/gated_pathways_presentation_validation.json');assert presentation['status']=='passed'
    required_docs={'docs/gated-pathways-findings.md','docs/gated-pathways-reproduction.md'}|set(EVOLVING_INDEXES)
    assert required_docs<=set(presentation['document_sha256'])|set(presentation['additional_file_sha256'])
    required_additional={'scripts/render_gated_pathways.py','results/gated_pathways_figures_v1/plotted_values.json',
        'results/gated_pathways_figures_v1/provenance.json'}|{'results/gated_pathways_figures_v1/'+name for name in expected_figures}
    assert required_additional<=set(presentation['additional_file_sha256'])
    for key in ('document_sha256','additional_file_sha256'):
        for name,digest in presentation[key].items():assert sha(ROOT/name)==digest
    paths=set(reg['source_files'])|set(reg['parent_files'])|set(EVOLVING_INDEXES)|prepaths|{
        'results/registration_gated_pathways_v1.json','results/gated_pathways_publication_proof.json',
        'results/gated_pathways_presentation_validation.json','docs/gated-pathways-findings.md','docs/gated-pathways-reproduction.md',
        'scripts/check_gated_pathways_archive.py','scripts/check_gated_pathways_baseline.py','scripts/render_gated_pathways.py'}
    for directory in (folder,figures):paths|={p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
    write_new(out,{'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'baseline_hash_receipt_sha256':sha(ROOT/'results/gated_pathways_baseline_preservation.json'),
        'hash_compared_files':baseline['hash_compared_files'],'hash_compared_bytes':baseline['hash_compared_bytes'],
        'subsequent_old_file_changes':sorted(old&changed),'allowed_evolving_indexes':sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Separate prior-tree raw-byte Git blob hash pass plus later old-path diff and registered source/parent/result/document/figure snapshots; no new literal-byte whole-tree comparison claimed.',
        'source_git_commit':reg['source_git_commit'],'registration_sha256':sha(ROOT/'results/registration_gated_pathways_v1.json'),
        'prefreeze_receipts_committed_source_sha256':{name:sha(ROOT/name) for name in sorted(prepaths)},
        'prefreeze_publication_analysis_verification_utc_order':chronology,
        'counts':summary['counts'],'scientific_verification_sha256':sha(folder/'verification.json'),
        'current_file_sha256':{name:sha(ROOT/name) for name in sorted(paths)},
        'notice':'Common reference gate splits recorded time; destinations use each method actual observations. Contributions telescope, without unique causal or generalization claims; no new learning.'})
    print('Gated-pathways archive passed: '+str(baseline['hash_compared_files'])+' old Git blobs preserved;480,000 destination effects independently checked.')


if __name__=='__main__':main()
