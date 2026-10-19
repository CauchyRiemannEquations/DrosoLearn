"""Literal Phase5-baseline preservation and completed follow-up byte receipts; no learning."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.representation_envelope import (BASE_COMMIT, BASE_TAG, read, sha,
                                               verify_envelope, write_new)
ALLOWED = {'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'}


def main():
    output = ROOT / 'results/representation_update_archive_validation.json'
    assert not output.exists()
    assert subprocess.check_output(['git','rev-parse',BASE_TAG],cwd=ROOT).decode().strip() == BASE_COMMIT
    raw = subprocess.check_output(['git','ls-tree','-r','--full-tree','-z',BASE_COMMIT],cwd=ROOT)
    entries = []
    for row in raw.split(b'\0'):
        if not row: continue
        header,name = row.split(b'\t',1)
        mode,kind,digest = header.split(); assert kind == b'blob'
        name = name.decode('utf8'); assert (ROOT / name).resolve().is_relative_to(ROOT)
        entries.append((digest.decode(),name))
    proc = subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    count = byte_count = 0
    failures, excluded = [], []
    try:
        for digest,name in entries:
            proc.stdin.write((digest+'\n').encode()); proc.stdin.flush()
            header = proc.stdout.readline().decode().split(); assert header[:2] == [digest,'blob']
            size = int(header[2]); blob = proc.stdout.read(size)
            assert len(blob) == size and proc.stdout.read(1) == b'\n'
            same = (ROOT / name).is_file() and (ROOT / name).read_bytes() == blob
            if name in ALLOWED:
                excluded.append({'path':name,'old_bytes':size,'unchanged':same}); continue
            count += 1; byte_count += size
            if not same: failures.append(name)
    finally:
        proc.stdin.close(); proc.wait(timeout=30); assert proc.returncode == 0
    assert not failures and {v['path'] for v in excluded} == ALLOWED
    directory = ROOT / 'results/representation_update_v1'
    proof = ROOT / 'results/representation_update_publication_proof.json'
    cfg,reg,meta = verify_envelope(ROOT, ROOT / 'configs/representation_update_v1.json',
                                  ROOT / 'results/registration_representation_update_v1.json', directory, proof)
    verification = read(directory / 'verification.json')
    assert verification['status'] == 'passed' and verification['scientific_producer_imported'] is False
    assert verification['summary_sha256'] == sha(directory / 'summary.json')
    assert verification['registration_sha256'] == meta['registration_sha256']
    assert verification['counts']['independently_replayed_training_actions'] == 48000
    assert meta['counts']['static_probe_cases'] == verification['counts']['checkpoint_static_probes'] == 2400
    assert datetime.fromisoformat(meta['completed_at_utc']) <= datetime.fromisoformat(verification['verified_at_utc'])
    figure_proof = read(ROOT / 'results/representation_update_figures_v1/provenance.json')
    assert figure_proof['verification_sha256'] == sha(directory / 'verification.json')
    assert figure_proof['summary_sha256'] == sha(directory / 'summary.json')
    for name,digest in figure_proof['figure_sha256'].items():
        assert sha(ROOT / 'results/representation_update_figures_v1' / name) == digest
    tests = read(ROOT / 'results/representation_update_prefreeze_validation.json')
    assert tests['status'] == 'passed' and tests['scientific_seed_draws_before_freeze'] == 0
    assert tests['frozen_source_hashes'] == reg['source_files']
    status = read(ROOT / 'docs/repository-status.json')
    assert status['representation_update_completed'] is True
    assert status['continuation_automation']['status'] == 'PAUSED'
    assert status['level4_similarity_supported'] is False and status['shared_neural_mechanism_verified'] is False
    paths = sorted(set(reg['source_files']) | set(reg['input_files']) | {
        'results/registration_representation_update_v1.json',
        'results/representation_update_publication_proof.json',
        'results/representation_update_prefreeze_validation.json',
        'docs/representation-update-findings.md', 'docs/representation-update-reproduction.md',
        'scripts/check_representation_archive.py', *ALLOWED,
    } | {p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file()}
       | {p.relative_to(ROOT).as_posix() for p in (ROOT / 'results/representation_update_figures_v1').iterdir() if p.is_file()})
    write_new(output, {'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,
        'old_tree_files':len(entries),'literal_compared_files':count,'literal_compared_bytes':byte_count,
        'allowed_evolving_indexes':excluded,'old_literal_failures':failures,
        'all_old_baseline_files_literal_preserved_except_three_indexes':True,
        'method':'Ordered git cat-file --batch literal byte comparison, no old learner rerun',
        'registered_source_files':len(reg['source_files']),'source_git_commit':reg['source_git_commit'],
        'registration_sha256':meta['registration_sha256'],'counts':meta['counts'],
        'scientific_verification_sha256':sha(directory / 'verification.json'),
        'current_file_sha256':{name:sha(ROOT/name) for name in paths},
        'notice':'Preserves completed Phase5 report and prior failures; new diagnostic is a separate follow-up. This receipt does not hash itself or assert future completion-tag/remote equality.'})
    print(json.dumps({'status':'passed','literal_files':count,'literal_bytes':byte_count,'new_receipt':str(output)}))


if __name__ == '__main__': main()
