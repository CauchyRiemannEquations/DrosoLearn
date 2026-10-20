"""Hash every prior Git blob against raw current bytes; no task computation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE_TAG='gated-factors-v1';BASE_COMMIT='d94922361e8a17e156bed0d5c4f240750f75f5ef'
ALLOWED={'README.md','docs/research-roadmap.md','docs/repository-status.json'}


def main():
    out=ROOT/'results/self_blend_baseline_preservation.json';assert not out.exists()
    start=datetime.now(timezone.utc).isoformat()
    assert subprocess.check_output(['git','rev-parse',BASE_TAG],cwd=ROOT).decode().strip()==BASE_COMMIT
    assert subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=ROOT).decode().strip()=='sha1'
    entries=subprocess.check_output(['git','ls-tree','-r','-z',BASE_COMMIT],cwd=ROOT).split(b'\0')
    count=size=0;excluded=[]
    for row in entries:
        if not row:continue
        head,name=row.split(b'\t',1);mode,kind,digest=head.split();assert kind==b'blob'
        name=name.decode('utf-8');path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        raw=path.read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if name in ALLOWED:excluded.append(name);continue
        assert actual==digest.decode(),name
        count+=1;size+=len(raw)
    assert set(excluded)==ALLOWED
    data={'status':'passed','started_at_utc':start,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,'hash_compared_files':count,'hash_compared_bytes':size,
        'allowed_evolving_indexes':sorted(excluded),'all_old_files_Git_blob_hash_preserved_except_three_indexes':True,
        'method':'Git SHA1 blob header plus raw current bytes for every prior tracked file; cryptographic hash comparison, not a repeated literal-byte comparison.',
        'prior_baseline_receipt_preserved':'results/gated_factors_baseline_preservation.json','new_learning_calls':0}
    with out.open('x',encoding='utf-8',newline='\n') as h:json.dump(data,h,ensure_ascii=False,indent=2);h.write('\n')
    print(json.dumps({'status':'passed','old_Git_blob_files':count,'bytes':size}))


if __name__=='__main__':main()
