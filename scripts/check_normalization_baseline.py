"""Read-only literal preservation of the completed parent release; no task arithmetic."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE_TAG='selective-spillover-v1'
BASE_COMMIT='a19a2abafcdb0db4b1ebdcc4b7629d237d112c55'
ALLOWED={'README.md','docs/repository-status.json','docs/research-roadmap.md'}


def main():
    out=ROOT/'results/normalization_loss_baseline_preservation.json'
    assert not out.exists()
    start=datetime.now(timezone.utc).isoformat()
    assert subprocess.check_output(['git','rev-parse',BASE_TAG],cwd=ROOT).decode().strip()==BASE_COMMIT
    rows=subprocess.check_output(['git','ls-tree','-r','--full-tree','-z',BASE_COMMIT],cwd=ROOT)
    entries=[]
    for row in rows.split(b'\0'):
        if not row:continue
        head,name=row.split(b'\t',1);mode,kind,digest=head.split();assert kind==b'blob'
        name=name.decode('utf8');assert (ROOT/name).resolve().is_relative_to(ROOT)
        entries.append((digest.decode(),name))
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    compared=byte_count=0;failures=[];excluded=[]
    try:
        for digest,name in entries:
            proc.stdin.write((digest+'\n').encode());proc.stdin.flush()
            head=proc.stdout.readline().decode().split();assert head[:2]==[digest,'blob']
            size=int(head[2]);blob=proc.stdout.read(size);assert len(blob)==size and proc.stdout.read(1)==b'\n'
            same=(ROOT/name).is_file() and (ROOT/name).read_bytes()==blob
            if name in ALLOWED:excluded.append({'path':name,'bytes':size,'unchanged':same});continue
            compared+=1;byte_count+=size
            if not same:failures.append(name)
            if compared%5000==0:print('Literal old-file comparison: '+str(compared),flush=True)
    finally:
        proc.stdin.close();proc.wait(timeout=30);assert proc.returncode==0
    assert not failures and {item['path'] for item in excluded}==ALLOWED
    value={'status':'passed','started_at_utc':start,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
           'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,'old_tree_files':len(entries),
           'literal_compared_files':compared,'literal_compared_bytes':byte_count,
           'allowed_evolving_indexes':excluded,'old_literal_failures':failures,
           'all_old_baseline_files_literal_preserved_except_three_indexes':True,
           'method':'Ordered Git cat-file --batch blob bytes compared with read_bytes; no old learner execution'}
    with out.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'passed','literal_files':compared,'literal_bytes':byte_count}))


if __name__=='__main__':main()
