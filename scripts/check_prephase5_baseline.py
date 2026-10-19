"""Literal preservation of the pre-followup release; reads Git/files only."""
from pathlib import Path
import json, subprocess
ROOT=Path(__file__).resolve().parents[1]
TAG='human-shell-comparison-v2'
EXPECTED='3ee71a97fa9d6ffc7fce3bd4d7c3832437977171'
ALLOWED={'README.md','docs/research-roadmap.md','docs/repository-status.json'}
def main():
 commit=subprocess.check_output(['git','rev-parse',TAG],cwd=ROOT).decode().strip();assert commit==EXPECTED
 raw=subprocess.check_output(['git','ls-tree','-r','--full-tree','-z',TAG],cwd=ROOT)
 entries=[]
 for row in raw.split(b'\0'):
  if not row:continue
  head,path=row.split(b'\t',1);mode,kind,digest=head.split();assert kind==b'blob'
  relative=path.decode('utf8');assert (ROOT/relative).resolve().is_relative_to(ROOT.resolve())
  entries.append((digest.decode(),relative))
 proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 count=byte_count=0;excluded=[];failures=[]
 try:
  for digest,name in entries:
   proc.stdin.write((digest+'\n').encode());proc.stdin.flush()
   header=proc.stdout.readline().decode().strip().split();assert header[:2]==[digest,'blob']
   size=int(header[2]);blob=proc.stdout.read(size);assert len(blob)==size and proc.stdout.read(1)==b'\n'
   target=ROOT/name;same=target.is_file() and target.read_bytes()==blob
   if name in ALLOWED:excluded.append({'path':name,'bytes':size,'unchanged':same});continue
   count+=1;byte_count+=size
   if not same:failures.append(name)
 finally:
  proc.stdin.close();proc.wait(timeout=30);assert proc.returncode==0,proc.stderr.read().decode(errors='replace')
 assert {x['path'] for x in excluded}==ALLOWED
 result={'tag':TAG,'tag_commit':commit,'old_tree_files':len(entries),'compared_files':count,
         'compared_literal_bytes':byte_count,'excluded_files':excluded,'passed':not failures,'failures':failures,
         'method':'Literal Git cat-file --batch blob bytes compared with read_bytes; no mutations'}
 print(json.dumps(result,ensure_ascii=False,indent=2));assert not failures
if __name__=='__main__':main()
