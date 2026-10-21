"""Register or execute the saved-evidence project closure once."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drosolearn.closure_envelope import register,run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--registration',type=Path)
    p.add_argument('--publication-proof',type=Path);a=p.parse_args()
    if a.mode=='register':value=register(a.root,a.config,a.out)
    else:value=run(a.root,a.config,a.registration,a.publication_proof,a.out)
    print(json.dumps({'status':value.get('status','registered'),'out':str(a.out)}))


if __name__=='__main__':main()
