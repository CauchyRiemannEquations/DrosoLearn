"""Render registered comparison summaries without recomputing effects or learning."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def plot(result_dir,out,cache):
    read=lambda name:json.loads((result_dir/name).read_text(encoding='utf8'))
    reg=read('registration.json');meta=read('metadata.json');check=read('verification.json');artifacts=read('artifact_hashes.json')
    if out.exists() or meta['status']!='completed' or check['status']!='passed':
        raise ValueError('Verified complete result and exclusive figure path required')
    if reg['source_files']['scripts/plot_human_shell.py']!=sha(Path(__file__)) or artifacts['summary.json']!=sha(result_dir/'summary.json'):
        raise ValueError('Frozen plot or primary summary bytes differ')
    summary=read('summary.json')
    rows=summary['behavior_family_comparisons']
    conditions=reg['config']['conditions']
    if [row['condition'] for row in rows]!=conditions:raise ValueError('Complete family order required')
    out.mkdir(parents=True)
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),sharey=True)
    titles=['Initial acquisition gain','Early reversal choices','Reversal recovery gain']
    for index,ax in enumerate(axes):
        ax.axvspan(-10,10,color='#e5eced',label='Prespecified +/-10 pp margin')
        ax.axvline(0,color='#667777',linewidth=.7)
        for y,row in enumerate(rows):
            cell=row['metrics'][index]
            est=100*cell['difference'];lo,hi=[100*v for v in cell['ci90']]
            color='#bc4036' if row['condition']=='plastic' else '#354f70'
            ax.plot([lo,hi],[y,y],color=color,linewidth=1.7)
            ax.plot(est,y,'o',color=color,markersize=5)
        ax.set_title(titles[index]);ax.set_xlabel('Model minus human (percentage points)')
        ax.grid(axis='x',alpha=.2);ax.set_yticks(range(len(conditions)),conditions)
    axes[0].invert_yaxis()
    fig.suptitle('Recorded shell-trace benchmark: 90% crossed-bootstrap intervals\nPrimary condition: plastic; other families are descriptive controls',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(out/'behavior_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    values=summary['conditional_family_mean_nll']
    ax.bar(conditions,[values[c] for c in conditions],color=['#bc4036' if c=='plastic' else '#54769a' for c in conditions])
    ax.set_ylabel('Held-out participant mean NLL (nats/trial; lower is better)')
    ax.set_title('Next-choice prediction from the recorded human past\nConditional predictions; separate from autonomous learning')
    ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'heldout_prediction.png',dpi=150);plt.close(fig)
    import sys
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    from drosolearn.human_shell_source import load_cohort
    manifest=json.loads((root/reg['config']['manifest']).read_text(encoding='utf8'))
    records,_=load_cohort(cache,manifest)
    with np.load(result_dir/'trajectories.npz',allow_pickle=False) as values:
        free=values['actions'][1]
    # Use the same seven source exposures in every reversal; later short phases
    # are not interpolated. Six episodes stay nested in each original person.
    human_curves=[];model_curves={c:[] for c in ('plastic','fixed','q','uniform','persistence')}
    for pi,pid in enumerate(reg['config']['test_ids']):
        rows=records[pid];person_human=[];person_models={c:[] for c in model_curves}
        for shell in (3,4):
            episodes=[]
            for i,row in enumerate(rows):
                if row['shell']!=shell:continue
                if not episodes or rows[episodes[-1][-1]]['state']!=row['state']:episodes.append([])
                episodes[-1].append(i)
            if len(episodes)!=4 or any(len(e)<7 for e in episodes):raise ValueError('Original full seven-exposure reversal windows required')
            for episode in episodes[1:]:
                idx=episode[:7];state=rows[idx[0]]['state']
                optimal=int((shell==3 and state==1) or (shell==4 and state==2))
                person_human.append(np.asarray([rows[i]['action']==optimal for i in idx],float))
                for c in model_curves:
                    person_models[c].append((free[conditions.index(c),:,pi][:,idx]==optimal).mean(axis=0))
        human_curves.append(np.mean(person_human,axis=0))
        for c in model_curves:model_curves[c].append(np.mean(person_models[c],axis=0))
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(range(1,8),np.mean(human_curves,axis=0),'-o',label='Human recorded choices',color='#181818',linewidth=2.5)
    for c,curves in model_curves.items():ax.plot(range(1,8),np.mean(curves,axis=0),'-o',label=c,alpha=.85)
    ax.set(ylim=(0,1),xlabel='Exposure to the same shell after reversal',ylabel='Optimal action proportion',
           title='Descriptive reversal curves on the same recorded traces\nSix episodes averaged within each person; no new hypothesis test')
    ax.legend(ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'reversal_curves.png',dpi=150);plt.close(fig)
    record={'scope':'presentation_only_from_independently_verified_frozen_summary','summary_sha256':sha(result_dir/'summary.json'),
            'verification_sha256':sha(result_dir/'verification.json'),'source_registration_sha256':sha(result_dir/'registration.json'),
            'plot_source_sha256':sha(Path(__file__)),'figures':{p.name:sha(p) for p in sorted(out.iterdir())}}
    with (out/'provenance.json').open('x',encoding='utf8',newline='\n') as f:json.dump(record,f,indent=2);f.write('\n')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();plot(a.result_dir,a.out,a.cache);print(json.dumps({'status':'rendered','out':str(a.out)}))
