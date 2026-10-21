"""Render only the saved, independently checked descriptive pathways result."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/objective_pathways_v1'
OUT=ROOT/'results/objective_pathways_figures_v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=150);plt.close()


def main():
    s=json.loads((RESULT/'summary.json').read_text(encoding='utf8'))
    v=json.loads((RESULT/'verification.json').read_text(encoding='utf8'))
    assert s['status']=='completed' and v['status']=='passed' and not OUT.exists()
    OUT.mkdir();plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['U - B','P - B','P - U'];colors=['#4477AA','#EE7733','#228833']
    primary=s['families'][:3]
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    for group in range(2):
        for ci,f in enumerate(primary):
            vals=np.array(f['means']['component_endpoint_contributions'])[group]*100
            ax[group].plot(np.arange(6),vals,'o-',color=colors[ci],label=names[ci])
        ax[group].axhline(0,color='black',lw=.7)
        ax[group].set_xticks(np.arange(6),['State','Scalar','Direction','B/control\nonly Go','New only\nGo','Neither Go'])
        ax[group].set_ylabel('Contribution to endpoint difference (percentage points)')
        ax[group].set_title(['Current input (own)','Other three training inputs'][group]);ax[group].legend()
    fig.suptitle('Primary real/raw: symmetric accounting of actual saved paths')
    save('factor-accounting.png')
    fig,ax=plt.subplots(figsize=(12,5))
    labels=[]
    for ti,top in enumerate(('real','rewired','ann')):
        for ri,rep in enumerate(('raw','identity')):
            i=ti*2+ri;labels.append(top+'/'+rep)
            for ci in range(3):
                f=s['families'][i*3+ci];x=i+(ci-1)*.22
                vals=[r['final_endpoint_difference']*100 for r in f['seeds']]
                ax.scatter([x]*len(vals),vals,c=colors[ci],alpha=.4,s=20)
                ax.plot(x,np.mean(vals),'_',c=colors[ci],markersize=18,markeredgewidth=3,label=names[ci] if i==0 else None)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(6),labels);ax.set_ylabel('Final expected accuracy difference (percentage points)')
    ax.set_title('All 18 comparisons: all 10 computational seeds, equal-seed means');ax.legend();save('all-contrasts.png')
    primary_paths=[f for f in s['actual_paths'] if f['topology']=='real' and f['representation']=='raw'
        and f['method'] in ('gated_balanced','gated_uniform','gated_probability')]
    values=[]
    for f in primary_paths:
        records=[r['phases']['post'] for r in f['seeds']]
        values.append([np.mean([r[k] for r in records]) for k in ('Go','reward_positive','reward_negative','head_path_l2')])
    values=np.array(values);fig,ax=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(3)
    ax[0].bar(x-.15,values[:,1],.3,label='Positive obtained reward',color='#228833')
    ax[0].bar(x+.15,values[:,2],.3,label='Negative obtained reward',color='#CC6677')
    ax[0].set_xticks(x,['B','U','P']);ax[0].set_ylabel('Actual Go events per seed (mean)')
    ax[0].set_title('Post-gate obtained rewards');ax[0].set_ylim(0,140);ax[0].legend()
    ax[1].bar(x,values[:,3],color=['#999999',colors[0],colors[1]])
    ax[1].set_xticks(x,['B','U','P']);ax[1].set_ylabel('Realized output-weight movement length')
    ax[1].set_title('Post-gate movement: same ceilings, different use');save('choice-reward-paths.png')
    plotted={'comparison_order':s['comparison_order'],'component_order':s['component_order'],
        'primary_component_endpoint_contributions':[f['means']['component_endpoint_contributions'] for f in primary],
        'all_final_endpoint_differences':[[r['final_endpoint_difference'] for r in f['seeds']] for f in s['families']],
        'primary_post_reward_and_movement_means':values.tolist(),
        'reward_and_movement_column_order':['Go','reward_positive','reward_negative','head_path_l2']}
    with (OUT/'plotted_values.json').open('x',encoding='utf8',newline='\n') as h:json.dump(plotted,h,indent=2);h.write('\n')
    bound=[RESULT/'summary.json',RESULT/'verification.json',RESULT/'arrays.npz',Path(__file__).resolve()]
    provenance={'status':'passed','rendered_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in bound},
        'artifact_sha256':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        'new_learning_or_solver_calls':0,'primary_real_raw_only_for_factor_and_reward_figures':True}
    with (OUT/'provenance.json').open('x',encoding='utf8',newline='\n') as h:json.dump(provenance,h,indent=2);h.write('\n')
    print('Rendered three saved-only accounting figures with exact plotted values.')


if __name__=='__main__':main()
