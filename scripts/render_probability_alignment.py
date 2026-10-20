"""Render verified fixed-state effects; no new learning or optimization."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r=ROOT/'results/probability_alignment_v1'
    s=json.loads((r/'summary.json').read_text(encoding='utf8'))
    v=json.loads((r/'verification.json').read_text(encoding='utf8'))
    assert v['status']=='passed'
    out=ROOT/'results/probability_alignment_figures_v1';out.mkdir(exist_ok=False)
    fs=s['families'];primary=[f for f in fs if f['topology']=='real' and f['representation']=='raw']
    assert len(primary)==5
    values={'study_id':s['study_id'],'unit':'percentage points per fixed-state eligible single update',
        'equal_seed_means':True,'learned_trajectory_or_final_accuracy':False,
        'primary_families':primary,'all_families':fs,
        'finite_alignment_summary':s['finite_alignment_summary']}
    (out/'plotted_values.json').write_text(json.dumps(values,indent=2)+'\n',encoding='utf8',newline='\n')
    labels=['B legacy','I isolated','W 3:1','U uniform','P probability']
    means=np.asarray([f['means']['exact_Q_mean'] for f in primary])*100.
    fig,ax=plt.subplots(figsize=(9,5.6));x=np.arange(5)
    ax.bar(x,means[:,0],label='Own input / 4',color='#4078a8')
    ax.bar(x,means[:,1],bottom=means[:,0],label='Other three / 4',color='#72b59a')
    ax.set_xticks(x,labels);ax.set_ylabel('Mean exact gain (percentage points / saved update)')
    ax.set_title('Same saved state and actual step: four-input probability gain\nreal/raw; 10 computational seeds with equal weight')
    ax.axhline(0,color='black',linewidth=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'primary-gains.png',dpi=160);plt.close(fig)
    p=primary[4];seed=[r['seed'] for r in p['seeds']]
    delta=np.asarray([r['difference_from_U_mean'] for r in p['seeds']])*100.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].plot(seed,delta[:,0],'o-',label='Own / 4')
    axes[0].plot(seed,delta[:,1],'o-',label='Other three / 4')
    axes[1].plot(seed,delta[:,2],'o-',color='#8658a3',label='Total / 4')
    for ax in axes:
        ax.set_xlabel('Computational seed');ax.set_ylabel('P minus U exact gain (pp / saved update)')
        ax.axhline(0,color='black',linewidth=.6);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
        ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('State weighting versus the uniform four-input objective\nCounterfactual single steps, not a new learning curve')
    fig.tight_layout();fig.savefig(out/'state-weighting-by-seed.png',dpi=160);plt.close(fig)
    cond=[(t,r) for t in ('real','rewired','ann') for r in ('raw','identity')]
    lookup={(f['topology'],f['representation'],f['candidate']):f for f in fs}
    data=[]
    for top,rep in cond:
        u=lookup[top,rep,'uniform_score']['means'];p=lookup[top,rep,'state_probability']['means']
        data.append([u['difference_from_B_mean'][2],p['difference_from_U_mean'][2]])
    data=np.asarray(data)*100.;fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    for j,ax in enumerate(axes):
        ax.bar(np.arange(6),data[:,j],color=['#4078a8','#5b8db6','#72b59a','#91c8b3','#8658a3','#b199c3'])
        ax.set_xticks(np.arange(6),[t+'\n'+r for t,r in cond]);ax.axhline(0,color='black',linewidth=.6)
        ax.set_ylabel('Mean exact total gain difference (pp / saved update)')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.grid(axis='y',alpha=.2)
    axes[0].set_title('U minus B: change the objective scope')
    axes[1].set_title('P minus U: add current-state sensitivity')
    fig.suptitle('All six topology / representation conditions; equal seed means')
    fig.tight_layout();fig.savefig(out/'all-conditions.png',dpi=160);plt.close(fig)
    provenance={'study_id':s['study_id'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'renderer':'scripts/render_probability_alignment.py','renderer_sha256':sha(Path(__file__)),
        'input_sha256':{('results/probability_alignment_v1/'+name):sha(r/name)
            for name in ('summary.json','verification.json')},
        'output_sha256':{q.name:sha(q) for q in sorted(out.iterdir())},
        'new_learning_calls':0,'new_optimization_calls':0}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Three static figures and exact plotted values written.')


if __name__=='__main__':main()
