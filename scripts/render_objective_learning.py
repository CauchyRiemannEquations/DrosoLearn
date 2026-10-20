"""Plot independently verified actual U/P learning; no fresh training or solves."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
LABELS=['Shared','Isolated','Selective','Balanced','Gated B','Gated W','Uniform U','Probability P']
COLORS=['#858585','#8056a0','#ca8a12','#497eae','#0b8074','#c45a69','#144dac','#d64619']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder=ROOT/'results/objective_learning_v1';out=ROOT/'results/objective_learning_figures_v1'
    s=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    v=json.loads((folder/'verification.json').read_text(encoding='utf8'))
    assert v['status']=='passed' and v['summary_sha256']==sha(folder/'summary.json')
    with np.load(folder/'arrays.npz',allow_pickle=False) as h:a={k:h[k] for k in h.files}
    out.mkdir(exist_ok=False);methods=s['methods'];grid=a['checkpoints'];seeds=a['model_seeds']
    correct=np.where(a['targets']==1,a['checkpoint_probabilities'],1-a['checkpoint_probabilities'])
    scores=correct.mean(-1);records={'learning':[],'conditions':[],'protection':[],'costs':[]}
    fig,axes=plt.subplots(2,3,figsize=(15,8.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.075,top=.80,hspace=.34,wspace=.23)
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            ax=axes[ri,ti]
            for mi,method in enumerate(methods):
                mean=scores[ti,ri,mi].mean(0)
                ax.plot(grid,mean,'o-',color=COLORS[mi],label=LABELS[mi],markersize=3,linewidth=1.5 if mi>=6 else 1.)
                records['learning'].append({'topology':top,'representation':rep,'method':method,
                    'checkpoints':grid.tolist(),'mean_expected_accuracy':mean.tolist()})
            ax.axhline(.8,color='gray',ls=':',lw=.7);ax.set(title=top+' / '+rep,ylim=(.45,1.02),xlabel='Actual learning trials')
            ax.yaxis.set_major_formatter(PercentFormatter(1.))
            if ti==0:ax.set_ylabel('Expected correct choice probability')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.suptitle('Actual learning with whole-input U and current-state P objectives\nSame 400-trial budget and maximum single-step norm; realized costs can differ',y=.995)
    fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            ax=axes[ri,ti]
            for mi,method in enumerate(methods):
                vals=scores[ti,ri,mi,:,-1]
                ax.bar(mi,vals.mean(),color=COLORS[mi],alpha=.7)
                ax.scatter(mi+np.linspace(-.16,.16,len(seeds)),vals,s=10,color=COLORS[mi],edgecolors='white',zorder=3)
                damage=a['trial_unknown_other_negative_correct_probability_sum'][ti,ri,mi].sum(-1)
                path=a['trial_cumulative_head_path_l2'][ti,ri,mi,:,-1]
                records['conditions'].append({'topology':top,'representation':rep,'method':method,'seeds':seeds.tolist(),
                    'final_expected_accuracy_by_seed':vals.tolist(),'unknown_negative_probability_mass_by_seed':damage.tolist(),
                    'actual_head_path_l2_by_seed':path.tolist()})
            ax.set(title=top+' / '+rep,ylim=(0,1.02));ax.set_xticks(range(8),LABELS,rotation=40,ha='right')
            ax.yaxis.set_major_formatter(PercentFormatter(1.))
            if ti==0:ax.set_ylabel('Final expected correct choice probability')
    fig.suptitle('All 48 predeclared conditions: ten computational seeds each; no seed discarded')
    fig.savefig(out/'all-conditions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.20,top=.77,wspace=.28)
    for mi,method in enumerate(methods):
        damage=a['trial_unknown_other_negative_correct_probability_sum'][0,0,mi]
        cumulative=np.concatenate((np.zeros(1),damage.mean(0).cumsum()))
        path=a['trial_cumulative_head_path_l2'][0,0,mi,:,-1]
        axes[0].plot(np.arange(len(cumulative)),cumulative*100,color=COLORS[mi],label=LABELS[mi])
        axes[1].bar(mi,path.mean(),color=COLORS[mi],alpha=.75)
        axes[1].scatter(mi+np.linspace(-.16,.16,len(seeds)),path,s=14,color=COLORS[mi],edgecolors='white',zorder=3)
        records['protection'].append({'method':method,'cumulative_unknown_negative_probability_mass_mean':cumulative.tolist()})
        records['costs'].append({'method':method,'actual_head_path_l2_by_seed':path.tolist()})
    axes[0].set(xlabel='Actual learning trials',ylabel='Cumulative unknown-other harm (pp sum)',title='Feedback-unobserved training inputs')
    axes[1].set(ylabel='Actual sum of output-head step norms',title='Realized movement, same ceiling 40')
    axes[1].set_xticks(range(8),LABELS,rotation=35,ha='right');axes[1].axhline(40,color='gray',ls=':',lw=.8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.suptitle('Primary real/raw: operational protection and actual cost',y=1.)
    fig.savefig(out/'protection-and-cost.png',dpi=150);plt.close(fig)
    (out/'plotted_values.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8',newline='\n')
    proof={'study_id':s['study_id'],'generated_at_utc':datetime.now(timezone.utc).isoformat(),'verified_arrays_only':True,
        'new_training_trials':0,'new_parameter_selection':False,'new_optimization_calls':0,
        'arrays_sha256':sha(folder/'arrays.npz'),'summary_sha256':sha(folder/'summary.json'),
        'verification_sha256':sha(folder/'verification.json'),'render_script_sha256':sha(Path(__file__)),
        'plotted_values_sha256':sha(out/'plotted_values.json'),'figure_sha256':{p.name:sha(p) for p in sorted(out.glob('*.png'))}}
    (out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Three actual-learning figures and exact plotted values written.')


if __name__=='__main__':main()
