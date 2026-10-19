from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import argparse
parser=argparse.ArgumentParser(description='Render verified balanced-learning results to a fresh directory; no training.')
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
R=args.root.resolve()
folder=R/'results/balanced_learning_v1';out=args.out.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
verification=read(folder/'verification.json');summary=read(folder/'summary.json')
assert verification['status']=='passed' and verification['summary_sha256']==sha(folder/'summary.json')
out.mkdir(exist_ok=False)
with np.load(folder/'arrays.npz',allow_pickle=False) as a:arrays={key:a[key] for key in a.files}
targets=arrays['targets'];grid=arrays['checkpoints'];p=arrays['checkpoint_probabilities']
score=np.where(targets==1,p,1-p).mean(-1)
methods=['shared','isolated','feedback_selective','balanced'];labels=['Shared','Isolated','Selective','Balanced']
colors=['#686868','#7550a0','#db8c20','#087e73'];records={'learning':[],'primary':[],'unknown_cumulative':[]}
fig,axes=plt.subplots(2,3,figsize=(14,8));fig.subplots_adjust(left=.065,right=.985,bottom=.08,top=.83,hspace=.4,wspace=.2)
for ti,top in enumerate(['real','rewired','ann']):
    for ri,rep in enumerate(['raw','identity']):
        ax=axes[ri,ti]
        for mi,method in enumerate(methods):
            mean=score[ti,ri,mi].mean(0)
            ax.plot(grid,mean,'o-',color=colors[mi],label=labels[mi],markersize=3)
            records['learning'].append({'topology':top,'representation':rep,'method':method,
                'checkpoints':grid.tolist(),'mean_expected_accuracy':mean.tolist()})
        ax.axhline(.8,color='#aaaaaa',lw=.8,ls=':');ax.set_ylim(.45,1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.));ax.set(title=top+' / '+rep,xlabel='Actual learning trials')
        if ti==0:ax.set_ylabel('Expected correct choice probability')
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,frameon=False)
fig.suptitle('Autonomous balanced learning: all predeclared conditions',y=.98,fontsize=14)
fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,5));fig.subplots_adjust(left=.06,right=.985,bottom=.2,top=.82,wspace=.3)
for mi,method in enumerate(methods):
    accuracy=score[0,0,mi,:,-1]
    damage=arrays['trial_unknown_other_negative_correct_probability_sum'][0,0,mi].sum(-1)
    path=arrays['trial_cumulative_head_path_l2'][0,0,mi,:,-1]
    for ax,values in zip(axes,[accuracy,damage,path]):
        ax.bar(mi,float(values.mean()),color=colors[mi],alpha=.75,width=.65)
        ax.scatter(mi+np.linspace(-.18,.18,len(values)),values,color=colors[mi],edgecolor='white',s=28,zorder=3)
    records['primary'].append({'method':method,'seeds':arrays['model_seeds'].tolist(),
        'final_expected_accuracy_by_seed':accuracy.tolist(),'unknown_negative_probability_mass_by_seed':damage.tolist(),
        'actual_head_path_l2_by_seed':path.tolist()})
for ax in axes:ax.set_xticks(range(4),labels,rotation=20)
axes[0].set(ylabel='Expected correct choice probability',ylim=(0,1.02));axes[0].yaxis.set_major_formatter(PercentFormatter(1.))
axes[1].set_ylabel('Unknown-other negative probability mass\nTotal over 400 trials; lower is better')
axes[2].set_ylabel('Actual output movement path (L2)')
fig.suptitle('Primary real / raw: accuracy, unknown damage, actual movement\nBars = ten-seed means; points = computational seeds',y=.98,fontsize=12)
fig.savefig(out/'primary-outcomes.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.8));fig.subplots_adjust(left=.1,right=.98,bottom=.14,top=.78)
for mi,method in enumerate(methods):
    cumulative=np.cumsum(arrays['trial_unknown_other_negative_correct_probability_sum'][0,0,mi],axis=-1).mean(0)
    ax.plot(np.arange(1,401),cumulative,color=colors[mi],label=labels[mi])
    records['unknown_cumulative'].append({'method':method,'trials':list(range(1,401)),
        'mean_cumulative_unknown_negative_probability_mass':cumulative.tolist()})
ax.set(xlabel='Actual learning trials',ylabel='Unknown-other negative probability mass',xlim=(1,400))
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=4,frameon=False)
fig.suptitle('Primary real / raw: harm while reward signs remain unobserved',y=.98,fontsize=12)
fig.savefig(out/'unknown-harm.png',dpi=150);plt.close(fig)
with (out/'plotted_values.json').open('x',encoding='utf8',newline='\n') as h:json.dump(records,h,indent=2);h.write('\n')
proof={'study_id':'balanced_learning_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'verified_arrays_only':True,'new_training_trials':0,'new_parameter_selection':False,
    'arrays_sha256':sha(folder/'arrays.npz'),'summary_sha256':sha(folder/'summary.json'),
    'verification_sha256':sha(folder/'verification.json'),'render_script_sha256':sha(Path(__file__)),
    'plotted_values_sha256':sha(out/'plotted_values.json'),
    'figure_sha256':{p.name:sha(p) for p in out.glob('*.png')}}
with (out/'provenance.json').open('x',encoding='utf8',newline='\n') as h:json.dump(proof,h,indent=2);h.write('\n')
print(json.dumps({'status':'rendered','figures':len(proof['figure_sha256'])}))
