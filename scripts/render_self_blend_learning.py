"""Render a verified fixed self-blend cohort; no model, optimizer or random calls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

P=argparse.ArgumentParser(description=__doc__);P.add_argument('--out',type=Path,required=True);args=P.parse_args()
R=Path(__file__).resolve().parents[1];folder=R/'results/self_blend_learning_v1';out=args.out.resolve()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(folder/'verification.json');s=read(folder/'summary.json')
assert v['status']=='passed' and v['summary_sha256']==sha(folder/'summary.json')
with np.load(folder/'arrays.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
out.mkdir(exist_ok=False)
methods=s['methods'];labels=['Shared','Isolated','Selective','Balanced','Gated','Self blend']
colors=['#777777','#7550a0','#cc8500','#3e78ae','#087e73','#c04447'];seeds=arrays['model_seeds'];grid=arrays['checkpoints']
score=np.where(arrays['targets']==1,arrays['checkpoint_probabilities'],1-arrays['checkpoint_probabilities']).mean(-1)
records={'learning':[],'conditions':[],'direction':[],'factual_post_pathways':[]}
fig,axes=plt.subplots(2,3,figsize=(14,8))
fig.subplots_adjust(left=.065,right=.985,bottom=.07,top=.85,hspace=.34,wspace=.22)
for ti,top in enumerate(['real','rewired','ann']):
    for ri,rep in enumerate(['raw','identity']):
        ax=axes[ri,ti]
        for mi,method in enumerate(methods):
            mean=score[ti,ri,mi].mean(0);ax.plot(grid,mean,'o-',color=colors[mi],label=labels[mi],markersize=3)
            records['learning'].append({'topology':top,'representation':rep,'method':method,'checkpoints':grid.tolist(),'mean_expected_accuracy':mean.tolist()})
        ax.axhline(.8,color='#aaaaaa',ls=':',lw=.8);ax.set(title=top+' / '+rep,ylim=(.45,1.02),xlabel='Actual learning trials')
        ax.yaxis.set_major_formatter(PercentFormatter(1.))
        if ti==0:ax.set_ylabel('Expected correct choice probability')
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.945),ncol=6,frameon=False)
fig.suptitle('Fixed 3:1 self-blend: all predeclared conditions',y=.99,fontsize=14)
fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for ti,top in enumerate(['real','rewired','ann']):
    for ri,rep in enumerate(['raw','identity']):
        ax=axes[ri,ti]
        for mi,method in enumerate(methods):
            vals=score[ti,ri,mi,:,-1];damage=arrays['trial_unknown_other_negative_correct_probability_sum'][ti,ri,mi].sum(-1)
            path=arrays['trial_cumulative_head_path_l2'][ti,ri,mi,:,-1]
            ax.bar(mi,vals.mean(),color=colors[mi],alpha=.75);ax.scatter(mi+np.linspace(-.15,.15,10),vals,s=10,color=colors[mi],edgecolors='white',zorder=3)
            records['conditions'].append({'topology':top,'representation':rep,'method':method,'seeds':seeds.tolist(),
                'final_expected_accuracy_by_seed':vals.tolist(),'unknown_negative_probability_mass_by_seed':damage.tolist(),'actual_head_path_l2_by_seed':path.tolist()})
        ax.set(title=top+' / '+rep,ylim=(0,1.02));ax.set_xticks(range(6),labels,rotation=25)
        ax.yaxis.set_major_formatter(PercentFormatter(1.))
        if ti==0:ax.set_ylabel('Final expected correct choice probability')
fig.suptitle('All 36 conditions: ten computational seeds each; no seed discarded')
fig.savefig(out/'all-conditions.png',dpi=150);plt.close(fig)
# Same factual mixed-Go events: ingredient influence is not another policy outcome.
means=[]
for si,seed in enumerate(seeds):
    at=(0,0,5,si);mask=arrays['trial_blend_applicable'][at];src=arrays['schedules'][si][mask]
    assert mask.any(), 'Report a missing factual mixed-Go subset explicitly instead of plotting NaN'
    d=arrays['g_real_raw_unit_rows'][si];k=arrays['trial_known_after'][at][mask]
    record={'seed':int(seed),'actual_blend_Go_trials':int(mask.sum())}
    for name,key in [('isolated','trial_blend_isolated_directions'),('balanced','trial_blend_balanced_directions'),('blend','trial_directions')]:
        inf=arrays[key][at][mask]@d.T;aligned=k[np.arange(len(k)),src,None]*k*inf;other=np.arange(4)[None,:]!=src[:,None]
        record[name+'_own_unit_influence_mean']=float(inf[np.arange(len(k)),src].mean())
        record[name+'_other_aligned_unit_influence_mean']=float(aligned[other].mean())
    records['direction'].append(record)
for mi,method in enumerate(methods):
    own=[];other=[]
    for si in range(10):
        # Descriptive accounting at the fixed gate rule; not a preregistered primary hypothesis.
        start=int(arrays['first_blend_mode_trials'][0,0,5,si])-1
        delta=arrays['trial_correct_probability_delta'][0,0,mi,si,start:];src=arrays['schedules'][si,start:]
        own.append(float(delta[np.arange(len(src)),src].sum()/4));other.append(float((delta.sum()-delta[np.arange(len(src)),src].sum())/4))
    records['factual_post_pathways'].append({'method':method,'own_endpoint_contribution_by_seed':own,'other_endpoint_contribution_by_seed':other})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(3)
for j,(key,label) in enumerate([('own_unit_influence_mean','Own input'),('other_aligned_unit_influence_mean','Other inputs')]):
    values=[np.mean([r[name+'_'+key] for r in records['direction']]) for name in ('isolated','balanced','blend')]
    axes[0].bar(x+(j-.5)*.32,values,width=.32,label=label)
axes[0].set_xticks(x,['Isolated ingredient','Balanced ingredient','Actual blend']);axes[0].set_ylabel('Mean unit-score influence');axes[0].legend()
for mi in (1,4,5):
    r=records['factual_post_pathways'][mi];axes[1].bar(mi,np.mean(r['own_endpoint_contribution_by_seed'])*100,color=colors[mi])
    axes[1].bar(mi,np.mean(r['other_endpoint_contribution_by_seed'])*100,bottom=np.mean(r['own_endpoint_contribution_by_seed'])*100,color=colors[mi],alpha=.35)
axes[1].set_xticks([1,4,5],['Isolated','Gated','Self blend']);axes[1].set_ylabel('Post-switch endpoint contribution (pp)\nSolid=own; light=other, descriptive accounting')
fig.suptitle('Primary real / raw: direction ingredients and actual post-switch learning\nLeft uses the same mixed-Go states; right uses each policy factual choices')
fig.savefig(out/'unit-direction-and-pathways.png',dpi=150);plt.close(fig)
(out/'plotted_values.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8',newline='\n')
proof={'study_id':'self_blend_learning_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'verified_arrays_only':True,
    'new_training_trials':0,'new_parameter_selection':False,'arrays_sha256':sha(folder/'arrays.npz'),'summary_sha256':sha(folder/'summary.json'),
    'verification_sha256':sha(folder/'verification.json'),'render_script_sha256':sha(Path(__file__)),
    'plotted_values_sha256':sha(out/'plotted_values.json'),'figure_sha256':{p.name:sha(p) for p in out.glob('*.png')}}
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps({'status':'rendered','figures':len(proof['figure_sha256'])}))
