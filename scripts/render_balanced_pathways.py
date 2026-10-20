"""Render independently verified saved-path accounting to a fresh directory."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();R=args.root.resolve();out=args.out.resolve();folder=R/'results/balanced_pathways_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(folder/'summary.json');v=read(folder/'verification.json')
assert v['status']=='passed' and v['summary_sha256']==sha(folder/'summary.json')
with np.load(folder/'arrays.npz',allow_pickle=False) as f:a={name:f[name] for name in f.files}
out.mkdir(exist_ok=False);c=s['contrasts'][0];means=c['means'];plotted={}
phase=np.asarray(means['phase_category_endpoint_difference']).reshape(-1)*100
labels=['Partial\nown','Partial\nknown other','Partial\nunknown other','Full\nown','Full\nknown other','Full\nunknown other']
fig,ax=plt.subplots(figsize=(12,5));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.8)
bars=ax.bar(range(6),phase,color=['#ad7348']*3+['#087e73']*3,width=.65)
ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(6),labels)
ax.set_ylabel('Contribution to final expected accuracy difference (pp)')
ax.bar_label(bars,labels=[f'{x:+.3f}' for x in phase],padding=3,fontsize=9);ax.margins(y=.22)
fig.suptitle('Primary real / raw: where the small final gain comes from\nBalanced minus selective; phase uses memory BEFORE current feedback',y=.98,fontsize=12)
fig.savefig(out/'phase-accounting.png',dpi=150);plt.close(fig)
plotted['phase']={'phase_order':s['phase_order'],'category_order':s['category_order'],
    'endpoint_contribution_difference_pp':phase.reshape(2,3).tolist(),
    'per_seed_endpoint_contribution_difference': [r['phase_category_endpoint_difference'] for r in c['seeds']]}
gain=np.asarray([*means['both_Go_probability_shapley'],*means['action_case_endpoint_difference'][1:3]])*100
harm=np.asarray([*means['both_Go_shared_unknown_harm_shapley'],means['unpaired_Go_shared_unknown_harm_difference'],
    *means['unknown_membership_damage_difference'][1:]])*100
fig,axes=plt.subplots(1,2,figsize=(14,5.7));fig.subplots_adjust(left=.07,right=.98,bottom=.3,top=.8,wspace=.32)
for ax,values,names in [(axes[0],gain,['Pre-state','Scalar','Direction','Selective-only Go','Balanced-only Go']),
    (axes[1],harm,['Pre-state','Scalar','Direction','Unpaired common','Selective-only unknown','Balanced-only unknown'])]:
    bars=ax.bar(range(len(values)),values,color=['#bd6550' if x<0 else '#087e73' for x in values],width=.65)
    ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(len(values)),names,rotation=24,ha='right',fontsize=9)
    ax.bar_label(bars,labels=[f'{x:+.4g}' for x in values],padding=3,fontsize=8);ax.margins(y=.25)
axes[0].set_ylabel('Final expected accuracy difference (pp)')
axes[1].set_ylabel('Accumulated unknown harm difference (mass x 100)')
fig.suptitle('Recorded one-step accounting: state, scalar, direction and actual-Go remainders\nSymmetric allocation of stored cells; no alternative learning trajectory',y=.98,fontsize=12)
fig.savefig(out/'bridge-accounting.png',dpi=150);plt.close(fig)
plotted['bridge']={'endpoint_components_pp':gain.tolist(),'unknown_harm_components_mass_x100':harm.tolist()}
mask=a['objective_eligible'][0,0];obj=a['objective_values'][0,0][mask]
x=obj[:,1]-obj[:,0];q=a['objective_matched_probability_gain'][0,0][mask]*100
sources=a['schedules'][mask];positive=sources<2
fig,ax=plt.subplots(figsize=(9,5));fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.8)
for use,label,color in [(positive,'Source rewards Go','#087e73'),(~positive,'Source penalizes Go','#ac6a40')]:
    ax.scatter(x[use],q[use],s=13,alpha=.3,label=label,color=color,edgecolors='none')
ax.axhline(0,color='#555555',lw=.8);ax.set_xlabel('Improvement in the fixed geometric objective per unit direction')
ax.set_ylabel('Matched one-step expected accuracy gain (pp)')
ax.legend(frameon=False,fontsize=9)
fig.suptitle('Primary full-known, both-Go: objective and probability gains\nSame selective pre-state and scalar; recorded balanced direction substituted once',y=.98,fontsize=12)
fig.savefig(out/'objective-gain.png',dpi=150);plt.close(fig)
plotted['objective']={'objective_gain':x.tolist(),'matched_probability_gain_pp':q.tolist(),'sources':sources.tolist()}
plotted['all_contrasts']=[{'topology':z['topology'],'representation':z['representation'],'endpoint_difference':z['means']['endpoint_difference'],
    'unknown_damage_difference':z['means']['unknown_damage_difference']} for z in s['contrasts']]
with (out/'plotted_values.json').open('x',encoding='utf8',newline='\n') as h:json.dump(plotted,h,indent=2);h.write('\n')
proof={'study_id':'balanced_pathways_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'summary_sha256':sha(folder/'summary.json'),'arrays_sha256':sha(folder/'arrays.npz'),
    'verification_sha256':sha(folder/'verification.json'),'render_script':'scripts/render_balanced_pathways.py',
    'render_script_sha256':sha(Path(__file__)),'plotted_values_sha256':sha(out/'plotted_values.json'),
    'figure_sha256':{p.name:sha(p) for p in out.glob('*.png')},'new_training_trials':0,'new_random_draws':0,'new_optimization_calls':0}
with (out/'provenance.json').open('x',encoding='utf8',newline='\n') as h:json.dump(proof,h,indent=2);h.write('\n')
print(json.dumps({'status':'rendered','figures':len(proof['figure_sha256'])}))
