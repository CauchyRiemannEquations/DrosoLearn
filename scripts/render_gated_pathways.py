"""Render verified saved-path accounting to a fresh folder, without learning."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();R=args.root.resolve();out=args.out.resolve();folder=R/'results/gated_pathways_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(folder/'verification.json');s=read(folder/'summary.json')
assert v['status']=='passed' and v['summary_sha256']==sha(folder/'summary.json')
out.mkdir(exist_ok=False)
primary=next(c for c in s['contrasts'] if c['role']=='primary')
values=100*np.array(primary['means']['phase_category_endpoint_difference'])
records={'primary_phase_category_difference_pp':values.tolist(),'primary_methods':[],'all_contrasts':[]}
fig,ax=plt.subplots(figsize=(11,4.8));fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.84)
labels=['Pre: own','Pre: known other','Pre: unknown other','Post: own','Post: known other','Post: unknown other']
flat=values.ravel();x=np.arange(6);colors=np.where(flat>=0,'#087e73','#b64c4c')
ax.bar(x,flat,color=colors);ax.axhline(0,color='#555555',lw=.8)
for i,value in enumerate(flat):ax.annotate(f'{value:+.3f}',(i,value),xytext=(0,5 if value>=0 else -5),
    textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_ylabel('Contribution to final difference (percentage points)')
ax.margins(y=.2);fig.suptitle('Gated minus always-balanced: common gate time, own knowledge categories',fontsize=12)
fig.savefig(out/'phase-accounting.png',dpi=150);plt.close(fig)
methods=['shared','isolated','feedback_selective','balanced','gated_balanced']
names=['Shared','Isolated','Selective','Balanced','Gated'];primary_families=[]
for method in methods:
    f=next(f for f in s['families'] if f['topology']=='real' and f['representation']=='raw' and f['method']==method)
    primary_families.append(f)
    records['primary_methods'].append({'method':method,
        'net_endpoint_contribution_pp':(100*np.array(f['mean_phase_category_endpoint_contributions'])).tolist(),
        'positive_endpoint_contribution_pp':(25*np.array(f['mean_phase_category_positive_probability_mass'])).tolist(),
        'negative_endpoint_contribution_pp':(25*np.array(f['mean_phase_category_negative_probability_mass'])).tolist(),
        'phase_costs':f['mean_phase_costs']})
fig,axes=plt.subplots(1,2,figsize=(13,5));fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.78,wspace=.25)
for ph,ax in enumerate(axes):
    net=np.array([f['mean_phase_category_endpoint_contributions'][ph] for f in primary_families])*100
    pos=np.array([f['mean_phase_category_positive_probability_mass'][ph] for f in primary_families])*25
    neg=np.array([f['mean_phase_category_negative_probability_mass'][ph] for f in primary_families])*25
    components=[net[:,0],pos[:,1],pos[:,2],-neg[:,1],-neg[:,2]]
    labels2=['Own net','Known-other help','Unknown-other help','Known-other harm','Unknown-other harm']
    colors2=['#7550a0','#087e73','#8dc8b9','#b64c4c','#e69c82'];up=np.zeros(5);down=np.zeros(5)
    for vals,label,color in zip(components,labels2,colors2):
        ax.bar(np.arange(5),np.maximum(vals,0),bottom=up,label=label,color=color)
        ax.bar(np.arange(5),np.minimum(vals,0),bottom=down,color=color)
        up+=np.maximum(vals,0);down+=np.minimum(vals,0)
    ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(np.arange(5),names,rotation=20)
    ax.set_title(['Before reference gate','After reference gate'][ph]);ax.set_ylabel('Contribution to learning gain (percentage points)')
    low,high=float(down.min()),float(up.max());padding=max((high-low)*.08,.02)
    ax.set_ylim(low-padding,high+padding)
handles,legendlabels=axes[0].get_legend_handles_labels()
fig.legend(handles,legendlabels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
fig.suptitle('Primary real / raw: own correction, helpful spread, and harmful spread',fontsize=13,y=.99)
fig.savefig(out/'help-and-harm.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(13,8));fig.subplots_adjust(left=.07,right=.98,bottom=.1,top=.83,hspace=.42,wspace=.25)
controls=['balanced','isolated','feedback_selective','shared'];controlnames=['Balanced','Isolated','Selective','Shared']
for ti,top in enumerate(['real','rewired','ann']):
    for ri,rep in enumerate(['raw','identity']):
        ax=axes[ri,ti];vals=[]
        for control in controls:
            c=next(c for c in s['contrasts'] if c['topology']==top and c['representation']==rep and c['control']==control)
            d=100*np.array(c['means']['phase_endpoint_difference']);vals.append(d)
            records['all_contrasts'].append({'topology':top,'representation':rep,'control':control,'pre_post_difference_pp':d.tolist()})
        vals=np.array(vals);up=np.zeros(4);down=np.zeros(4)
        for ph,color in enumerate(['#7550a0','#087e73']):
            ax.bar(np.arange(4),np.maximum(vals[:,ph],0),bottom=up,color=color,label=['Pre','Post'][ph])
            ax.bar(np.arange(4),np.minimum(vals[:,ph],0),bottom=down,color=color)
            up+=np.maximum(vals[:,ph],0);down+=np.minimum(vals[:,ph],0)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(np.arange(4),controlnames,rotation=20)
        ax.set_title(top+' / '+rep);ax.set_ylabel('Gated minus control (percentage points)')
handles,legendlabels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legendlabels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('All predeclared controls: common gated switch; recorded paths only',y=.99,fontsize=13)
fig.savefig(out/'all-contrasts.png',dpi=150);plt.close(fig)
with (out/'plotted_values.json').open('x',encoding='utf8',newline='\n') as h:json.dump(records,h,indent=2);h.write('\n')
proof={'study_id':'gated_pathways_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'verified_arrays_only':True,'new_training_trials':0,'new_parameter_selection':False,
    'arrays_sha256':sha(folder/'arrays.npz'),'summary_sha256':sha(folder/'summary.json'),
    'verification_sha256':sha(folder/'verification.json'),'render_script_sha256':sha(Path(__file__)),
    'plotted_values_sha256':sha(out/'plotted_values.json'),
    'figure_sha256':{p.name:sha(p) for p in out.glob('*.png')}}
with (out/'provenance.json').open('x',encoding='utf8',newline='\n') as h:json.dump(proof,h,indent=2);h.write('\n')
print(json.dumps({'status':'rendered','figures':len(proof['figure_sha256'])}))
