"""Render independently verified post-gate factor accounting, without learning."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();R=args.root.resolve();out=args.out.resolve();folder=R/'results/gated_factors_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(folder/'verification.json');s=read(folder/'summary.json')
assert v['status']=='passed' and v['summary_sha256']==sha(folder/'summary.json')
out.mkdir(exist_ok=False);primary=next(f for f in s['families'] if f['role']=='primary')
means=primary['means'];values=100*np.array(means['component_endpoint_contributions'])
labels=['Output state','Update scalar','Direction','Control only Go','Gated only Go','Neither Go']
records={'primary_component_difference_pp':values.tolist(),
    'primary_positive_endpoint_allocations_pp':(100*np.array(means['component_positive_endpoint_allocations'])).tolist(),
    'primary_negative_endpoint_allocations_pp':(100*np.array(means['component_negative_endpoint_allocations'])).tolist(),
    'diagnostic_order':s['diagnostic_order'],'primary_diagnostic_means':means['diagnostic_means'],'all_contrasts':[]}
fig,axes=plt.subplots(1,2,figsize=(13,5));fig.subplots_adjust(left=.07,right=.98,bottom=.29,top=.81,wspace=.26)
for gi,ax in enumerate(axes):
    x=np.arange(6);colors=np.where(values[gi]>=0,'#087e73','#b64c4c')
    ax.bar(x,values[gi],color=colors);ax.axhline(0,color='#555555',lw=.8)
    for j,value in enumerate(values[gi]):ax.annotate(f'{value:+.3f}',(j,value),xytext=(0,5 if value>=0 else -5),
        textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
    ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_ylabel('Post-gate difference (percentage points)')
    ax.set_title(['Current input (own correction)','Other three trained inputs'][gi]);ax.margins(y=.22)
fig.suptitle('Gated minus isolated: symmetric state / scalar / direction allocation\nOne-sided actual actions are accounted for separately',fontsize=12)
fig.savefig(out/'factor-accounting.png',dpi=150);plt.close(fig)
diagnostic=np.array(means['diagnostic_means'],dtype=float)
titles=['Aligned source logit before','Correct source probability','Logistic sensitivity value','Absolute update scalar',
    'Actual output step length','Own unit-direction influence','Other aligned influence mean','Direction length']
fig,axes=plt.subplots(2,4,figsize=(14,7));fig.subplots_adjust(left=.06,right=.98,bottom=.09,top=.85,hspace=.4,wspace=.4)
for di,ax in enumerate(axes.ravel()):
    vals=diagnostic[:,di];ax.bar([0,1],vals,color=['#7550a0','#087e73'],alpha=.75)
    for pi in range(2):
        points=[r['diagnostic_means'][pi][di] for r in primary['seeds'] if r['diagnostic_means'][pi][di] is not None]
        ax.scatter(pi+np.linspace(-.15,.15,len(points)),points,color=['#7550a0','#087e73'][pi],edgecolor='white',s=20,zorder=3)
    ax.set_xticks([0,1],['Isolated','Gated']);ax.set_title(titles[di],fontsize=10);ax.axhline(0,color='#777777',lw=.6)
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
fig.suptitle('Factual diagnostics on paired actual Go trials only\nBars = equal-seed means; dots = computational seed means; descriptive, not isolated causal effects',fontsize=12)
fig.savefig(out/'paired-diagnostics.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(13,8));fig.subplots_adjust(left=.07,right=.98,bottom=.11,top=.83,hspace=.4,wspace=.27)
controls=['isolated','balanced','feedback_selective','shared'];names=['Isolated','Balanced','Selective','Shared']
for ti,top in enumerate(['real','rewired','ann']):
    for ri,rep in enumerate(['raw','identity']):
        ax=axes[ri,ti];vals=[]
        for control in controls:
            f=next(f for f in s['families'] if f['topology']==top and f['representation']==rep and f['control']==control)
            row=100*np.array(f['means']['factual_post_endpoint_difference']);vals.append(row)
            records['all_contrasts'].append({'topology':top,'representation':rep,'control':control,'own_other_post_difference_pp':row.tolist()})
        vals=np.array(vals);up=np.zeros(4);down=np.zeros(4)
        for gi,color in enumerate(['#7550a0','#087e73']):
            ax.bar(np.arange(4),np.maximum(vals[:,gi],0),bottom=up,color=color,label=['Own','Other'][gi])
            ax.bar(np.arange(4),np.minimum(vals[:,gi],0),bottom=down,color=color)
            up+=np.maximum(vals[:,gi],0);down+=np.minimum(vals[:,gi],0)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(np.arange(4),names,rotation=20)
        ax.set_title(top+' / '+rep);ax.set_ylabel('Post-gate difference (percentage points)')
        low,high=float(down.min()),float(up.max());pad=max((high-low)*.08,.02);ax.set_ylim(low-pad,high+pad)
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('All recorded post-gate comparisons: own versus other-input contributions',fontsize=13,y=.99)
fig.savefig(out/'all-contrasts.png',dpi=150);plt.close(fig)
with (out/'plotted_values.json').open('x',encoding='utf8',newline='\n') as h:json.dump(records,h,indent=2);h.write('\n')
proof={'study_id':'gated_factors_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
    'verified_arrays_only':True,'new_training_trials':0,'new_parameter_selection':False,
    'arrays_sha256':sha(folder/'arrays.npz'),'summary_sha256':sha(folder/'summary.json'),
    'verification_sha256':sha(folder/'verification.json'),'render_script_sha256':sha(Path(__file__)),
    'plotted_values_sha256':sha(out/'plotted_values.json'),'figure_sha256':{p.name:sha(p) for p in out.glob('*.png')}}
with (out/'provenance.json').open('x',encoding='utf8',newline='\n') as h:json.dump(proof,h,indent=2);h.write('\n')
print(json.dumps({'status':'rendered','figures':len(proof['figure_sha256'])}))
