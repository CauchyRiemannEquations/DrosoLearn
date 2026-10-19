"""Presentation-only layout repair; frozen analysis and initial PNGs are preserved."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as h:json.dump(value,h,ensure_ascii=False,indent=2);h.write('\n')


def main():
    folder=ROOT/'results/signed_spillover_v1';out=ROOT/'results/signed_spillover_figures_v1'
    verification=json.loads((folder/'verification.json').read_text());assert verification['status']=='passed'
    summary=json.loads((folder/'summary.json').read_text())
    initial=json.loads((out/'plotted_values.json').read_text())
    names=['probability-accounting-final.png','signed-static-influence-final.png']
    assert all(not (out/name).exists() for name in names)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.subplots_adjust(left=.065,right=.985,bottom=.08,top=.84,hspace=.4,wspace=.23)
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            data=next(x for x in initial['actual_cross_probability_variation'] if x['topology']==top and x['representation']==rep)
            rows=[x for x in summary['families'] if x['topology']==top and x['representation']==rep]
            for i,row in enumerate(rows):
                for field,metric,sign in [('helpful_pp','probability_helpful_sum',1),('harmful_signed_pp','probability_harmful_sum',-1),('net_pp','probability_net_sum',1)]:
                    assert abs(data[field][i]-sign*row['categories']['cross_all'][metric]['mean']/4*100)<1e-12
            ax=axes[ri,ti];x=np.arange(5)
            ax.bar(x-.12,data['helpful_pp'],.24,label='Helpful cross changes',color='#247ba0')
            ax.bar(x+.12,data['harmful_signed_pp'],.24,label='Harmful cross changes',color='#d1495b')
            ax.plot(x,data['net_pp'],'o-',label='Net cross contribution',color='#222222')
            ax.axhline(0,color='#666666',lw=.7);ax.set_xticks(x,['0','.25','.5','.75','1'])
            ax.set_title(top+' / '+rep);ax.set_xlabel('Direction mixture alpha')
            if ti==0:ax.set_ylabel('Saved-path probability contribution (pp)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle('Cross-input changes along the saved path: means across 10 computational seeds',y=.98,fontsize=14)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
    fig.savefig(out/names[0],dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(14,4.8),constrained_layout=True)
    grid=fig.add_gridspec(1,4,width_ratios=[1,1,1,.05]);axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    cax=fig.add_subplot(grid[0,3]);labels=['ABAB','BABA','ABBA','BAAB']
    with np.load(folder/'arrays.npz',allow_pickle=False) as data:
        matrices=data['static_target_aligned'][0,0,[0,2,4]].mean(axis=1)
    limit=max(1e-8,float(np.abs(matrices).max()))
    for index,(ax,alpha,matrix) in enumerate(zip(axes,[0,.5,1],matrices)):
        assert np.allclose(matrix,initial['primary_static_signed_matrices'][index]['mean_target_aligned_influence'],atol=1e-12,rtol=0)
        im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(4),labels,rotation=30);ax.set_yticks(range(4),labels)
        ax.set_title('real / raw, alpha='+str(alpha));ax.set_xlabel('Destination input');ax.set_ylabel('Source input')
        for i in range(4):
            for j in range(4):
                text='0.00' if abs(matrix[i,j])<.005 else f'{matrix[i,j]:.2f}'
                ax.text(j,i,text,ha='center',va='center',color='white' if abs(matrix[i,j])>.6*limit else 'black',fontsize=10)
    fig.colorbar(im,cax=cax,label='Target-aligned score influence')
    fig.savefig(out/names[1],dpi=150);plt.close(fig)
    new(out/'final_provenance.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_git_commit':json.loads((folder/'metadata.json').read_text())['source_git_commit'],
        'renderer_script':'scripts/render_signed_figures.py','renderer_script_sha256':sha(Path(__file__)),
        'summary_sha256':sha(folder/'summary.json'),'arrays_sha256':sha(folder/'arrays.npz'),
        'verification_sha256':sha(folder/'verification.json'),'plotted_values_sha256':sha(out/'plotted_values.json'),
        'numeric_plot_summary_agreement_atol':1e-12,'scientific_analysis_changed':False,
        'figure_sha256':{name:sha(out/name) for name in names}})
    new(out/'layout_correction.json',{'reason':'Visual inspection found initial probability title overlapping legend and long heatmap colorbar title near crop edge',
        'changed_only':'Legend/title spacing, margins, shorter colorbar label, rounded display of near-zero signed values',
        'initial_pngs_preserved':{name:sha(out/name) for name in ['probability-accounting.png','signed-static-influence.png']},
        'final_pngs':{name:sha(out/name) for name in names},
        'all30_bar_values_and3_mean_signed_matrices_unchanged':True,'frozen_scientific_sources_changed':False})
    print('Final figure layouts written to new filenames; initial PNGs and numerical results preserved.')


if __name__=='__main__':main()
