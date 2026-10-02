"""Deterministic scientific figures for the paired representation analysis."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
MODELS=['qwen25','qwen36','qwen35','qwen38']
NAMES={'qwen25':'Qwen2.5-Coder-7B','qwen36':'Qwen3.6-27B','qwen35':'Qwen3.5-9B','qwen38':'Qwen3.8-27B'}
def main():
    base=ROOT/'outputs/semantic_representation';out=base/'figures';out.mkdir(exist_ok=True)
    analysis=json.loads((base/'representation_analysis.json').read_text());data=pd.read_csv(base/'pair_layer_metrics.csv')
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.4,4.6))
    for index,(ax,m) in enumerate(zip(axes.flat,MODELS)):
        d=data[data.model==m];s=d.groupby('depth').cosine_distance.quantile([.25,.5,.75]).unstack()
        ax.fill_between(s.index,s[.25],s[.75],color='#0072B2',alpha=.17)
        ax.plot(s.index,s[.5],color='#0072B2',lw=1.6)
        ax.set(title=f'({chr(97+index)}) {NAMES[m]}',xlabel='Relative layer depth',ylabel='Cosine distance',xlim=(0,1),ylim=(0,None))
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(alpha=.18,linewidth=.5)
    fig.tight_layout(h_pad=1.5,w_pad=1.6);save(fig,'layer_geometry')
    fig,axes=plt.subplots(2,2,figsize=(7.4,4.8));handles=[]
    for index,(ax,m) in enumerate(zip(axes.flat,MODELS)):
        layers=analysis['models'][m]['layers'][1:];depth=np.array([r['depth'] for r in layers])
        for method,color,style,label in [('pooled_spearman','#0072B2','-','Pooled'),('within_family_length_adjusted_correlation','#D55E00','--','Within family, length adjusted')]:
            rows=[r['associations']['cosine_distance']['paired_switch'][method] for r in layers]
            val=np.array([r['estimate'] for r in rows],dtype=float);ci=np.array([r['ci95'] for r in rows],dtype=float)
            line,=ax.plot(depth,val,color=color,ls=style,lw=1.6,label=label)
            ax.fill_between(depth,ci[:,0],ci[:,1],color=color,alpha=.11)
            if index==0:handles.append(line)
        ax.axhline(0,color='.4',ls=':',lw=.8);ax.set(title=f'({chr(97+index)}) {NAMES[m]}',xlabel='Relative layer depth',ylabel='Rank correlation',xlim=(0,1),ylim=(-1,1));ax.grid(alpha=.15,linewidth=.5)
    fig.tight_layout(rect=(0,.08,1,1),h_pad=1.4,w_pad=1.5);fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False);save(fig,'layer_behavior_association')
    print(out)
if __name__=='__main__':main()
