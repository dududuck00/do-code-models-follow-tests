"""Paired semantic versus surface distances, clustered by the original rule families."""
import csv,json
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def load(path):
    records=rows(path);out={}
    for r in records:
        with np.load(ROOT/r['state_path']) as f:h=f['prompt_end_hidden'].astype(np.float64)
        assert np.isfinite(h).all();key=(r['task_id'],r['condition']);assert key not in out
        out[key]=h
    return out

def distances(a,b):
    na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
    return {'cosine_distance':np.clip(1-(a*b).sum(axis=-1)/(na*nb),0,2),'relative_l2':np.linalg.norm(a-b,axis=-1)/((na+nb)/2)}
def main():
    cfg=yaml.safe_load((ROOT/'configs/semantic_surface.yaml').read_text());base=ROOT/'outputs/semantic_surface'
    tasks=sorted(rows(ROOT/cfg['dataset']),key=lambda r:r['task_id']);families=sorted({t['family_id'] for t in tasks})
    groups=np.array([t['family_id'] for t in tasks]);draws=np.random.default_rng(0).integers(0,len(families),(5000,len(families)))
    report={'complete':True,'protocol':cfg['protocol'],'new_states':480*len(cfg['models']),'reused_states':240*len(cfg['models']),'tasks_per_model':120,'families':20,'bootstrap_samples':5000,'primary_layer':'final','models':{}}
    detail=[]
    for m in cfg['models']:
        original=load(ROOT/cfg['original_states'].format(model=m));new=load(base/m/'replay/generations.jsonl');assert len(original)==240 and len(new)==480
        a,b=[np.stack([original[t['task_id'],'tests_'+rule] for t in tasks]) for rule in ['a','b']]
        ab=distances(a,b);summary={'variants':{},'layers_including_embedding':a.shape[1]}
        accumulated={}
        for variant in cfg['variants']:
            av,bv=[np.stack([new[t['task_id'],'tests_'+rule+'__'+variant] for t in tasks]) for rule in ['a','b']]
            assert av.shape==bv.shape==a.shape
            same_a,same_b,changed=distances(a,av),distances(b,bv),distances(av,bv)
            summary['variants'][variant]={}
            for metric in cfg['analysis']['metrics']:
                same=(same_a[metric]+same_b[metric])/2;different=(ab[metric]+changed[metric])/2;delta=different-same
                accumulated.setdefault(metric,[]).append(delta)
                family=np.stack([delta[groups==g].mean(axis=0) for g in families]);ci=np.quantile(family[draws].mean(axis=1),[.025,.975],axis=0)
                layers=[dict(layer=l,depth=l/(a.shape[1]-1),same_rule_mean=float(same[:,l].mean()),different_rule_mean=float(different[:,l].mean()),difference=float(delta[:,l].mean()),ci95=ci[:,l].tolist()) for l in range(a.shape[1])]
                summary['variants'][variant][metric]={'layers':layers,'final_layer':layers[-1]}
                for i,t in enumerate(tasks):
                    for l in range(a.shape[1]):detail.append(dict(model=m,task_id=t['task_id'],family_id=t['family_id'],variant=variant,metric=metric,layer=l,same_rule=float(same[i,l]),different_rule=float(different[i,l]),difference=float(delta[i,l])))
        summary['variant_mean_secondary']={}
        for metric,ds in accumulated.items():
            delta=np.mean(ds,axis=0);family=np.stack([delta[groups==g].mean(axis=0) for g in families]);ci=np.quantile(family[draws].mean(axis=1),[.025,.975],axis=0)
            summary['variant_mean_secondary'][metric]={'final_difference':float(delta[:,-1].mean()),'ci95':ci[:,-1].tolist()}
        report['models'][m]=summary;print(m,{v:s['cosine_distance']['final_layer'] for v,s in summary['variants'].items()},flush=True)
    (base/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    with (base/'pair_layer_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
if __name__=='__main__':main()
