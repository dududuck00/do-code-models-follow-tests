"""Within-model A/B representation geometry and family-clustered behavior associations."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import yaml
ROOT=Path(__file__).resolve().parents[1]

def correlation(x,y):
    x=x-np.mean(x,axis=-2,keepdims=True);y=y-np.mean(y,axis=-2,keepdims=True)
    den=np.sqrt(np.sum(x*x,axis=-2)*np.sum(y*y,axis=-2))
    return np.divide(np.sum(x*y,axis=-2),den,out=np.full_like(den,np.nan),where=den>1e-12)

def intervals(values):
    out=[]
    for col in values.T:
        valid=col[np.isfinite(col)]
        out.append({'ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None],'valid_bootstrap_samples':len(valid)})
    return out

def association(x,y,z,groups,draws):
    """x: tasks x layers; rank within each six-instance family for within-family estimates."""
    pooled=correlation(rankdata(x,axis=0),rankdata(y)[:,None])
    ids=np.array([i for i in sorted(set(groups))]);family_indices=[np.flatnonzero(groups==g) for g in ids]
    cx=np.zeros_like(x);cy=np.zeros(len(y));cz=np.zeros_like(z)
    for ind in family_indices:
        for dest,source in [(cx,x),(cz,z)]:
            v=rankdata(source[ind],axis=0);dest[ind]=v-v.mean(axis=0)
        v=rankdata(y[ind]);cy[ind]=v-v.mean()
    sx=np.array([(cx[i]**2).sum(axis=0) for i in family_indices]);sy=np.array([(cy[i]**2).sum() for i in family_indices]);sxy=np.array([(cx[i]*cy[i,None]).sum(axis=0) for i in family_indices])
    sz=np.array([cz[i].T@cz[i] for i in family_indices]);szx=np.array([cz[i].T@cx[i] for i in family_indices]);szy=np.array([cz[i].T@cy[i] for i in family_indices])
    def within(weights,adjust=False):
        xx=weights@sx;yy=weights@sy;xy=weights@sxy
        if adjust:
            zz=np.einsum('bg,gij->bij',weights,sz);zx=np.einsum('bg,gil->bil',weights,szx);zy=weights@szy
            inv=np.linalg.pinv(zz)
            xx=xx-np.einsum('bil,bij,bjl->bl',zx,inv,zx)
            yy=yy-np.einsum('bi,bij,bj->b',zy,inv,zy)
            xy=xy-np.einsum('bil,bij,bj->bl',zx,inv,zy)
        den=np.sqrt(np.maximum(xx,0)*np.maximum(yy[:,None],0))
        return np.divide(xy,den,out=np.full_like(xy,np.nan),where=den>1e-12)
    weights=np.array([np.bincount(d,minlength=len(ids)) for d in draws])
    overall=np.ones((1,len(ids)))
    within_point=within(overall)[0];adjusted_point=within(overall,True)[0]
    pooled_boot=[]
    for start in range(0,len(draws),100):
        indexes=np.array([np.concatenate([family_indices[g] for g in draw]) for draw in draws[start:start+100]])
        rx=rankdata(x[indexes],axis=1);ry=rankdata(y[indexes],axis=1)[:,:,None]
        pooled_boot.append(correlation(rx,ry))
    return {'pooled_spearman':(pooled,intervals(np.concatenate(pooled_boot))),
            'within_family_spearman':(within_point,intervals(within(weights))),
            'within_family_length_adjusted_correlation':(adjusted_point,intervals(within(weights,True)))}

def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj

def load_model(model):
    source=ROOT/f'outputs/semantic_representation/{model}/replay/generations.jsonl'
    records=list(map(json.loads,source.open()));assert len(records)==240
    by={(r['task_id'],r['condition']):r for r in records}
    tasks=sorted({r['task_id'] for r in records});assert len(tasks)==120
    arrays={c:[] for c in ['tests_a','tests_b']};tokens=[]
    for tid in tasks:
        for c in arrays:
            r=by[tid,c];assert r['prompt_tokens_verified'] and r['state_prompt_token_hash']==r['prompt_token_hash']
            h=np.load(ROOT/r['state_path'])['prompt_end_hidden'];assert h.ndim==2 and np.isfinite(h).all()
            arrays[c].append(h)
        tokens.append([by[tid,c]['state_prompt_token_count'] for c in arrays])
    return tasks,np.stack(arrays['tests_a']).astype(np.float64),np.stack(arrays['tests_b']).astype(np.float64),np.array(tokens)

def main():
    cfg=yaml.safe_load((ROOT/'configs/semantic_representation.yaml').read_text());base=ROOT/'outputs/semantic_representation'
    paired={(r['model'],r['task_id']):r for r in map(json.loads,(base/'paired_behavior.jsonl').open())}
    draws=np.random.default_rng(cfg['analysis']['seed']).integers(0,20,(cfg['analysis']['bootstrap_samples'],20))
    report={'complete':True,'unique_prompts':240*len(cfg['models']),'task_pairs_per_model':120,'families':20,'bootstrap_samples':len(draws),'primary_layer':'final','within_family_method':'Ranks computed within each family; centered within family. Length adjustment residualizes mean prompt length and absolute A/B length difference.','models':{}}
    layer_rows=[];task_rows=[]
    for m in cfg['models']:
        tids,a,b,tokens=load_model(m);p=[paired[m,t] for t in tids];groups=np.array([r['family_id'] for r in p]);assert len(set(groups))==20
        na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
        cosine=np.clip(1-np.sum(a*b,axis=-1)/(na*nb),0,2)
        relative=np.linalg.norm(b-a,axis=-1)/((na+nb)/2)
        z=np.column_stack([tokens.mean(axis=1),np.abs(tokens[:,0]-tokens[:,1])])
        layers=a.shape[1];summary={'states':240,'layers_including_embedding':layers,'hidden_dimension':a.shape[2],'layers':[],'family_behavior':{}}
        for g in sorted(set(groups)):
            summary['family_behavior'][g]={outcome:float(np.mean([p[i][outcome] for i in np.flatnonzero(groups==g)])) for outcome in ['paired_switch','target_adoption','alternative_adoption','neither_fraction']}
        associations={}
        for metric,x in [('cosine_distance',cosine),('relative_l2',relative)]:
            associations[metric]={}
            for outcome in ['paired_switch','target_adoption']:
                associations[metric][outcome]=association(x,np.array([r[outcome] for r in p]),z,groups,draws)
        for layer in range(layers):
            lr={'model':m,'layer':layer,'depth':layer/(layers-1),'median_cosine_distance':float(np.median(cosine[:,layer])),'median_relative_l2':float(np.median(relative[:,layer]))}
            detail=dict(lr,associations={})
            for metric,outcomes in associations.items():
                detail['associations'][metric]={}
                for outcome,methods in outcomes.items():
                    detail['associations'][metric][outcome]={method:{'estimate':float(values[layer]),**cis[layer]} for method,(values,cis) in methods.items()}
            for method,values in detail['associations']['cosine_distance']['paired_switch'].items():
                lr[method]=values['estimate'];lr[method+'_lo'],lr[method+'_hi']=values['ci95']
            summary['layers'].append(detail);layer_rows.append(lr)
            for i,tid in enumerate(tids):
                task_rows.append(dict(model=m,task_id=tid,family_id=groups[i],layer=layer,depth=layer/(layers-1),cosine_distance=float(cosine[i,layer]),relative_l2=float(relative[i,layer]),mean_prompt_tokens=float(z[i,0]),absolute_token_difference=int(z[i,1]),**{k:p[i][k] for k in ['paired_switch','target_adoption','alternative_adoption','neither_fraction']}))
        summary['final_layer']=summary['layers'][-1]
        summary['switch_groups']={}
        for name,mask in [('never_switch',np.array([r['paired_switch']==0 for r in p])),('variable_switch',np.array([0<r['paired_switch']<1 for r in p])),('always_switch',np.array([r['paired_switch']==1 for r in p]))]:
            summary['switch_groups'][name]={'task_pairs':int(mask.sum()),'final_cosine_median':float(np.median(cosine[mask,-1])) if mask.any() else None}
        report['models'][m]=summary
        print(m,clean(summary['final_layer']['associations']['cosine_distance']['paired_switch']),flush=True)
    for name,rows in [('layer_metrics.csv',layer_rows),('pair_layer_metrics.csv',task_rows)]:
        with (base/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (base/'representation_analysis.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
