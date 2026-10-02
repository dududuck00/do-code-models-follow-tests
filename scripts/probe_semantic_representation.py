"""Nested family-held-out prediction of behavior, using representations or output text."""
import ast,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
import yaml
from analyze_semantic_representation import load_model,clean
ROOT=Path(__file__).resolve().parents[1]

def predict(kernel,train,test,y,alpha):
    k=kernel[np.ix_(train,train)];cross=kernel[np.ix_(test,train)]
    column=k.mean(axis=0);mean=k.mean()
    centered=k-column[None,:]-column[:,None]+mean
    scale=float(np.trace(centered)/len(train))
    if scale<1e-12:return np.full(len(test),y[train].mean())
    weights=np.linalg.solve(centered/scale+alpha*np.eye(len(train)),y[train]-y[train].mean())
    features=(cross-cross.mean(axis=1)[:,None]-column[None,:]+mean)/scale
    return np.clip(y[train].mean()+features@weights,0,1)

def group_ci(values,groups,draws):
    families=sorted(set(groups));means=np.array([np.mean(values[groups==g]) for g in families])
    return np.quantile(means[draws].mean(axis=1),[.025,.975]).tolist()

def main():
    cfg=yaml.safe_load((ROOT/'configs/semantic_representation.yaml').read_text());pcfg=cfg['probe'];base=ROOT/'outputs/semantic_representation'
    tasks={r['task_id']:r for r in map(json.loads,(ROOT/cfg['dataset']).open())};tids=sorted(tasks)
    groups=np.array([tasks[t]['family_id'] for t in tids]);families=np.array(sorted(set(groups)))
    perm=np.random.default_rng(pcfg['seed']).permutation(families);folds=np.array_split(perm,pcfg['outer_folds'])
    fold_ids={str(g):i for i,fold in enumerate(folds) for g in fold}
    (base/'family_folds.json').write_text(json.dumps({'seed':pcfg['seed'],'outer_fold_by_family':fold_ids,'inner_folds':pcfg['inner_folds']},indent=2)+'\n')
    text=[]
    for tid in tids:
        parts=[]
        for c in cfg['conditions']:
            outputs=[ast.unparse(ast.parse(test).body[0].test.comparators[0]) for test in tasks[tid]['condition_tests'][c]]
            parts.append(c+': '+' ; '.join(outputs))
        text.append('\n'.join(parts))
    texts=np.array(text);text_cache={}
    def text_kernel(train):
        key=tuple(train)
        if key not in text_cache:
            vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,sublinear_tf=True,dtype=np.float64)
            vectorizer.fit(texts[train]);x=vectorizer.transform(texts);text_cache[key]=(x@x.T).toarray()
        return text_cache[key]
    behavior={(r['model'],r['task_id']):r for r in map(json.loads,(base/'paired_behavior.jsonl').open())}
    draws=np.random.default_rng(0).integers(0,20,(5000,20));result={'complete':True,'target':'three_run_mean_paired_switching','tasks_per_model':120,'outer_folds':5,'inner_folds':4,'families':20,'models':{}}
    all_predictions=[]
    for model in cfg['models']:
        ids,a,b,tokens=load_model(model);assert ids==tids
        relative=(b-a)/((np.linalg.norm(a,axis=-1)+np.linalg.norm(b,axis=-1))/2)[:,:,None]
        layer_ids=sorted({max(1,round(d*(a.shape[1]-1))) for d in pcfg['layer_depth_candidates']})
        kernels={layer:relative[:,layer,:]@relative[:,layer,:].T for layer in layer_ids}
        y=np.array([behavior[model,t]['paired_switch'] for t in tids]);predictions={k:np.full(len(tids),np.nan) for k in ['mean','median','output_text','representation']};selection=[]
        for fold,test_families in enumerate(folds):
            test=np.flatnonzero(np.isin(groups,test_families));train=np.flatnonzero(~np.isin(groups,test_families))
            inner=[(train[tr],train[va]) for tr,va in GroupKFold(n_splits=pcfg['inner_folds']).split(train,groups=groups[train])]
            assert not set(groups[train])&set(groups[test])
            predictions['mean'][test]=y[train].mean()
            predictions['median'][test]=np.median(y[train])
            selected={}
            for kind,candidates in [('output_text',[None]),('representation',layer_ids)]:
                scored=[]
                for layer in candidates:
                    for alpha in pcfg['alpha_candidates']:
                        errors=[]
                        for tr,va in inner:
                            assert not set(groups[tr])&set(groups[va])
                            kernel=text_kernel(tr) if kind=='output_text' else kernels[layer]
                            errors.extend(np.abs(predict(kernel,tr,va,y,alpha)-y[va]))
                        scored.append((float(np.mean(errors)),-alpha,layer if layer is not None else -1))
                loss,negative_alpha,layer=min(scored);alpha=-negative_alpha
                kernel=text_kernel(train) if kind=='output_text' else kernels[layer]
                predictions[kind][test]=predict(kernel,train,test,y,alpha)
                selected[kind]=dict(layer=layer if kind=='representation' else None,alpha=alpha,inner_mae=loss)
            selection.append(dict(fold=fold,test_families=list(test_families),training_families=sorted(set(groups[train])),selected=selected))
        assert all(np.isfinite(pred).all() for pred in predictions.values())
        metrics={}
        for name,pred in predictions.items():
            absolute=np.abs(pred-y);squared=(pred-y)**2
            metrics[name]={'mae':float(absolute.mean()),'mae_ci95':group_ci(absolute,groups,draws),'mse':float(squared.mean())}
        contrasts={}
        for baseline in ['mean','median','output_text']:
            difference=np.abs(predictions['representation']-y)-np.abs(predictions[baseline]-y)
            contrasts['representation_minus_'+baseline]={'mae_difference':float(difference.mean()),'ci95':group_ci(difference,groups,draws)}
        result['models'][model]={'metrics':metrics,'contrasts':contrasts,'fold_selection':selection,'candidate_layers':layer_ids}
        for i,tid in enumerate(tids):all_predictions.append(dict(model=model,task_id=tid,family_id=groups[i],fold=fold_ids[groups[i]],paired_switch=y[i],**{k:pred[i] for k,pred in predictions.items()}))
        print(model,metrics,contrasts,flush=True)
    (base/'family_probe.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    (base/'family_probe_predictions.jsonl').write_text(''.join(json.dumps(clean(r))+'\n' for r in all_predictions))
if __name__=='__main__':main()
