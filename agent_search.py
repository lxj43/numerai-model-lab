"""Bounded LLM proposal -> local training -> development feedback.
No generated Python execution. Freeze one winner before a later final-test run.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,time
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from run_ten_models import build,FEATURES,TARGET,ROOT
from llm_client import chat, catalog, config
from paths import SOURCE
OUT=ROOT/'agent_results'
FAMILIES={'ridge','lightgbm','random_forest','catboost','ensemble'}
BOUNDS={
 'ridge':{'alpha':(0.01,100000)},
 'lightgbm':{'n_estimators':(50,500),'max_depth':(2,8),'num_leaves':(4,63),'learning_rate':(0.01,0.15),'min_child_samples':(50,2000),'reg_alpha':(0,100),'reg_lambda':(0,100),'colsample_bytree':(0.3,1.0),'subsample':(0.4,1.0)},
 'random_forest':{'n_estimators':(40,160),'max_depth':(3,10),'min_samples_leaf':(50,2000),'max_features':(0.2,1.0),'max_samples':(0.4,1.0)},
 'catboost':{'iterations':(50,500),'depth':(3,7),'learning_rate':(0.01,0.15),'l2_leaf_reg':(1,100)}
}
INT_PARAMS={'n_estimators','max_depth','num_leaves','min_child_samples','min_samples_leaf','iterations','depth'}

def save(name,obj):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)

def matrices(df,mode):
 current=df[FEATURES].to_numpy(dtype='float32')
 if mode=='current':return current
 first=df[[f'{f}_lag1w' for f in FEATURES]].to_numpy(dtype='float32')
 second=df[[f'{f}_lag2w' for f in FEATURES]].to_numpy(dtype='float32')
 if mode=='history':return np.concatenate([current,first,second],axis=1)
 if mode=='delta':return np.concatenate([current,current-first,current-second],axis=1)
 if mode=='smooth':return np.concatenate([current,(current+first+second)/3],axis=1)
 raise ValueError('Unknown feature mode')

def validate(p,available):
 if not isinstance(p,dict):raise ValueError('Recipe must be an object')
 family=p.get('family');mode=p.get('feature_mode','current')
 if family not in FAMILIES or mode not in {'current','history','delta','smooth'}:raise ValueError('Unsupported family or feature mode')
 if p.get('target_mode','raw') not in {'raw','date_rank'}:raise ValueError('Unsupported target mode')
 if p.get('weighting','uniform') not in {'uniform','equal_date','recent'}:raise ValueError('Unsupported weighting')
 for key in ['name','rationale']:
  if not isinstance(p.get(key),str) or not p[key].strip() or len(p[key])>1200:raise ValueError('Missing or oversized '+key)
 p={k:v for k,v in p.items() if k in {'name','rationale','family','feature_mode','target_mode','weighting','params','members'}}
 p.update(feature_mode=mode,target_mode=p.get('target_mode','raw'),weighting=p.get('weighting','uniform'))
 if family=='ensemble':
  members=p.get('members',[])
  if not 2<=len(members)<=5:raise ValueError('Ensemble requires 2 to 5 existing members')
  if len({m['id'] for m in members})!=len(members):raise ValueError('Duplicate ensemble members')
  for m in members:
   if set(m)!={'id','weight'} or m['id'] not in available or not isinstance(m['weight'],(int,float)) or not np.isfinite(m['weight']) or not 0<m['weight']<=1:raise ValueError('Invalid ensemble member')
  if abs(sum(m['weight'] for m in members)-1)>0.001:raise ValueError('Ensemble weights must sum to one')
  p['params']={}
 else:
  params=p.get('params',{})
  if not isinstance(params,dict):raise ValueError('params must be an object')
  for k,v in params.items():
   if k not in BOUNDS[family] or isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not BOUNDS[family][k][0]<=v<=BOUNDS[family][k][1]:raise ValueError('Parameter outside supported bounds: '+k)
   if k in INT_PARAMS:
    if int(v)!=v:raise ValueError(k+' must be an integer')
    params[k]=int(v)
  p['params']=params
 return p

def estimator(recipe):
 p=recipe['params'];f=recipe['family']
 if f=='ridge':return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(**p))
 if f=='lightgbm':
  defaults=dict(n_estimators=200,max_depth=4,num_leaves=15,learning_rate=0.03,min_child_samples=100,colsample_bytree=0.8,subsample=0.8)
  defaults.update(p)
  return LGBMRegressor(**defaults,subsample_freq=1,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
 if f=='random_forest':
  defaults=dict(n_estimators=80,max_depth=8,min_samples_leaf=100,max_features=0.5,max_samples=0.7);defaults.update(p)
  return make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(**defaults,random_state=42,n_jobs=4))
 if f=='catboost':
  defaults=dict(iterations=200,depth=4,learning_rate=0.03,l2_leaf_reg=3);defaults.update(p)
  return CatBoostRegressor(**defaults,loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
 raise ValueError('No trainable estimator')

def fit_recipe(recipe,train):
 x=matrices(train,recipe['feature_mode'])
 y=(train.groupby('date')[TARGET].rank(pct=True) if recipe['target_mode']=='date_rank' else train[TARGET]).to_numpy(dtype='float32')
 weights=np.ones(len(train),dtype='float64')
 if recipe['weighting']=='equal_date':weights=1/train.groupby('date')['date'].transform('size').to_numpy()
 if recipe['weighting']=='recent':weights=0.5**((train.date.max()-train.date).dt.days.to_numpy()/180)
 weights/=weights.mean()
 model=estimator(recipe)
 if hasattr(model,'named_steps'):
  last=list(model.named_steps)[-1];model.fit(x,y,**{last+'__sample_weight':weights})
 else:model.fit(x,y,sample_weight=weights)
 return model

def metric(pred,frame):
 d=frame[['date',TARGET]].copy();d['prediction']=pred
 ic=d.groupby('date').apply(lambda g:g.prediction.corr(g[TARGET],method='spearman'),include_groups=False)
 if not np.isfinite(pred).all() or ic.isna().any():raise ValueError('Nonfinite predictions or undefined IC')
 return dict(mean_rank_ic=float(ic.mean()),ic_std=float(ic.std()),positive_dates=int((ic>0).sum()),n_dates=len(ic),series=[{'date':str(date.date()),'ic':float(v)} for date,v in ic.items()])

def combine(recipe,predictions,frame):
 result=np.zeros(len(frame))
 for member in recipe['members']:
  ranks=pd.Series(predictions[member['id']]).groupby(frame.date.reset_index(drop=True)).rank(pct=True).to_numpy()
  result+=member['weight']*ranks
 return result

def prompt_for(round_number,previous,base_report,available):
 knowledge=json.loads((SOURCE/'agent_knowledge.json').read_text())
 context={'knowledge':knowledge,'baseline_results':{k:{'name':v['name'],'mean_rank_ic':v['mean_rank_ic'],'params':v['params']} for k,v in base_report['models'].items()},'previous_agent_results':previous,'allowed_ensemble_members':available,'parameter_bounds':BOUNDS,'round':round_number}
 instructions='''You are a quantitative research proposer. Read the supplied knowledge and observed DEVELOPMENT results, then propose exactly 3 executable recipes to try to improve mean per-date Spearman Rank IC toward 0.02. This is not a guarantee. You do not receive final-test labels or metrics. Never invent feature meanings. Maximize robust ranking, not fitted labels. Explain hypotheses in Chinese. Budget permits only these structured recipes; no Python code or shell commands.
Return ONLY JSON: {"knowledge_applied":["..."],"proposals":[{"name":"...","rationale":"...","family":"lightgbm|ridge|random_forest|catboost|ensemble","feature_mode":"current|history|delta|smooth","target_mode":"raw|date_rank","weighting":"uniform|equal_date|recent","params":{},"members":[]}]}. knowledge_applied must cite supplied source URLs or protocol rules.
current=100 current features; history=current+7day+14day=300; delta=current+(current-7day)+(current-14day)=300; smooth=current+mean(current,7day,14day)=200. These are feature-bin differences, not stock-price returns. Only use past/current features.
All non-ensemble params must be in the provided family bounds. Integers must be integers. For ensemble: params={}, members=[{"id":"an allowed existing id","weight":0.5},...], 2-5 UNIQUE existing members with strictly positive weights summing to 1. Members are blended by per-date prediction rank. No references to other new proposals in this batch.
Use the budget to cover meaningfully different hypotheses, including a rank ensemble and a train-only temporal feature transformation. Do not modify the target identity, dataset split, or scoring metric. Predictions must cover every common sample. Treat 31 development dates as a small, noisy and overlapping evaluation set.'''
 return [{'role':'system','content':instructions},{'role':'user','content':json.dumps(context,ensure_ascii=False)}]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',default=config()['model']);parser.add_argument('--rounds',type=int,default=2,choices=[1,2]);args=parser.parse_args()
 OUT.mkdir(exist_ok=True)
 if (OUT/'frozen_selection.json').exists():raise SystemExit('Selection already frozen. Preserve this run; use a new output directory for a new protocol.')
 models=catalog();save('available_models.json',{'models':models,'selected':args.model})
 if args.model not in models:raise SystemExit('Selected model is not in current catalog')
 train,dev=build('train'),build('validation')
 initial=pd.read_parquet(ROOT/'ten_results/predictions.parquet')
 assert len(dev)==len(initial) and (dev.numerai_ticker==initial.numerai_ticker).all()
 allowed=['current_only','three_weekly_snapshots','ridge','elastic_net','random_forest','lightgbm']
 predictions={k:initial[k].to_numpy() for k in allowed}
 base_report=json.loads((ROOT/'ten_results/report.json').read_text())
 records=[]
 for round_number in range(1,args.rounds+1):
  messages=prompt_for(round_number,[{k:v for k,v in r.items() if k not in {'series'}} for r in records],base_report,list(predictions))
  prompt_path=OUT/f'round_{round_number}_prompt.json';save(prompt_path.name,messages)
  parsed=None
  for attempt in range(2):
   print('LLM proposal round',round_number,'attempt',attempt+1,flush=True)
   content,meta=chat(args.model,messages)
   save(f'round_{round_number}_response_{attempt+1}.json',{'response':content,**meta})
   try:
    obj=json.loads(content[content.find('{'):content.rfind('}')+1])
    if len(obj.get('proposals',[]))!=3:raise ValueError('Exactly 3 proposals required')
    parsed=[validate(p,set(predictions)) for p in obj['proposals']]
    if not isinstance(obj.get('knowledge_applied'),list) or not obj['knowledge_applied']:raise ValueError('Knowledge application summary required')
    save(f'round_{round_number}_plans.json',obj);break
   except (ValueError,TypeError,KeyError) as error:
    messages+=[{'role':'assistant','content':content},{'role':'user','content':'Fix the JSON schema/recipe error and return a complete JSON object: '+str(error)}]
  if parsed is None:raise RuntimeError('The model did not return executable recipes after repair')
  for i,recipe in enumerate(parsed,1):
   key=f'agent_r{round_number}_p{i}';started=time.perf_counter()
   print('TRAIN',key,recipe['name'],recipe['family'],recipe['feature_mode'],flush=True)
   if recipe['family']=='ensemble':pred=combine(recipe,predictions,dev)
   else:
    model=fit_recipe(recipe,train);pred=model.predict(matrices(dev,recipe['feature_mode']))
    joblib.dump(model,OUT/f'{key}.joblib')
   predictions[key]=pred;np.save(OUT/f'{key}_dev.npy',pred)
   result=metric(pred,dev)
   record={'id':key,'round':round_number,'llm':args.model,'recipe':recipe,'seconds':round(time.perf_counter()-started,3),**result}
   records.append(record);save('development_results.json',{'model':args.model,'target_ic':0.02,'train_rows':len(train),'development_rows':len(dev),'records':records})
   print('SCORE',key,round(result['mean_rank_ic'],6),flush=True)
 best=max(records,key=lambda r:r['mean_rank_ic'])
 fingerprint=hashlib.sha256((OUT/'development_results.json').read_bytes()).hexdigest()
 selection={'selected_id':best['id'],'recipe':best['recipe'],'development_metric':best['mean_rank_ic'],'target_ic':0.02,'llm':args.model,'freeze_time':datetime.now().isoformat(timespec='seconds'),'development_results_sha256':fingerprint,'selection_rule':'highest DEVELOPMENT mean Rank IC among bounded model proposals; no final-test feedback'}
 save('frozen_selection.json',selection)
 print('FROZEN',best['id'],'development IC',best['mean_rank_ic'],flush=True)
if __name__=='__main__':main()
