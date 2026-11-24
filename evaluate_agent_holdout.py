"""Evaluate ONLY the frozen development winner and fixed baseline on later data.
Scores are saved once and never fed back into the proposer.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
from datetime import datetime
import hashlib,json
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from agent_search import ROOT,OUT,FEATURES,TARGET,matrices,metric,combine,save
from run_ten_models import build

def holdout_frame():
 df=pd.read_parquet(ROOT/'data/agent_holdout.parquet');df['date']=pd.to_datetime(df.date)
 assert not df.duplicated(['date','numerai_ticker']).any()
 for weeks in (1,2):
  past=df[['date','numerai_ticker']+FEATURES].copy();past['date']+=pd.Timedelta(weeks=weeks)
  past[f'_present{weeks}']=True;past.rename(columns={f:f'{f}_lag{weeks}w' for f in FEATURES},inplace=True)
  df=df.merge(past,on=['date','numerai_ticker'],how='left',validate='one_to_one')
 return df.loc[df['_present1'].eq(True)&df['_present2'].eq(True)].dropna(subset=[TARGET]).reset_index(drop=True)

def main():
 report_path=OUT/'final_test.json'
 if report_path.exists():
  print('Final evaluation already exists; preserving its scores.');return
 selection=json.loads((OUT/'frozen_selection.json').read_text())
 assert hashlib.sha256((OUT/'development_results.json').read_bytes()).hexdigest()==selection['development_results_sha256']
 records=json.loads((OUT/'development_results.json').read_text())['records']
 recipes={r['id']:r['recipe'] for r in records}
 train=build('train');test=holdout_frame();assert (test.date.min()-pd.Timestamp('2019-08-16')).days>180
 cached={};base_report=json.loads((ROOT/'ten_results/report.json').read_text())
 def predict(key):
  if key in cached:return cached[key]
  if key in recipes:
   recipe=recipes[key]
   if recipe['family']=='ensemble':
    for member in recipe['members']:predict(member['id'])
    result=combine(recipe,cached,test)
   else:result=joblib.load(OUT/f'{key}.joblib').predict(matrices(test,recipe['feature_mode']))
  elif key in ['current_only','three_weekly_snapshots']:
   model=XGBRegressor();model.load_model(ROOT/f'sequence_results/{key}.json')
   result=model.predict(matrices(test,'current' if key=='current_only' else 'history'))
  else:
   params=base_report['models'][key]['params']
   if key=='ridge':model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(**params))
   elif key=='elastic_net':model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),ElasticNet(**params,random_state=42))
   elif key=='random_forest':model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(**params,random_state=42,n_jobs=4))
   elif key=='lightgbm':model=LGBMRegressor(**params,subsample_freq=1,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
   else:raise ValueError('Unsupported base ensemble dependency')
   print('Reconstructing ensemble member from fixed original recipe:',key,flush=True)
   model.fit(matrices(train,'current'),train[TARGET].to_numpy(dtype='float32'))
   result=model.predict(matrices(test,'current'))
  assert len(result)==len(test) and np.isfinite(result).all()
  cached[key]=result;return result
 chosen=predict(selection['selected_id']);baseline=predict('current_only')
 # Dependency predictions may be computed, but ONLY these two scores are evaluated.
 chosen_metric=metric(chosen,test);baseline_metric=metric(baseline,test)
 pred=test[['date','numerai_ticker',TARGET]].copy();pred['baseline']=baseline;pred['selected_agent']=chosen
 pred.to_parquet(OUT/'final_test_predictions.parquet',index=False)
 result={'evaluated_at':datetime.now().isoformat(timespec='seconds'),'selected_id':selection['selected_id'],
         'freeze_time':selection['freeze_time'],'target_ic':0.02,'metric':'mean per-date Spearman Rank IC',
         'rows':len(test),'dates':[str(test.date.min().date()),str(test.date.max().date())],
         'selection_was_frozen_before_scoring':True,'baseline':baseline_metric,'selected_agent':chosen_metric,
         'delta_vs_baseline':chosen_metric['mean_rank_ic']-baseline_metric['mean_rank_ic'],
         'target_reached':chosen_metric['mean_rank_ic']>=0.02,
         'note':'Later historical validation block, unseen by this proposer before freeze. Not live performance. Labels have overlapping 60-day horizons; dates are not independent trials. Reusing this block for future tuning invalidates its held-out status.'}
 save('final_test.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','selected_agent']},ensure_ascii=False,indent=2))
 print('FINAL BASELINE IC',baseline_metric['mean_rank_ic'],'FINAL AGENT IC',chosen_metric['mean_rank_ic'],flush=True)
if __name__=='__main__':main()
