"""Build the static comparison dashboard from ten saved real-data experiments."""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from paths import ROOT, SOURCE
import shutil
DEST=ROOT/'dashboard';(DEST/'data').mkdir(parents=True, exist_ok=True)
if ROOT != SOURCE:
    for asset in ('index.html', 'style.css', 'app.js'):
        shutil.copy2(SOURCE / 'dashboard' / asset, DEST / asset)
def load(p):return json.loads((ROOT/p).read_text())
base=load('results/report.json');ten=load('ten_results/report.json')
assert len(ten['models'])==10,'Run run_ten_models.py to completion first.'
models=[]
for key,r in ten['models'].items():
    ic=pd.read_csv(ROOT/f'ten_results/{key}_ic.csv')
    models.append(dict(id=key,name=r['name'],description=r['description'],family=r['family'],color=r['color'],features=r['n_features'],trainRows=ten['train_rows'],validRows=ten['validation_rows'],trainDates=ten['train_dates'],validDates=ten['validation_dates'],params=r['params'],seconds=r['seconds'],warnings=r['warnings'],actualIterations=r.get('actual_iterations'),series=[dict(date=row['date'][:10],ic=row[key]) for _,row in ic.iterrows()]))
prediction_models=[m['id'] for m in models]
ic=pd.read_csv(ROOT/'results/validation_by_date.csv')
models.append(dict(id='original',name='最初基线 XGBoost',description='原始样本，仅单独参考，不参与十方案排名。',family='历史参考',color='#8390a5',features=base['n_features'],trainRows=base['train_rows'],validRows=base['validation_rows'],trainDates=base['train_dates'],validDates=base['validation_dates'],params={'n_estimators':100,'max_depth':3,'learning_rate':0.05},seconds=None,warnings=[],series=[dict(date=row['date'][:10],ic=row['spearman_rank_ic']) for _,row in ic.iterrows()]))
summary=dict(schema=2,synthetic=base.get('synthetic', False),models=models,predictionModels=prediction_models,target=base['target'],version=base['version'],builtAt=datetime.now().isoformat(timespec='seconds'),seed=42,live=dict(date=base['live_date'],rows=base['live_rows'],round=base['round'],uploaded=base['uploaded']),dates=[x['date'] for x in models[0]['series']],packages=ten['packages'])
pred=pd.read_parquet(ROOT/'ten_results/predictions.parquet')
assert len(pred)==ten['validation_rows']
for date,group in pred.groupby('date'):
    group=group.copy()
    for key in prediction_models:group[key+'_rank']=group[key].rank(pct=True)
    columns=['numerai_ticker','target_jupiter_60']+prediction_models+[k+'_rank' for k in prediction_models]
    rows=[[str(row[0])]+[round(float(v),7) for v in row[1:]] for row in group[columns].itertuples(index=False,name=None)]
    date_string=str(date)[:10]
    (DEST/f'data/{date_string}.json').write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':')))
# Add the finished Stack experiment separately: development and final-test scores must not be mixed.
if (ROOT/'agent_results/final_test.json').exists():
    summary['agent']={
        'development':load('agent_results/development_results.json'),
        'selection':load('agent_results/frozen_selection.json'),
        'test':load('agent_results/final_test.json'),
        'audit':load('agent_results/recipe_audit.json') if (ROOT/'agent_results/recipe_audit.json').exists() else {},
        'knowledge':json.loads((SOURCE/'agent_knowledge.json').read_text()),
        'policy': {'provider': 'DeepSeek'},
    }
if (ROOT/'official_benchmark/report.json').exists():
    summary['officialBenchmark']=load('official_benchmark/report.json')
# Publish the schema only after every detail file has been generated.
(DEST/'data/summary.json').write_text(json.dumps(summary,ensure_ascii=False))
# Refresh the newest local official-metric research snapshot as part of a normal rebuild.
import subprocess,sys
for folder in ['official_research_v2','official_research']:
    publisher=ROOT/folder/'publish_dashboard.py'
    if publisher.exists() and (ROOT/folder/'protocol.json').exists():
        subprocess.run([sys.executable,str(publisher)],check=True)
        break
print('Built dashboard:',len(summary['dates']),'dates;',len(pred),'rows; 10 comparable models + original reference.')
