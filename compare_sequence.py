"""Compare current-only vs three weekly snapshots on identical historical rows.
Uses cached public data from run_demo.py; no downloads or submissions.
"""
from pathlib import Path
import json
import pandas as pd
from xgboost import XGBRegressor

from paths import ROOT
OUT = ROOT / 'sequence_results'
OUT.mkdir(exist_ok=True)
TARGET = 'target_jupiter_60'
features = [f for f in json.loads((ROOT/'data/features.json').read_text())['feature_sets']['small'] if f != 'feature_country']
lag_features = [f'{f}_lag{w}w' for w in (1,2) for f in features]

def build(name):
    df = pd.read_parquet(ROOT/f'data/{name}_sample.parquet')
    df['date'] = pd.to_datetime(df['date'])
    assert not df.duplicated(['numerai_ticker','date']).any()
    # Exact 7/14-calendar-day matching, not "previous available row".
    # Histories use only past features, never past/future labels.
    for weeks in (1,2):
        past = df[['numerai_ticker','date']+features].copy()
        past['date'] += pd.Timedelta(weeks=weeks)
        past[f'_present{weeks}'] = True
        past.rename(columns={f:f'{f}_lag{weeks}w' for f in features}, inplace=True)
        df = df.merge(past, on=['numerai_ticker','date'], how='left', validate='one_to_one')
    # A real inference service needs the same history or an explicit fallback.
    df = df.loc[df['_present1'].eq(True) & df['_present2'].eq(True)].copy()
    return df.dropna(subset=[TARGET])

train,valid = build('train'),build('validation')
report = {'task':'historical weekly sequence comparison; not live evaluation',
          'target':TARGET,'train_rows':len(train),'validation_rows':len(valid),
          'train_dates':[str(train.date.min().date()),str(train.date.max().date())],
          'validation_dates':[str(valid.date.min().date()),str(valid.date.max().date())],
          'models':{}}
for name,cols in [('current_only',features),('three_weekly_snapshots',features+lag_features)]:
    model = XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.05,
                         subsample=0.8,colsample_bytree=0.8,tree_method='hist',
                         objective='reg:squarederror',n_jobs=4,random_state=42)
    print('training',name,len(train),len(cols),flush=True)
    model.fit(train[cols],train[TARGET])
    valid[name] = model.predict(valid[cols])
    ic = valid.groupby('date').apply(lambda d:d[name].corr(d[TARGET],method='spearman'),include_groups=False)
    report['models'][name] = {'n_features':len(cols),'mean_rank_ic':float(ic.mean()),
                              'positive_dates':int((ic>0).sum()),'n_dates':len(ic)}
    ic.rename(name).to_csv(OUT/f'{name}_ic.csv')
    model.save_model(OUT/f'{name}.json')
valid[['date','numerai_ticker',TARGET,'current_only','three_weekly_snapshots']].to_csv(OUT/'predictions.csv',index=False)
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print(valid[['date','numerai_ticker',TARGET,'current_only','three_weekly_snapshots']].head(5).to_string(index=False))
