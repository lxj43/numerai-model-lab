"""Public Numerai v6 + XGBoost teaching demo. Downloads selected columns only.
Run: .venv/bin/python run_demo.py (from this directory). Never uploads predictions.
"""
from pathlib import Path
import json
import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from xgboost import XGBRegressor

from paths import ROOT
DATA = ROOT / 'data'
OUT = ROOT / 'results'
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
VERSION = 'v6.0'
TARGET = 'target_jupiter_60'
offline = (DATA / 'synthetic.marker').exists()
napi = None if offline else NumerAPI(show_progress_bars=False)
round_number = 'synthetic' if offline else napi.get_current_round()
meta_path = DATA / 'features.json'
if not meta_path.exists():
    if offline:
        raise FileNotFoundError('Run generate_demo_data.py before the offline demo')
    napi.download_dataset(f'{VERSION}/features.json', str(meta_path))
features = [f for f in json.loads(meta_path.read_text())['feature_sets']['small']
            if f != 'feature_country']
# Remove the text country column before fitting numeric XGBoost.
assert 'feature_country' not in features

def parquet(name):
    if offline:
        raise FileNotFoundError('Synthetic workspace is missing local fixtures')
    query = 'query($filename:String!,$round:Int){dataset(filename:$filename,round:$round)}'
    url = napi.raw_query(query, {'filename':f'{VERSION}/{name}.parquet',
                                'round':round_number})['data']['dataset']
    return fsspec.open(url, 'rb', block_size=1024*1024)

def sample(name):
    path = DATA / f'{name}_sample.parquet'
    if path.exists():
        return pd.read_parquet(path)
    with parquet(name) as handle:
        pf = pq.ParquetFile(handle)
        columns = ['date', 'numerai_ticker'] + features + [TARGET]
        if name == 'train':
            # Conservative six-month buffer before validation starts in 2019.
            # Select two latest complete row groups ending before this cutoff.
            date_idx = pf.schema.names.index('date')
            eligible = [i for i in range(pf.num_row_groups)
                        if str(pf.metadata.row_group(i).column(date_idx).statistics.max)
                        <= '2018-06-01']
            groups = eligible[-2:]
            if len(groups) < 2:
                raise RuntimeError('Dataset layout changed; inspect chronological split.')
        else:
            groups = [0]
        print('reading', name, 'row groups', groups, flush=True)
        frame = pd.concat([pf.read_row_group(i, columns=columns).to_pandas()
                           for i in groups], ignore_index=True)
    frame = frame.dropna(subset=[TARGET])
    frame.to_parquet(path, index=False)
    return frame

train = sample('train')
valid = sample('validation')
assert pd.to_datetime(valid.date.min()) - pd.to_datetime(train.date.max()) > pd.Timedelta(days=180)
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                     objective='reg:squarederror', n_jobs=4, random_state=42)
print('training', len(train), 'validation', len(valid), flush=True)
model.fit(train[features], train[TARGET])
model.save_model(OUT / 'model.json')
valid['raw_prediction'] = model.predict(valid[features])
valid['prediction'] = valid.groupby('date')['raw_prediction'].rank(pct=True)
# Spearman Rank IC is an introductory diagnostic, NOT official NCORR/NMMC.
ics = valid.groupby('date').apply(
    lambda x: x['raw_prediction'].corr(x[TARGET], method='spearman'),
    include_groups=False)
ics.rename('spearman_rank_ic').to_csv(OUT / 'validation_by_date.csv')
valid[['date','numerai_ticker',TARGET,'raw_prediction','prediction']].to_csv(
    OUT / 'validation_predictions.csv', index=False)
if offline:
    live = pd.read_parquet(DATA / 'live_sample.parquet', columns=['date', 'numerai_ticker'] + features)
else:
    with parquet('live') as handle:
        live = pd.read_parquet(handle, columns=['date', 'numerai_ticker'] + features)
if live.date.nunique() != 1 or live.numerai_ticker.duplicated().any():
    raise RuntimeError('Unexpected live date/ticker layout.')
live['raw_prediction'] = model.predict(live[features])
live['prediction'] = live['raw_prediction'].rank(pct=True)
assert live.prediction.between(0,1).all()
live[['numerai_ticker','prediction']].to_csv(OUT / 'submission.csv', index=False)
live[['date','numerai_ticker']+features[:3]+['raw_prediction','prediction']].head(5).to_csv(
    OUT / 'live_examples.csv', index=False)
report = dict(version=VERSION, round=round_number, target=TARGET,
              features=features, n_features=len(features), synthetic=offline,
              train_rows=len(train), train_dates=[train.date.min(),train.date.max()],
              validation_rows=len(valid), validation_dates=[valid.date.min(),valid.date.max()],
              validation_n_dates=len(ics), mean_spearman_rank_ic=float(ics.mean()),
              positive_ic_fraction=float((ics>0).mean()),
              live_date=live.date.iloc[0], live_rows=len(live), uploaded=False)
(OUT / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='features'},indent=2))
print('VALIDATION EXAMPLES:')
print(valid[['date','numerai_ticker',features[0],TARGET,'raw_prediction','prediction']].head(5).to_string(index=False))
print('LIVE-FORMAT EXAMPLES:')
print(live[['date','numerai_ticker','raw_prediction','prediction']].head(5).to_string(index=False))
