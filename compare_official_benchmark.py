"""Score official v6 benchmark predictions on exactly the saved local sample keys.
These are local Spearman diagnostics, not official NCORR/NMMC leaderboard scores.
"""
from pathlib import Path
import json
import pandas as pd
import fsspec,pyarrow.parquet as pq
from numerapi import NumerAPI
from paths import ROOT
if (ROOT/'data/synthetic.marker').is_file():
    raise SystemExit('Official benchmark comparisons require actual Numerai data.')
OUT=ROOT/'official_benchmark';OUT.mkdir(exist_ok=True)
path=OUT/'validation_predictions.parquet'
COL='lgbm_jupiter_60'
if not path.exists():
 n=NumerAPI(show_progress_bars=False)
 url=n.raw_query('query($filename:String!){dataset(filename:$filename)}',{'filename':'v6.0/validation_benchmark_models.parquet'})['data']['dataset']
 with fsspec.open(url,'rb',block_size=1024*1024) as handle:
  pf=pq.ParquetFile(handle);di=pf.schema.names.index('date')
  groups=[]
  for i in range(pf.num_row_groups):
   st=pf.metadata.row_group(i).column(di).statistics
   if str(st.min)<='2020-11-06' and str(st.max)>='2019-01-18':groups.append(i)
  print('Reading official prediction groups',groups,flush=True)
  df=pd.concat([pf.read_row_group(i,columns=['date','numerai_ticker',COL]).to_pandas() for i in groups],ignore_index=True)
  df=df.loc[(df.date.astype(str)>='2019-01-18')&(df.date.astype(str)<='2020-11-06')].copy()
  df.to_parquet(path,index=False)
official=pd.read_parquet(path);official['date']=pd.to_datetime(official.date)
assert not official.duplicated(['date','numerai_ticker']).any()
report={'source':'v6.0/validation_benchmark_models.parquet','benchmark_column':COL,'metric':'mean per-date Spearman Rank IC','not_official_neutral_score':True,'periods':{}}
for label,file in [('development','ten_results/predictions.parquet'),('later_test','agent_results/final_test_predictions.parquet')]:
 local=pd.read_parquet(ROOT/file);local['date']=pd.to_datetime(local.date)
 joined=local[['date','numerai_ticker','target_jupiter_60']].merge(official,on=['date','numerai_ticker'],how='left',validate='one_to_one')
 assert len(joined)==len(local) and joined[COL].notna().all()
 ic=joined.groupby('date').apply(lambda g:g[COL].corr(g.target_jupiter_60,method='spearman'),include_groups=False)
 assert ic.notna().all()
 ic.rename('official_benchmark_rank_ic').to_csv(OUT/f'{label}_ic.csv')
 joined.to_parquet(OUT/f'{label}_aligned.parquet',index=False)
 report['periods'][label]={'rows':len(joined),'dates':[str(joined.date.min().date()),str(joined.date.max().date())],'n_dates':len(ic),'mean_rank_ic':float(ic.mean()),'positive_dates':int((ic>0).sum())}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
