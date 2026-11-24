"""Download one later official validation block for a separately locked final test.
Only date ranges and counts are printed; do not evaluate candidate scores here.
"""
from pathlib import Path
import json,fsspec
import pyarrow.parquet as pq
from numerapi import NumerAPI
from paths import ROOT
path=ROOT/'data/agent_holdout.parquet'
if not path.exists():
 features=[f for f in json.loads((ROOT/'data/features.json').read_text())['feature_sets']['small'] if f!='feature_country']
 n=NumerAPI(show_progress_bars=False)
 url=n.raw_query('query($filename:String!){dataset(filename:$filename)}',{'filename':'v6.0/validation.parquet'})['data']['dataset']
 with fsspec.open(url,'rb',block_size=1024*1024) as handle:
  pf=pq.ParquetFile(handle);idx=pf.schema.names.index('date')
  eligible=[i for i in range(pf.num_row_groups) if str(pf.metadata.row_group(i).column(idx).statistics.min)>='2020-03-01']
  if not eligible:raise RuntimeError('No suitable later test block')
  group=eligible[0];print('Downloading later test block',group,flush=True)
  table=pf.read_row_group(group,columns=['date','numerai_ticker']+features+['target_jupiter_60'])
  pq.write_table(table,path)
  (ROOT/'agent_results').mkdir(exist_ok=True)
  (ROOT/'agent_results/holdout_manifest.json').write_text(json.dumps({'dataset':'v6.0/validation.parquet','row_group':group,'cutoff_rule':'first full row group starting on/after 2020-03-01; selected without scoring'},indent=2))
import pandas as pd
d=pd.read_parquet(path,columns=['date']);print('Later raw block:',len(d),'rows;',d.date.min(),d.date.max(),flush=True)
