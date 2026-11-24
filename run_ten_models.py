"""Ten fixed prediction recipes on identical chronological samples.
Run with .venv/bin/python run_ten_models.py. No live submission or validation tuning.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'4')
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
import json,time,warnings,hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from paths import ROOT
OUT=ROOT/'ten_results'; OUT.mkdir(exist_ok=True)
TARGET='target_jupiter_60'
FEATURES=[f for f in json.loads((ROOT/'data/features.json').read_text())['feature_sets']['small'] if f!='feature_country']

def build(name):
    df=pd.read_parquet(ROOT/f'data/{name}_sample.parquet')
    df['date']=pd.to_datetime(df.date)
    for weeks in (1,2):
        past=df[['date','numerai_ticker']+FEATURES].copy()
        past['date']+=pd.Timedelta(weeks=weeks)
        past[f'_present{weeks}']=True
        past.rename(columns={f:f'{f}_lag{weeks}w' for f in FEATURES},inplace=True)
        df=df.merge(past,on=['numerai_ticker','date'],how='left',validate='one_to_one')
    return df.loc[df['_present1'].eq(True)&df['_present2'].eq(True)].dropna(subset=[TARGET]).reset_index(drop=True)

def main():
    train,valid=build('train'),build('validation')
    assert (valid.date.min()-train.date.max()).days>180
    original=pd.read_csv(ROOT/'sequence_results/predictions.csv',parse_dates=['date'])
    assert len(valid)==len(original)
    assert (valid.date==original.date).all()
    assert (valid.numerai_ticker==original.numerai_ticker).all()
    assert np.allclose(valid[TARGET],original[TARGET])
    X=train[FEATURES].to_numpy(dtype='float32');V=valid[FEATURES].to_numpy(dtype='float32')
    y=train[TARGET].to_numpy(dtype='float32')
    scale=lambda estimator:make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),estimator)
    impute=lambda estimator:make_pipeline(SimpleImputer(strategy='median'),estimator)
    specs=[
      ('current_only','单时点 XGBoost','提升树','当前100个特征，逐棵树修正预测误差。',100,'#139c8c',None,{'n_estimators':100,'max_depth':3,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'random_state':42}),
      ('three_weekly_snapshots','时序输入 XGBoost','时序特征','拼接当前、前一周、前两周的100个特征。',300,'#7860d9',None,{'n_estimators':100,'max_depth':3,'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,'random_state':42}),
      ('ridge','Ridge 岭回归','线性模型','学习线性加权关系，以L2正则约束系数。',100,'#3478c5',scale(Ridge(alpha=100.0)),{'alpha':100.0}),
      ('elastic_net','Elastic Net','线性模型','结合L1和L2正则，压缩并筛选特征权重。',100,'#d18b24',scale(ElasticNet(alpha=0.0001,l1_ratio=0.5,max_iter=2000,tol=0.0001,random_state=42)),{'alpha':0.0001,'l1_ratio':0.5,'max_iter':2000,'tol':0.0001}),
      ('random_forest','随机森林','袋装树','对多棵重采样决策树的预测取平均。',100,'#ce657a',impute(RandomForestRegressor(n_estimators=80,max_depth=8,min_samples_leaf=100,max_features=0.5,max_samples=0.7,n_jobs=4,random_state=42)),{'n_estimators':80,'max_depth':8,'min_samples_leaf':100,'max_features':0.5,'max_samples':0.7}),
      ('extra_trees','Extra Trees','随机树','使用更随机的分裂点，平均多棵树的预测。',100,'#57915c',impute(ExtraTreesRegressor(n_estimators=80,max_depth=8,min_samples_leaf=100,max_features=0.5,n_jobs=4,random_state=42)),{'n_estimators':80,'max_depth':8,'min_samples_leaf':100,'max_features':0.5}),
      ('hist_gb','Hist Gradient Boosting','提升树','将特征分箱，再逐步拟合预测误差。',100,'#b16cba',HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,max_depth=4,learning_rate=0.05,l2_regularization=1.0,early_stopping=False,random_state=42),{'max_iter':100,'max_leaf_nodes':15,'max_depth':4,'learning_rate':0.05,'l2_regularization':1.0,'early_stopping':False}),
      ('lightgbm','LightGBM','提升树','采用叶子优先的梯度提升树，捕捉非线性关系。',100,'#2e92a6',LGBMRegressor(n_estimators=100,num_leaves=15,max_depth=4,learning_rate=0.05,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=100,n_jobs=4,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True),{'n_estimators':100,'num_leaves':15,'max_depth':4,'learning_rate':0.05,'min_child_samples':100,'subsample':0.8,'colsample_bytree':0.8}),
      ('catboost','CatBoost','提升树','采用对称树结构的梯度提升；本例只用数值特征。',100,'#bd784d',CatBoostRegressor(iterations=100,depth=4,learning_rate=0.05,loss_function='RMSE',l2_leaf_reg=3,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False),{'iterations':100,'depth':4,'learning_rate':0.05,'l2_leaf_reg':3,'loss_function':'RMSE'}),
      ('mlp','MLP 神经网络','神经网络','两层隐藏层，学习特征之间的非线性组合。',100,'#6571c8',scale(MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',alpha=0.01,batch_size=4096,learning_rate_init=0.001,max_iter=80,tol=0.000001,n_iter_no_change=10,early_stopping=False,random_state=42)),{'hidden_layers':[32,16],'alpha':0.01,'batch_size':4096,'learning_rate_init':0.001,'max_iter':80,'early_stopping':False})
    ]
    report={'version':'v6.0','target':TARGET,'seed':42,'built_at':datetime.now().isoformat(timespec='seconds'),
            'train_rows':len(train),'validation_rows':len(valid),'train_dates':[str(train.date.min().date()),str(train.date.max().date())],
            'validation_dates':[str(valid.date.min().date()),str(valid.date.max().date())],
            'validation_n_dates':valid.date.nunique(),'protocol':'Fixed recipes; identical chronological samples; no validation-driven tuning; no live submission.',
            'packages':{p:version(p) for p in ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost']},'models':{}}
    predictions=valid[['date','numerai_ticker',TARGET]].copy()
    for key,name,family,description,n_features,color,model,params in specs:
        started=time.perf_counter(); captured=[]; actual_iterations=None
        print('RUN',name,flush=True)
        cache=OUT/f'{key}_prediction.npy'
        if key in ('current_only','three_weekly_snapshots'):
            pred=original[key].to_numpy(); source='Reused existing identical-sample XGBoost run';duration=None
        elif cache.exists() and (OUT/f'{key}_metadata.json').exists():
            meta=json.loads((OUT/f'{key}_metadata.json').read_text());pred=np.load(cache)
            assert len(pred)==len(valid)
            assert meta['params']==params
            duration=meta['seconds'];captured=meta['warnings'];source='Cached identical run'
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(X,y)
                pred=model.predict(V)
                if hasattr(model,'named_steps'):
                    estimator=list(model.named_steps.values())[-1]
                    if getattr(estimator,'n_iter_',None) is not None:
                        actual_iterations=int(np.max(estimator.n_iter_))
                captured=[str(w.message) for w in caught]
            duration=round(time.perf_counter()-started,3);source='Trained locally'
            np.save(cache,np.asarray(pred))
            (OUT/f'{key}_metadata.json').write_text(json.dumps({'params':params,'seconds':duration,'warnings':captured}))
        assert np.isfinite(pred).all()
        predictions[key]=pred
        bydate=predictions.groupby('date').apply(lambda g:g[key].corr(g[TARGET],method='spearman'),include_groups=False)
        assert bydate.notna().all()
        bydate.rename(key).to_csv(OUT/f'{key}_ic.csv')
        record={'name':name,'family':family,'description':description,'n_features':n_features,'color':color,'params':params,
                'mean_rank_ic':float(bydate.mean()),'positive_dates':int((bydate>0).sum()),'n_dates':len(bydate),
                'ic_std':float(bydate.std()),'actual_iterations':actual_iterations,'seconds':duration,'source':source,'warnings':captured}
        report['models'][key]=record
        (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(name,'mean IC',round(record['mean_rank_ic'],6),'seconds',duration,flush=True)
    predictions.to_parquet(OUT/'predictions.parquet',index=False)
    predictions.to_csv(OUT/'predictions.csv',index=False)
    print('COMPLETE',len(report['models']),'models',flush=True)
if __name__=='__main__':main()
