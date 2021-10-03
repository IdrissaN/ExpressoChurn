import gc, os
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from src.config import Config
from src.utils import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean_feats', help='1 if include mean features and 0 otherwise', type=bool, default=0)
    parser.add_argument('--diff_feats', help='1 if include difference features and 0 otherwise', type=bool, default=1)
    parser.add_argument('--log_feats', help='1 if include log features and 0 otherwise', type=bool, default=1)
    parser.add_argument('--quartiles_feats', help='1 if include quartiles features and 0 otherwise', type=bool, default=1)
    args = parser.parse_args()
    return args

def compute_mean_feats_categ(data, categ_col):
    temp_dv = data.groupby(categ_col)['DATA_VOLUME'].agg(['mean']).rename({'mean':f'DATA_VOLUME_{categ_col}_MEAN'},axis=1).reset_index()
    temp_fr = data.groupby(categ_col)['FREQUENCE_RECH'].agg(['mean']).rename({'mean':f'FREQUENCE_RECH_{categ_col}_MEAN'},axis=1).reset_index()
    temp_on_net = data.groupby(categ_col)['ON_NET'].agg(['mean']).rename({'mean':f'ON_NET_{categ_col}_MEAN'},axis=1).reset_index()
    temp_mt = data.groupby(categ_col)['MONTANT'].agg(['mean']).rename({'mean':f'MONTANT_{categ_col}_MEAN'},axis=1).reset_index()
    temp_rv = data.groupby(categ_col)['REVENUE'].agg(['mean']).rename({'mean':f'REVENUE_{categ_col}_MEAN'},axis=1).reset_index()
    temp_freq = data.groupby(categ_col)['FREQUENCE'].agg(['mean']).rename({'mean':f'FREQ_{categ_col}_MEAN'},axis=1).reset_index()
    temp_reg = data.groupby(categ_col)['REGULARITY'].agg(['mean']).rename({'mean':f'REG_{categ_col}_MEAN'},axis=1).reset_index()
    temp_org = data.groupby(categ_col)['ORANGE'].agg(['mean']).rename({'mean':f'ORANGE_{categ_col}_MEAN'},axis=1).reset_index()
    temp_tigo = data.groupby(categ_col)['TIGO'].agg(['mean']).rename({'mean':f'TIGO_{categ_col}_MEAN'},axis=1).reset_index()
    temp_freq_pack = data.groupby(categ_col)['FREQ_TOP_PACK'].agg(['mean']).rename({'mean':f'FREQ_TOP_PACK_{categ_col}_MEAN'},axis=1).reset_index()
    
    list_dfs = [temp_dv, temp_fr, temp_on_net, temp_mt, temp_rv, temp_freq, temp_reg, temp_org, temp_tigo, temp_freq_pack]
    dfs_merged = reduce(lambda  left, right: pd.merge(left, right, on=[categ_col], how='outer'), list_dfs)
    
    for temp in list_dfs:
        del temp
    
    return dfs_merged

def compute_diff_feats_to_mean(data, categ_col):
    data[f'DIFF_DV_MEAN_{categ_col}'] = data['DATA_VOLUME'] - data[f'DATA_VOLUME_{categ_col}_MEAN']
    data[f'DIFF_FR_MEAN_{categ_col}'] = data['FREQUENCE_RECH'] - data[f'FREQUENCE_RECH_{categ_col}_MEAN']
    data[f'DIFF_ON_NET_MEAN_{categ_col}'] = data['ON_NET'] - data[f'ON_NET_{categ_col}_MEAN']
    data[f'DIFF_MT_MEAN_{categ_col}'] = data['MONTANT'] - data[f'MONTANT_{categ_col}_MEAN']
    data[f'DIFF_RV_MEAN_{categ_col}'] = data['REVENUE'] - data[f'REVENUE_{categ_col}_MEAN']
    data[f'DIFF_FREQ_MEAN_{categ_col}'] = data['FREQUENCE'] - data[f'FREQ_{categ_col}_MEAN']
    data[f'DIFF_REG_MEAN_{categ_col}'] = data['REGULARITY'] - data[f'REG_{categ_col}_MEAN']
    data[f'DIFF_ORG_MEAN_{categ_col}'] = data['ORANGE'] - data[f'ORANGE_{categ_col}_MEAN']
    data[f'DIFF_TIGO_MEAN_{categ_col}'] = data['TIGO'] - data[f'TIGO_{categ_col}_MEAN']
    data[f'DIFF_FREQ_TOP_PACK_MEAN_{categ_col}'] = data['FREQ_TOP_PACK'] - data[f'FREQ_TOP_PACK_{categ_col}_MEAN']
    
    return data


def compute_feats(df, test_feats, tenure_categ_mapper, tenure_int_mapper):

    df['N_MISSING'] = df[test_feats].isna().sum(axis=1)
    df['AVG_MISSING_OP'] = df[['ON_NET', 'ORANGE', 'TIGO']].isna().sum(axis=1) / 3
    df['SUM_OPERATORS'] = df[['ON_NET', 'ORANGE', 'TIGO']].sum(axis=1)
    df['TS_ON_NET'] = df['ON_NET'] / df['SUM_OPERATORS']

    df['LOG_ARPU_SEGMENT'] = np.log1p(df['ARPU_SEGMENT'])
    df['LOG_DATA_VOLUME'] = np.log1p(df['DATA_VOLUME'])
    df['LOG_REVENUE'] = np.log1p(df['REVENUE'])
    df['LOG_MONTANT'] = np.log1p(df['MONTANT'])


    df['CD_TENURE'] = df['TENURE'].map(tenure_categ_mapper)
    df['AMOUNT_FREQ_RECH'] = df['MONTANT'] / df['FREQUENCE_RECH']
    df['RATIO_SPENT_RECH'] = df['MONTANT'] / (df['REVENUE'])
    df['RATIO_REGULARITY'] = df['REGULARITY'] / 90
    df['RATIO_DATA_VOLUME'] = df['DATA_VOLUME'] / 90
    df['UNLIMITED_CALL'] = df["TOP_PACK"].map(lambda x: 1 if "Unlimited" in str(x) else 0)

    df['MIN_TENURE'] = df['TENURE'].map(tenure_int_mapper).astype(int)
    df['INCOME_PER_AMOUNT'] = df['REVENUE'] / df['MONTANT']
    df['INCOME_AFT_AMOUNT'] = df['REVENUE'] - df['MONTANT']
    df['TENURE_TO_FREQ'] = df['MIN_TENURE'] / df['FREQUENCE_RECH']
    df['TENURE_REGULARITY'] = df['MIN_TENURE'] * df['REGULARITY']
    df['ACTIVE_REG'] =  df['MIN_TENURE'] / df['REGULARITY']
    df['DATA_VOLUME_PER_REG'] =  df['DATA_VOLUME'] / df['REGULARITY']
    df['DATA_VOLUME_REGULARITY'] =  df['DATA_VOLUME'] * df['REGULARITY']
    df['FRE_RECH_REGULARITY'] =  df['FREQUENCE_RECH'] * df['REGULARITY']
    df['ON_NET_REGULARITY'] =  df['ON_NET'] * df['REGULARITY']
    df['ORANGE_REGULARITY'] =  df['ORANGE'] * df['REGULARITY']
    df['TIGO_REGULARITY'] =  df['TIGO'] * df['REGULARITY']
    
    return df


if __name__=='__main__':
    
    tenure_categ_mapper = {
      "K > 24 month": "K", 
      "I 18-21 month": "I", 
      "G 12-15 month": "G", 
      "H 15-18 month": "H", 
      "J 21-24 month": "J", 
      "F 9-12 month": "F", 
      "D 3-6 month": "D", 
      "E 6-9 month": "E"}

    tenure_int_mapper = {
      "K > 24 month": 24, 
      "I 18-21 month": 18, 
      "G 12-15 month": 12, 
      "H 15-18 month": 15, 
      "J 21-24 month": 21, 
      "F 9-12 month": 9, 
      "D 3-6 month": 3, 
      "E 6-9 month": 6}
    
    cfg = Config()

    train = pd.read_csv(os.path.join(cfg.PATH, 'Train.csv'))
    test = pd.read_csv(os.path.join(cfg.PATH, 'Test.csv'))
    sub = pd.read_csv(os.path.join(cfg.PATH, 'SampleSubmission.csv'))

    len_train = len(train)
    len_test = len(test)

    y = train['CHURN']
    test_feats = [col for col in test.columns]
    all_data = pd.concat([train[test_feats], test])

    for df in [train, test, all_data]:
        df = reduce_mem_usage(df)
        
    z = gc.collect()
    
    df = all_data.copy()

    df = compute_feats(df, test_feats, tenure_categ_mapper, tenure_int_mapper)
    
    for agg_col in ['REGION', 'TENURE', 'TOP_PACK', 'UNLIMITED_CALL']:
        tmp_dfs_merged = compute_mean_feats_categ(df, categ_col=agg_col)
        df = pd.merge(df, tmp_dfs_merged, on=agg_col, how='left')
        df = compute_diff_feats_to_mean(df, agg_col)
        del tmp_dfs_merged

    z = gc.collect()

    train = df[:len_train]
    test = df[len_train:]

    X_train = train.copy()
    X_test = test.copy()

    # FREQUENCE ENCODE
    def encode_FE(df1, df2, cols):
        for col in cols:
            df = pd.concat([df1[col],df2[col]])
            vc = df.value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            nm = col+'_FE'
            df1[nm] = df1[col].map(vc)
            df1[nm] = df1[nm].astype('float32')
            df2[nm] = df2[col].map(vc)
            df2[nm] = df2[nm].astype('float32')

    # LABEL ENCODE
    def encode_LE(col, train=X_train, test=X_test):
        df_comb = pd.concat([train[col],test[col]],axis=0)
        df_comb,_ = df_comb.factorize(sort=True)
        nm = col
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')

    def encode_CB(col1, col2, df1=X_train, df2=X_test):
        nm = col1+'_'+col2
        df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
        df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 
        encode_LE(nm)
        
    encode_FE(X_train, X_test, ['REGION','TENURE','TOP_PACK', 'CD_TENURE'])
    encode_LE('REGION')
    encode_CB('REGION','TOP_PACK')
    encode_CB('REGION','TENURE')
    encode_CB('TENURE','TOP_PACK')
    
    X_train['CHURN'] = y
    
    X_train.to_csv(os.path.join(cfg.PATH, 'Train_FE_noq_v3.csv'), index=False)
    X_test.to_csv(os.path.join(cfg.PATH, 'Test_FE_noq_v3.csv'), index=False)