import os
from random import shuffle
import time, gc
import pandas as pd
import numpy as np
import argparse

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils import *

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train file', type=str, default='data/Train_FE.csv')
    parser.add_argument('--seed', default=824679, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--shuffle', default=False, type=int)
    parser.add_argument('--replace_na', default=False, type=bool)
    parser.add_argument('--na_value', default=-555, type=int)
    args = parser.parse_args()
    return args

def train_model(train, features, target, n_splits, seed):

    oofs = np.zeros(train.shape[0])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=args.shuffle, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
        print("\n [{}/{}] - Fold n°{}".format(fold + 1, skf.n_splits, fold + 1), "started at ", time.ctime())
        print('=='*30)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]
            
        trn_x, trn_y = trn[features], y.iloc[trn_idx]
        val_x, val_y = val[features], y.iloc[val_idx]

    
        clf = XGBClassifier(
                    seed=1202,
                    n_estimators=10000,
                    verbosity=1,
                    eval_metric="auc",
                    tree_method="gpu_hist",
                    gpu_id=0,
                    alpha=7.10,
                    colsample_bytree=0.6,
                    reg_lambda=0.25,
                    learning_rate=0.01,
                    max_bin=338,
                    max_depth=8,
                    min_child_weight=6.28,
                    subsample=0.8,
                    )
    
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', 
                verbose=500, 
                early_stopping_rounds=100)
        
        oofs[val_idx] = clf.predict_proba(val_x)[:, 1]
    
        print(f'Fold {fold + 1} ROC AUC Score : {eval_auc(val_y, oofs[val_idx])}')
        del trn, val
        z = gc.collect()
    
    score = eval_auc(target, oofs)
    print('Full AUC score %.6f' % (score))

    return oofs, score


if __name__=='__main__':

    cfg = Config()
    args = get_args()
    seed_everything(args.seed)
    
    train = pd.read_csv(args.train_path)

    training = train.sample(n=700_000, random_state=args.seed)

    #if args.replace_na:
        #training.fillna(args.na_value, inplace=True)
        
    y = training.CHURN

    excluded_feats = ['CHURN', 'user_id', 'REGION', 'TENURE', 'CD_TENURE', 'MRG', 'TOP_PACK']
    features = [col for col in training.columns if col not in excluded_feats]
    oofs, score = train_model(training, features, y, args.n_splits, args.seed)

    # train.sample(n=700_000, random_state=args.seed)
    # 0.931214 with NAN 5 folds all cols
    # 0.931182 replace NAN 5 folds all cols






