import os
import time, gc
import pandas as pd
import numpy as np
import argparse

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train', type=str, default='data/Train_FE.csv', required=True)
    parser.add_argument('--test_path', help='path of Test', type=str, default='data/Test_FE.csv', required=True)
    parser.add_argument('--seed', default=128, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--n_splits', default=5, type=int)
    args = parser.parse_args()
    return args

def train_model(train, test, features, target, n_splits, seed):

    oofs = np.zeros(train.shape[0])
    preds = np.zeros(test.shape[0])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=args.shuffle, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
        print("\n Fold [{}/{}] - ".format(fold + 1, skf.n_splits), "started at ", time.ctime())
        print('=='*35)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]
            
        trn_x, trn_y = trn[features], y.iloc[trn_idx]
        val_x, val_y = val[features], y.iloc[val_idx]

    
        clf = LGBMClassifier(
                n_estimators=8000,
                learning_rate=0.03,
                metric='auc',
                reg_alpha=0.85,
                reg_lambda=0.15,
                min_child_weight=9.75,
                colsample_bytree=0.5,
                n_jobs = 16,
                random_state = fold * 455 + 1
                )
    
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', 
                verbose=500, 
                early_stopping_rounds=150)
        
        oofs[val_idx] = clf.predict_proba(val_x)[:, 1]
        preds += clf.predict_proba(test[features])[:, 1] / skf.n_splits
    
        print(f'Fold {fold + 1} ROC AUC Score : {eval_auc(val_y, oofs[val_idx])}')
        del trn, val
        z = gc.collect()
    
    score = eval_auc(target, oofs)
    print('Full AUC score %.6f' % (score))

    return oofs, preds, score


if __name__=='__main__':

    cfg = Config()
    args = get_args()
    seed_everything(args.seed)
    
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    

    y = train.CHURN

    excluded_feats = ['CHURN', 'user_id', 'REGION', 'TENURE', 'CD_TENURE', 'MRG', 'TOP_PACK']
    #excluded_feats.extend(cfg.MEAN_FEATS)
    #excluded_feats.extend(cfg.DIFF_MEAN_FEATS)
    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)
    excluded_feats.extend(cfg.LOG_FEATS)

    features = [col for col in train.columns if col not in excluded_feats]
    print(f"# features : {len(features)}")
    oofs, preds, score = train_model(train, test, features, y, args.n_splits, args.seed)

    submission = pd.DataFrame({'user_id': test.user_id, 'CHURN': preds})
    oof = pd.DataFrame({'user_id': train.user_id, 'CHURN': y, 'OOF': oofs})

    submission.to_csv(os.path.join(cfg.submissions_path, f"sub_lgb_feats{len(features)}_cv{str(score).split('.')[1][:5]}_spl{args.n_splits}_seed{args.seed}.csv"), index=False)
    oof.to_csv(os.path.join(cfg.submissions_path, f"oof_lgb_feats{len(features)}_cv{str(score).split('.')[1][:5]}_spl{args.n_splits}_seed{args.seed}.csv"), index=False)







