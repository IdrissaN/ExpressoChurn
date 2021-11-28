import os
import time, gc
import pandas as pd
import numpy as np
import argparse
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train', type=str, default='data/Train_te.pkl', required=True)
    parser.add_argument('--test_path', help='path of Test', type=str, default='data/Test_te.pkl', required=True)
    parser.add_argument('--seed', default=14, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--n_splits', default=10, type=int)
    args = parser.parse_args()
    return args

def train_model(train, test, features, target, n_splits, seed, cat_feats=None):

    oofs = np.zeros(train.shape[0])
    preds = np.zeros(test.shape[0])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=args.shuffle, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
        print("\n Fold [{}/{}] - ".format(fold + 1, skf.n_splits), "started at ", time.ctime())
        print('=='*35)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]
            
        trn_x, trn_y = trn[features].fillna(-999), target.iloc[trn_idx]
        val_x, val_y = val[features].fillna(-999), target.iloc[val_idx]

        best_params = {'n_estimators': 6118,  
                        'max_depth': 10, 
                        'num_leaves': 120, 
                        'learning_rate': 0.005532765646394192, 
                        'subsample': 0.672239781204874, 
                        'colsample_bytree': 0.4, 
                        'max_bin': 285, 
                        'reg_lambda': 1.587587255617028, 
                        'reg_alpha': 15.584931117199323,
                        'metric': 'auc',
                        'n_jobs': 16,
                        'random_state': 278}

        fl = FocalLoss(alpha=None, gamma=0)

        training = lgb.Dataset(trn_x, trn_y, init_score=np.full_like(trn_y, fl.init_score(trn_y)))
        valid = lgb.Dataset(val_x, val_y, init_score=np.full_like(val_y, fl.init_score(val_y)))
    
        lgb_model = lgb.train(params=best_params,
                            train_set=training,
                            num_boost_round=10000,
                            valid_sets=(training, valid),
                            valid_names=('Training', 'Valid'),
                            early_stopping_rounds=100,
                            verbose_eval=300,
                            fobj=fl.lgb_obj,
                            feval=fl.lgb_eval 
                            )
        
        oofs[val_idx] = special.expit(fl.init_score(trn_y) + lgb_model.predict(val_x))
        preds += special.expit(fl.init_score(trn_y) + lgb_model.predict(test[features].fillna(-999))) / skf.n_splits

        val_auc = eval_auc(val_y, oofs[val_idx])

        print(f'Fold {fold + 1} ROC AUC Score : {val_auc}')
        del trn, val
        z = gc.collect()
    
    score = eval_auc(target, oofs)
    print('Full AUC score %.6f' % (score))

    return oofs, preds, score


if __name__=='__main__':

    cfg = Config()
    args = get_args()
    seed_everything(args.seed)
    
    train = pd.read_pickle(args.train_path)
    test = pd.read_pickle(args.test_path)
    y = train.CHURN

    excluded_feats = ['CHURN', 'user_id', 'REGION', 'TENURE', 'CD_TENURE', 'MRG', 'ZONE1', 'ZONE2',
                     'TOP_PACK', 'ARPU_SEGMENT', 'REGULARITY_BIN', 'REGION_TENURE', 'REGION_TOP_PACK', 'TENURE_TOP_PACK',
                     'TE_TOP_PACK', 'TE_REGION_TOP_PACK', 'TE_TENURE_TOP_PACK']
    
    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)

    features = [col for col in train.columns if col not in excluded_feats]
    cat_cols = None

    print(f"# features : {len(features)}")
    oofs, preds, score = train_model(train, test, features, y, args.n_splits, args.seed, cat_cols)

    submission = pd.DataFrame({'user_id': test.user_id, 'CHURN': preds})
    oof = pd.DataFrame({'user_id': train.user_id, 'CHURN': y, 'OOF': oofs})

    np.save(f"oof_lgb_{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.npy", oof.to_numpy())
    np.save(f"sub_lgb_{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.npy", submission.to_numpy())

    submission.to_csv(os.path.join(cfg.submissions_path, f"sub_lgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.csv"), index=False)
    oof.to_csv(os.path.join(cfg.submissions_path, f"oof_lgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.csv"), index=False)