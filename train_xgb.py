import os
import argparse
import time, gc
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from src.config import Config
from src.utils import *
from src.target_encoding import *

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train', type=str, default='data/Train_te.pkl', required=True)
    parser.add_argument('--test_path', help='path of Test', type=str, default='data/Test_te.pkl', required=True)
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--n_splits', default=5, type=int)
    args = parser.parse_args()
    return args

def train_model(train, test, features, target, n_splits, seed):
    """
    Train a Cross-validation model, save oofs and predictions
    """

    oofs = np.zeros(train.shape[0])
    preds = np.zeros(test.shape[0])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=args.shuffle, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
        print("\n Fold [{}/{}] - ".format(fold + 1, skf.n_splits), "started at ", time.ctime())
        print('=='*35)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]       
        trn_x, trn_y = trn[features], y.iloc[trn_idx]
        val_x, val_y = val[features], y.iloc[val_idx]

        best_params = {'n_estimators': 2729, 
                        'max_depth': 10, 
                        'learning_rate': 0.01,
                        'subsample': 0.8,
                        'colsample_bytree': 0.4,
                        'gamma': 0, 
                        'reg_lambda': 1.22,
                        'reg_alpha': 7.1,
                        'max_bin': 335,
                        'seed':1202,
                        'eval_metric': 'auc',
                        'gpu_id': 0,
                        'tree_method': 'gpu_hist',
                        }
    
        clf = XGBClassifier(**best_params)
    
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', 
                verbose=500, 
                early_stopping_rounds=200)
        
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
    
    train = pd.read_pickle(args.train_path)
    test = pd.read_pickle(args.test_path)

    y = train.CHURN

    excluded_feats = ['CHURN', 'user_id', 'REGION', 'TENURE', 'CD_TENURE', 'MRG', 'ZONE1', 'ZONE2',
                     'TOP_PACK', 'ARPU_SEGMENT', 'REGULARITY_BIN', 'REGION_TENURE', 'REGION_TOP_PACK', 'TENURE_TOP_PACK']

    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)
    excluded_feats.extend(cfg.LOG_FEATS)

    features = [col for col in test.columns if col not in excluded_feats]
    print(f"# features : {len(features)}")
    oofs, preds, score = train_model(train, test, features, y, args.n_splits, args.seed)

    submission = pd.DataFrame({'user_id': test.user_id, 'CHURN': preds})
    oof = pd.DataFrame({'user_id': train.user_id, 'CHURN': y, 'OOF': oofs}) 

    np.save(f"oof_xgb_{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.npy", oof.to_numpy())
    np.save(f"sub_xgb_{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}.npy", submission.to_numpy())

    submission.to_csv(os.path.join(cfg.submissions_path, f"sub_xgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}_cs4_te.csv"), index=False)
    oof.to_csv(os.path.join(cfg.submissions_path, f"oof_xgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}_cs4_te.csv"), index=False)