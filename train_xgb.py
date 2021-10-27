import os
import argparse
import time, gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from src.config import Config
from src.utils import *
from src.target_encoding import *

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train', type=str, default='data/Train_meta.csv', required=True)
    parser.add_argument('--test_path', help='path of Test', type=str, default='data/Test_meta.csv', required=True)
    parser.add_argument('--seed', default=56, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
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

        fl = FocalLoss(alpha=None, gamma=0)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]       
        trn_x, trn_y = trn[features].fillna(-999), y.iloc[trn_idx]
        val_x, val_y = val[features].fillna(-999), y.iloc[val_idx]
                
        te_params = {
                    #'n_estimators': 6747, 
                    'max_depth': 8, 
                    'learning_rate': 0.037483943745390796, 
                    'subsample': 0.8223791003856742, 
                    'colsample_bytree': 0.21861513850251216, 
                    'max_bin': 130, 
                    'reg_lambda': 4.5921618757053535, 
                    'reg_alpha': 18.14882394157521,
                    'seed':1202,
                    'eval_metric': 'auc',
                    'gpu_id': 0,
                    'tree_method': 'gpu_hist'}

        dtrain = xgb.DMatrix(trn_x.values, trn_y.values)
        dvalid = xgb.DMatrix(val_x.values, val_y.values)

        xgb_model = xgb.train(te_params, dtrain, 5000, evals=[(dtrain, "train"), (dvalid, "eval")],
                                obj=fl.lgb_obj, 
                                early_stopping_rounds=100, verbose_eval=500)
        
        oofs[val_idx] = special.expit(fl.init_score(trn_y) + xgb_model.predict(xgb.DMatrix(val_x.values)))
        preds += special.expit(fl.init_score(trn_y) + xgb_model.predict(xgb.DMatrix(test[features].fillna(-999).values))) / skf.n_splits
    
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

    submission.to_csv(os.path.join(cfg.submissions_path, f"sub_xgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}_fl_params_te_na.csv"), index=False)
    oof.to_csv(os.path.join(cfg.submissions_path, f"oof_xgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_seed{args.seed}_fl_params_te_na.csv"), index=False)