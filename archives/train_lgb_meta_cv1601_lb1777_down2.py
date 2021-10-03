import os
import time, gc
import pandas as pd
import numpy as np
import argparse

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.utils import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='path of Train', type=str, default='data/Train_meta.csv', required=True)
    parser.add_argument('--test_path', help='path of Test', type=str, default='data/Test_meta.csv', required=True)
    parser.add_argument('--seed', default=128, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--downsample', help='1 means no downsampling the majority class', default=1, type=int)
    args = parser.parse_args()
    return args

def train_model(train, test, features, cat_feats, target, n_splits, seed):

    oofs = np.zeros(train.shape[0])
    preds = np.zeros(test.shape[0])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=args.shuffle, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
        print("\n Fold [{}/{}] - ".format(fold + 1, skf.n_splits), "started at ", time.ctime())
        print('=='*35)
        
        trn, val = train.iloc[trn_idx], train.iloc[val_idx]
            
        trn_x, trn_y = trn[features], y.iloc[trn_idx]
        val_x, val_y = val[features], y.iloc[val_idx]

        best_params = {'n_estimators': 6118,  
                        'max_depth': 10, 
                        'num_leaves': 120, 
                        'learning_rate': 0.005532765646394192, 
                        'subsample': 0.672239781204874, 
                        'colsample_bytree': 0.6677733000135537, 
                        'max_bin': 285, 
                        'reg_lambda': 1.587587255617028, 
                        'reg_alpha': 15.584931117199323,
                        'metric': 'auc',
                        'n_jobs': 16,
                        'random_state': 278}

    
        lgb_model = LGBMClassifier(**best_params)

        bag_clf = bagging_classifier(lgb_model, args.downsample)
    
        bag_clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc',
                categorical_feature = cat_feats)
        
        oofs[val_idx] = bag_clf.predict_proba(val_x)[:, 1]
        preds += bag_clf.predict_proba(test[features])[:, 1] / skf.n_splits
    
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

    excluded_feats = ['CHURN', 'user_id', 'TENURE', 'MRG', 'TOP_PACK', 'ARPU_SEGMENT', 'REGION_TENURE', 'CD_TENURE', 'REGULARITY_BIN']
    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)
    excluded_feats.extend(cfg.LOG_FEATS)
    
    features = [col for col in train.columns if col not in excluded_feats]

    for df in [train, test]:
        df['REGION'] = df['REGION'].fillna(15)

    cat_cols = ['REGION']

    print(f"# features : {len(features)}")
    oofs, preds, score = train_model(train, test, features, cat_cols, y, args.n_splits, args.seed)

    submission = pd.DataFrame({'user_id': test.user_id, 'CHURN': preds})
    oof = pd.DataFrame({'user_id': train.user_id, 'CHURN': y, 'OOF': oofs})

    submission.to_csv(os.path.join(cfg.submissions_path, f"sub_lgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_down{args.downsample}_seed{args.seed}_meta.csv"), index=False)
    oof.to_csv(os.path.join(cfg.submissions_path, f"oof_lgb_feats{len(features)}_cv{str(score).split('.')[1][:7]}_spl{args.n_splits}_down{args.downsample}_seed{args.seed}_meta.csv"), index=False)