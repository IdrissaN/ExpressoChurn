import os
import optuna
import pandas as pd
from src.utils import *
from src.config import Config
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=401)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 8000),
        'boosting': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'num_leaves': trial.suggest_int('num_leaves', 7, 120),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.2),
        'subsample': trial.suggest_uniform('subsample', 0.40, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.30, 0.80),
        'max_bin': trial.suggest_int('max_bin', 14, 400),
        'reg_lambda':  trial.suggest_uniform('reg_lambda', 0, 20),
        'reg_alpha':  trial.suggest_uniform('reg_alpha', 0, 20),
        'random_state': 278
        #'device_type': 'gpu'
    }


    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test,y_test)],
        verbose=0,
        early_stopping_rounds=100
    )
    
    y_hat = model.predict_proba(X_test)[:,1]
    score = eval_auc(y_test, y_hat)
    
    return score

if __name__== '__main__':

    cfg = Config()

    data = pd.read_pickle(os.path.join(cfg.PATH, 'Train_te.pkl'))
    excluded_feats = ['CHURN', 'user_id', 'CD_TENURE', 'MRG', 'ZONE1', 'ZONE2', 'REGION', 'TENURE',
                     'TOP_PACK', 'ARPU_SEGMENT', 'REGULARITY_BIN', 'REGION_TOP_PACK', 'REGION_TENURE', 'TENURE_TOP_PACK']
    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)
    features = [col for col in data.columns if col not in excluded_feats]

    X = data[features]
    y = data.CHURN

    study = optuna.create_study(
        direction='maximize',
        study_name='LGBClf'
    )

    study.optimize(objective, n_trials=100)

    print(f"Best Trial: {study.best_trial.value}")
    print(f"Best Params: {study.best_trial.params}")