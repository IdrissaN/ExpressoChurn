import os
import optuna
import pandas as pd
from src.utils import *
from src.config import Config
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")



def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=4001)
    
    params = {
        'iterations':trial.suggest_int("iterations", 1000, 8000),
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
        'od_wait':trial.suggest_int('od_wait', 500, 2000),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.02,0.8),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
        'random_strength': trial.suggest_uniform('random_strength',10,50),
        'depth': trial.suggest_int('depth',1,15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'verbose': False,
        'task_type' : 'GPU',
        'devices' : '0'
    }
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 15)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.3, 1)

    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test,y_test)],
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    # validation prediction
    y_hat = model.predict_proba(X_test)[:,1]
    score = eval_auc(y_test, y_hat)
    
    return score

if __name__== '__main__':

    cfg = Config()

    data = pd.read_csv(os.path.join(cfg.PATH, 'Train_FE.csv'))
    excluded_feats = ['CHURN', 'user_id', 'REGION', 'TENURE', 'CD_TENURE', 'MRG', 'TOP_PACK']
    excluded_feats.extend(cfg.DIFF_QRTLS_FEATS)
    features = [col for col in data.columns if col not in excluded_feats]

    X = data[features]
    y = data.CHURN

    study = optuna.create_study(
        direction='maximize',
        study_name='CatbClf'
    )

    study.optimize(objective, n_trials=70)

    print(f"Best Trial: {study.best_trial.value}")
    print(f"Best Params: {study.best_trial.params}")
