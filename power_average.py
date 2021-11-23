import os
import argparse
import numpy as np
import pandas as pd
from src.config import Config
from src.utils import *


cfg = Config()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs_path', help='path of submissions', type=str, default=cfg.submissions_path)
    parser.add_argument('--power', default=2, type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    args = get_args()
    PATH = args.subs_path
    FILES = os.listdir(PATH)

    OOF = np.sort( [f for f in FILES if 'oof' in f] )
    OOF_CSV = [pd.read_csv(os.path.join(PATH,k)) for k in OOF]
    oof_x_pow = np.zeros(( len(OOF_CSV[0]), len(OOF) ))
    oof_x_ens = np.zeros(( len(OOF_CSV[0]), len(OOF) ))

    SUB = np.sort( [f for f in FILES if 'sub' in f] )
    SUB_CSV = [pd.read_csv(os.path.join(PATH,k)) for k in SUB]
    pred_x_pow = np.zeros(( len(SUB_CSV[0]), len(SUB) ))
    pred_x_ens = np.zeros(( len(SUB_CSV[0]), len(SUB) ))
    
    for k in range(len(OOF)):
        oof_x_pow[:,k] = OOF_CSV[k].OOF.values**args.power
        oof_x_ens[:,k] = OOF_CSV[k].OOF.values

    for k in range(len(SUB)):
        pred_x_pow[:, k] = SUB_CSV[k].CHURN.values**args.power
        pred_x_ens[:, k] = SUB_CSV[k].CHURN.values

    TRUE = OOF_CSV[0].CHURN.values
    OOF_AUC_POW = eval_auc(TRUE, np.mean(oof_x_pow, axis=1))
    OOF_AUC_ENS = eval_auc(TRUE, np.mean(oof_x_ens, axis=1))

    OOF_AUC_ENS_POW = eval_auc(TRUE, (0.5*np.mean(oof_x_ens, axis=1) + 0.5*np.mean(oof_x_pow, axis=1)))

    print(f"OOF Power Average {args.power} AUC : {OOF_AUC_POW}")
    print(f"OOF Ensemble AUC : {OOF_AUC_ENS}")
    print(f"OOF POWER ENS AUC : {OOF_AUC_ENS_POW}")

    df = SUB_CSV[0].copy()
    df.CHURN = np.mean(pred_x_pow, axis=1)
    df.to_csv(f'pow_avg_te_n_models_{len(SUB_CSV)}_pow_{args.power}.csv', index=False)

    df = SUB_CSV[0].copy()
    df.CHURN = np.mean(pred_x_ens, axis=1)
    df.to_csv(f'blend_te_n_models_{len(SUB_CSV)}.csv', index=False)