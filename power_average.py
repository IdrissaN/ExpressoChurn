import os
import argparse
import numpy as np
import pandas as pd
from src.config import Config


cfg = Config()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs_path', help='path of submissions', type=str, default=cfg.submissions_path, required=True)
    parser.add_argument('--power', default=2, type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    args = get_args()
    PATH = args.subs_path
    FILES = os.listdir(PATH)

    SUB = np.sort( [f for f in FILES if 'sub' in f] )
    SUB_CSV = [pd.read_csv(os.path.join(PATH,k)) for k in SUB]
    pred_x = np.zeros(( len(SUB_CSV[0]), len(SUB) ))

    for k in range(len(SUB)):
        pred_x[:, k] = SUB_CSV[k].CHURN.values**args.power

    df = SUB_CSV[0].copy()
    df.CHURN = np.mean(pred_x, axis=1)
    df.to_csv(f'sub_models_{len(SUB_CSV)}_power_{args.power}.csv', index=False)