
import os
import gc, sys
import random
import pandas as pd
import numpy as np

from src.config import Config


if __name__=='__main__':
    cfg = Config()
    train = pd.read_pickle(os.path.join(cfg.PATH, 'Train_te.pkl'))
    test = pd.read_pickle(os.path.join(cfg.PATH, 'Test_te.pkl'))

    training = train.sample(n=700_000, random_state=2003)

    X = training[['UNLIMITED_CALL', 'FORFAIT_24h', 'FORFAIT_5d', 'FORFAIT_7d',
                'TE_REGION', 'TE_TENURE', 'TE_TOP_PACK', 'TE_REGION_TENURE', 
                'TE_REGION_TOP_PACK', 'TE_TENURE_TOP_PACK', 'REGULARITY',
                'N_MISSING', 'N_MISSING_OP', 'N_MISSING_REV', 'N_MISSING_ZONE', 
                'N_MISSING_PACK', 'N_MISSING_FREQ_DT', 'N_MISSING_IDENTITY']]

