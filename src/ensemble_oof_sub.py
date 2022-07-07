'''
General implementation of the hill-climbing procedure 
(Assumes as input a directory with files of the form oof_N.csv and sub_N.csv

The oof files contain both the target values and the out-of-fold values.
'''

import pandas as pd, numpy as np, os, sys
from sklearn.metrics import *
from src.utils import *

def get_oof_and_sub(PATH):
    FILES = os.listdir(PATH)
    OOF = np.sort( [f for f in FILES if 'oof' in f] )
    OOF_CSV = [pd.read_csv(PATH+k) for k in OOF]

    print('We have %i oof files...'%len(OOF))
    print(); print(OOF)

    SUB = np.sort( [f for f in FILES if 'sub' in f] )
    SUB_CSV = [pd.read_csv(PATH+k) for k in SUB]
    
    print('We have %i submission files...'%len(SUB))
    print(); print(SUB)

    a = np.array( [x for x in SUB ] )
    b = np.array( [ x for x in OOF ] )
    if len(a)!=len(b):
        print('ERROR submission files dont match oof files')
    else:
        for k in range(len(a)):
            if a[k]!=b[k]: print('ERROR submission files dont match oof files')

    return OOF, OOF_CSV, SUB, SUB_CSV


def get_models_and_weights(OOF, OOF_CSV, metric=eval_auc, sign=1,
                           RES=200, PATIENCE=10, TOL=0.0001, DUPLICATES=False):
    '''
    Determines the best models and associated weights using
    the hill-climbing procedure to maximize the CV.

    Writes ensemble_oof.csv to current directory.

    Inputs:
        OOF: List of oof filenames
        OOF_CSV: List of oof DataFrames containing predictions oof
                 predictions and true values.
        metric: The metric of interest to maximize, defaults to
                roc_auc_score; must be in sklearn.metrics
                Assumes the metric improves as the quantity increases.
        sign: 1 if you want to maximize the metric and -1 if you want to
              minimize the metric
        RES: The resolution used to scan for best weights s.t.
             1/RES is the delta in w.
        PATIENCE: Number of steps in weight-space to take in a given
                  model before moving on to the next one
        TOL: The tolerance for counting an increase to the metric as
             an improvement sufficient to add the model
        DUPLICATES: Whether or not to re-consider models already added
                    to the ensemble
    '''
    x = np.zeros(( len(OOF_CSV[0]),len(OOF) ))
    
    for k in range(len(OOF)):
        x[:,k] = OOF_CSV[k].OOF.values

    TRUE = OOF_CSV[0].CHURN.values

    all = []
    for k in range(x.shape[1]):
        met = sign*metric(OOF_CSV[0].CHURN,x[:,k])
        all.append(met)
        print('Model %i has OOF metric = %.6f'%(k,met))

    m = [np.argmax(all)]; w = []


    old = np.max(all);


    print('Ensemble metric = %.6f by beginning with model %i'%(old,m[0]))
    print()

    for kk in range(len(OOF)):

        # BUILD CURRENT ENSEMBLE
        md = x[:,m[0]]
        for i,k in enumerate(m[1:]):
            md = w[i]*x[:,k] + (1-w[i])*md

        # FIND MODEL TO ADD
        mx = 0; mx_k = 0; mx_w = 0
        print('Searching for best model to add... ')
        # TRY ADDING EACH MODEL
        for k in range(x.shape[1]):
            print(k,', ',end='')
            if not DUPLICATES and (k in m): 
                continue

            # EVALUATE ADDING MODEL K WITH WEIGHTS W
            bst_j = 0; bst = 0; ct = 0
            for j in range(RES):
                tmp = j/RES*x[:,k] + (1-j/RES)*md
                met = sign*metric(TRUE,tmp)
                if met>bst:
                    bst = met
                    bst_j = j/RES
                else: ct += 1
                if ct>PATIENCE: break
            if bst>mx:
                mx = bst
                mx_k = k
                mx_w = bst_j

        # STOP IF INCREASE IS LESS THAN TOL
        inc = mx-old
        if inc<=TOL:
            print(); print('No increase. Stopping.')
            break
            
        print(); #print(kk,mx,mx_k,mx_w,'%.5f'%inc)
        print('Ensemble metric = %.6f after adding model %i with weight %.3f. Increase of %.6f'%(mx,mx_k,mx_w,inc))
        print()

        old = mx; m.append(mx_k); w.append(mx_w)


    print('We are using models',m)
    print('with weights',w)
    print('and achieve ensemble metric = %.4f'%old)

    # GENERATE OOF ENSEMBLE PREDICTIONS
    md = x[:,m[0]]
    for i,k in enumerate(m[1:]):
        md = w[i]*x[:,k] + (1-w[i])*md

    df = OOF_CSV[0].copy()
    df.OOF = md
    df.to_csv('ensemble_oof.csv',index=False)

    return m, w


def build_sub_ensemble(SUB, SUB_CSV, models, weights):
    y = np.zeros(( len(SUB_CSV[0]),len(SUB) ))
    for k in range(len(SUB)):
        y[:,k] = SUB_CSV[k].CHURN.values

    md2 = y[:,models[0]]
    for i,k in enumerate(models[1:]):
        md2 = weights[i]*y[:,k] + (1-weights[i])*md2

    df = SUB_CSV[0].copy()
    df.CHURN = md2
    df.to_csv('ensemble_sub.csv',index=False)


if __name__ == "__main__":

    PATH = sys.argv[1]
    OOF, OOF_CSV, SUB, SUB_CSV = get_oof_and_sub(PATH)
    models, weights = get_models_and_weights(OOF, OOF_CSV)
    build_sub_ensemble(SUB, SUB_CSV, models, weights)