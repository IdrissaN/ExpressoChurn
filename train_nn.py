import os
import gc, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from torchmetrics import AUROC

from src.config import Config
from src.utils import seed_everything, seed_torch, AverageMeter, EarlyStopping

def preprocess_dataset(x, x_test, target = None):
    if target:
        x = x.copy().drop(target, 1)
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    qt = QuantileTransformer(n_quantiles=96, output_distribution='normal')
    bin_cat = KBinsDiscretizer(n_bins=96, encode='ordinal',strategy='uniform')
    
    x = imp.fit_transform(x)
    x = qt.fit_transform(x)
    x_bin = bin_cat.fit_transform(x)
    
    x_test = imp.transform(x_test)
    x_test = qt.transform(x_test)
    x_test_bin = bin_cat.transform(x_test)
    
    return x, x_bin, x_test, x_test_bin

class TabModel(nn.Module):
    def __init__(self, act_fn = nn.SELU(), dropout_num = 2):
        super().__init__()
        self.emb = nn.Embedding(96,18)
        self.fc = nn.Linear(110*18, 30)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) 
                                       for _ in range(dropout_num)])
        self.fc1 = nn.Linear(110,30)
        self.fc2 = nn.Linear(30*2,30)
        self.out = nn.Linear(30,1)
        self.act_fn = act_fn
        
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.xavier_normal_(self.emb.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
    
    def forward(self, x_bin, x):
        x_bin = self.emb(x_bin)
        x_bin = x_bin.view(-1,110*18)
        x_bin = self.act_fn(self.fc(x_bin))
        
        x = self.act_fn(self.fc1(x))
        x = torch.cat([x_bin,x], -1)
        x = self.act_fn(self.fc2(x))
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(x)
                out = self.out(out)
        
            else:
                temp_out = dropout(x)
                temp_out = self.out(temp_out)
                out += temp_out

        out /= len(self.dropouts)

        return torch.sigmoid(out)


class TabDataset(Dataset):
    def __init__(self, x, x_bin, target = None):
        super().__init__()
        self.x = x
        self.x_bin = x_bin
        self.target = target
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx, :]
        x_bin = self.x_bin[idx, :]
        
        _dict = {'x': torch.tensor(x, dtype = torch.float),
                 'x_bin': torch.tensor(x_bin, dtype = torch.long)}
        
        if self.target is not None:
            target = self.target[idx].item()
            _dict.update({'target': torch.tensor(target, dtype = torch.float)})
        
        return _dict

class Trainer:
    def __init__(self, model, device, loss_fn, opt, scheduler = None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.opt = opt
        self.scheduler = scheduler
        
    def fit_one_epoch(self, dl):
        self.model.train()
        losses = AverageMeter()
        prog_bar = tqdm(enumerate(dl), total = len(dl), file=sys.stdout, leave = False)
        
        for bi, d in prog_bar:
            x = d["x"].to(self.device)
            x_bin = d['x_bin'].to(self.device)
            target = d['target'].to(self.device)

            out = self.model(x_bin, x)
            loss = self.loss_fn(out.squeeze(-1), target)
            prog_bar.set_description('loss: {:.2f}'.format(loss.item()))
            losses.update(loss.item(), x.size(0))
            loss.backward()
            self.opt.step()
            
            if self.scheduler: 
                self.scheduler.step()
                    
            self.opt.zero_grad()

    def eval_one_epoch(self, dl, **kwargs):
        self.model.eval()
        losses = AverageMeter()
        metric = AUROC()
        prog_bar = tqdm(enumerate(dl), total = len(dl), file=sys.stdout, leave = False)
        
        for bi, d in prog_bar:  
            x = d["x"].to(self.device)
            x_bin = d['x_bin'].to(self.device)
            target = d['target'].to(self.device)
            
            with torch.no_grad():
                out = self.model(x_bin, x)
                loss = self.loss_fn(out.squeeze(-1), target)
                if metric:
                    auroc = metric(out.squeeze(-1), target.int())
                
                losses.update(loss.item(), x.size(0))
        auroc = metric.compute()
        print(f"F{kwargs['fold']} E{kwargs['epoch']}  Valid Loss: {losses.avg:.4f}  AUROC Score: {auroc:.4f}")
        
        return auroc.cpu() if metric else losses.avg


def create_dataloaders(splits, fold, x_train, y_train, x_bin):
    train_idx, valid_idx = splits[fold]
    
    _xtr, _xtr_bins, _ytr = x_train[train_idx], x_bin[train_idx], y_train[train_idx]
    _xval, _xval_bins, _yval = x_train[valid_idx], x_bin[valid_idx], y_train[valid_idx]
    
    train_ds = TabDataset(x = _xtr, x_bin = _xtr_bins, target = _ytr)
    valid_ds = TabDataset(x = _xval, x_bin = _xval_bins, target = _yval)
                          
    train_dl = DataLoader(train_ds, batch_size = cfg.batch_size, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = cfg.batch_size, shuffle = False)
    
    return train_dl, valid_dl


def train_fold(splits, fold, x_train, y_train, x_bin, epochs):
    train_dl, valid_dl = create_dataloaders(splits, fold, x_train, y_train, x_bin)
    es = EarlyStopping(patience = 7, mode="max", verbose = False)
    
    model = TabModel(dropout_num = 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr = cfg.learning_rate)
    scheduler = OneCycleLR(opt, 
                           max_lr=1e-3, 
                           steps_per_epoch=len(train_dl),
                           epochs = epochs)

    trainer = Trainer(model, 
                      device, 
                      loss_fn=nn.BCELoss(), 
                      opt = opt,
                      scheduler = scheduler,
                     )
    for epoch in range(epochs):
        trainer.fit_one_epoch(train_dl)
        valid_loss = trainer.eval_one_epoch(valid_dl, fold = fold, epoch = epoch)
        
        es(valid_loss, trainer.model, model_path = cfg.checkpoint(fold))
        
        if es.early_stop:
            break


if __name__=='__main__':

    cfg = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(cfg.seed)

    train = pd.read_pickle(os.path.join(cfg.PATH, 'Train_te_kmeans.pkl'))
    test = pd.read_pickle(os.path.join(cfg.PATH, 'Test_te_kmeans.pkl'))

    test_ids = test['user_id']

    FEATS = cfg.NN_FEATS
    CLUSTER_FEATS = train.filter(regex=("CLS_FEATS.*")).columns.tolist()
    FEATS.extend(CLUSTER_FEATS)
    FEATS.extend(cfg.DIFF_MEAN_FEATS)
    FEATS.extend(cfg.MEAN_FEATS)

    for col in ['UNLIMITED_CALL', 'FORFAIT_24h', 'FORFAIT_5d', 'FORFAIT_7d']:
        train[col].fillna(0, inplace=True)
        test[col].fillna(0, inplace=True)
        
    train_df = train[FEATS].copy()
    test_df = test[FEATS].copy()
    train_df['CHURN'] = train['CHURN']

    del train
    del test
    print(f"Number of feats : {len(FEATS)}")
    print('Preprocessing ...')

    x_train, x_bin, x_test, x_test_bin = preprocess_dataset(train_df, test_df, target = 'CHURN')
    y_train = train_df.CHURN.values

    print('Preprocessing done !')

    kfold = StratifiedKFold(n_splits = cfg.n_splits, 
                        random_state = cfg.seed, 
                        shuffle = True)

    splits = [*kfold.split(X = x_train, y = y_train)]

    seed_torch(cfg.seed)

    for fold in range(cfg.n_splits):
        train_fold(splits, fold, x_train, y_train, x_bin, cfg.epochs)

    
    y_pred = torch.zeros(len(x_test), 1).to(device)
    test_ds = TabDataset(x_test, x_test_bin)
    test_dl = DataLoader(test_ds, batch_size = cfg.batch_size, shuffle = False)

    print('Inference ...')
    with torch.no_grad():
        for fold in range(cfg.n_splits):
            preds = []
            model = TabModel(dropout_num = 2).to(device)
            state_dict = cfg.checkpoint(fold)
            model.load_state_dict(torch.load(state_dict))
            model.eval()
            
            for d in test_dl:
                x = d["x"].to(device)
                x_bin = d['x_bin'].to(device)
                out = model(x_bin, x)
                preds.append(out)          
            
            preds = torch.cat(preds, 0)
            y_pred += preds / cfg.n_splits
            
            z=gc.collect()

        sub = pd.DataFrame()
        sub['user_id'] = test_ids
        sub['CHURN'] = y_pred.cpu().numpy()
        sub.to_csv(os.path.join(cfg.submissions_path, f'sub_nn_{len(FEATS)}.csv'), index=False)