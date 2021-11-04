
import os
import gc, sys
import random
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
from src.utils import seed_torch, AverageMeter, EarlyStopping


class TabModel(nn.Module):
    def __init__(self, act_fn = nn.SELU(), dropout_num = 2):
        super().__init__()
        self.emb = nn.Embedding(98,16)
        self.fc = nn.Linear(98*16, 30)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) 
                                       for _ in range(dropout_num)])
        self.fc1 = nn.Linear(98,30)
        self.fc2 = nn.Linear(30*2,30)
        self.out = nn.Linear(30,1)
        self.act_fn = act_fn
        
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.xavier_normal_(self.emb.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
    
    def forward(self, x_bin, x):
        x_bin = self.emb(x_bin)
        x_bin = x_bin.view(-1,98*16)
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

def create_dataloaders(fold):
    train_idx, valid_idx = splits[fold]
    
    _xtr, _xtr_bins, _ytr = x_train[train_idx], x_bin[train_idx], y_train[train_idx]
    _xval, _xval_bins, _yval = x_train[valid_idx], x_bin[valid_idx], y_train[valid_idx]
    
    train_ds = TabDataset(x = _xtr, x_bin = _xtr_bins, target = _ytr)
    valid_ds = TabDataset(x = _xval, x_bin = _xval_bins, target = _yval)
                          
    train_dl = DataLoader(train_ds, batch_size = cfg.bs, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = cfg.bs, shuffle = False)
    
    return train_dl, valid_dl


if __name__=='__main__':
    cfg = Config()
    train = pd.read_pickle(os.path.join(cfg.PATH, 'Train_te.pkl'))
    test = pd.read_pickle(os.path.join(cfg.PATH, 'Test_te.pkl'))


## IN PROGRESS