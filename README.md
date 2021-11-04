The below command will create lot of features (mean per category, difference to mean, booleans and target encode somme categorical features)
```python
python extract_feats_te.py
``
xgboost : 10 folds cross validation using the above output dataset.

```python
python train_xgb.py --train_path data/Train_te.pkl --test_path data/Test_te.pkl --n_splits 10 --seed 56 --shuffle False
```