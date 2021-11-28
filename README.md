### How to run this code ?

* Create a **data** folder and put the competition data
* Create a **submissions** folder to store the submission files for each model.

Here we go ! 

The below command will create lot of features (mean per category, difference to mean, booleans and target encoding)
```python
python extract_feats_te.py
```
### Models

```python
python train_xgb.py --train_path data/Train_te.pkl --test_path data/Test_te.pkl
python train_lgb.py --train_path data/Train_te.pkl --test_path data/Test_te.pkl --n_splits 10 --seed 14
python train_cat.py --train_path data/Train_te.pkl --test_path data/Test_te.pkl
```
**Note :** the models are using different sets of features for diversity.

### Ensembling
```python
python power_average.py
```