python train_cat.py --train_path data/Train_te_kmeans.pkl --test_path data/Test_te_kmeans.pkl --n_splits 5 --seed 1024 --shuffle True
python train_lgb.py --train_path data/Train_te.pkl --test_path data/Test_te_kmeans.pkl --seed 256 --shuffle False
python train_xgb.py --train_path data/Train_te_kmeans.pkl --test_path data/Test_te_kmeans.pkl --n_splits 10 --seed 2021 --shuffle False