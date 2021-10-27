import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class TargetEncoding(object):
    """
    Mean Target Encoding

    """

    def __init__(self, C=15):
        self.C = C

    def fit(self, data_train, data_test, feature, target):

        skf = KFold(n_splits=5, random_state=56, shuffle=True)

        data_list = []
        for train_index, test_index in skf.split(data_train):
            enc_train = data_train.loc[train_index, [feature, target]].copy()
            enc_test = data_train.loc[test_index, [feature, target]].copy()

            global_mean = np.mean(enc_train[target])
            groupby_feature = enc_train.groupby(feature)
            current_mean = groupby_feature[target].mean()
            current_size = groupby_feature.size()
            feat_df = ((current_mean * current_size + global_mean * self.C) /
                       (current_size + self.C)).fillna(global_mean)
            values = pd.DataFrame(feat_df, columns=[f"TE_{feature}"], dtype=np.float64)
            data_list.append(enc_test.merge(values, how="left", left_on=feature, right_index=True)[[f"TE_{feature}"]].fillna(global_mean))

        global_mean = np.mean(data_train[target])
        groupby_feature = data_train.groupby([feature])
        current_mean = groupby_feature[target].mean()
        current_size = groupby_feature.size()
        feat_df = ((current_mean * current_size + global_mean * self.C) /
                   (current_size + self.C)).fillna(global_mean)
        values = pd.DataFrame(feat_df, columns=[f"TE_{feature}"], dtype=np.float64)

        data_train = data_train.join(pd.concat(data_list))

        data_test = data_test.merge(values, how="left", left_on=feature, right_index=True)
    
        return data_train, data_test