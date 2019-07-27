import pandas as pd
import numpy as np
from czx_tools_pack_local.Feature_tools import Feature_names, factor_data, freq_feature,factor_data
from scipy.stats import skew, kurtosis
from scipy import stats


def feature_get_func(train_df,test_df, mode='train'):
    train_df['data_flag'] = 1
    test_df['data_flag'] = 0
    data_all = pd.concat([train_df, test_df])
    
    useful_feature = Feature_names()
    
    for col in useful_feature.categorical_columns:
        data_all.loc[:, col] = data_all[col].astype('category')
        
    print('feature finish')
    return data_all, useful_feature

