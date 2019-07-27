import numpy as np
import pandas as pd
import math
import json
import os


def write_to_file(dictObj, file_path):
    f = open(file_path, 'w')
    f.write(str(dictObj))
    f.close()


def read_dict_from_file(file_path):
    f = open(file_path, 'r')
    a = f.read()
    dict_name = eval(a)
    return dict_name


class Feature_names(object):
    
    def __init__(self):
        self.categorical_columns = []
        self.numerical_columns = []
    
    def feature_add(self, col_name, feature_type):
        """
        :param col_name:
        :param feature_type: 'num' or 'cate'
        :return:
        """
        if isinstance(col_name, str):
            if col_name in self.numerical_columns:
                if feature_type == 'num':
                    print(col_name + ' already in numrical')
                    return
                else:
                    raise (col_name + ' already in numrical')
            if col_name in self.categorical_columns:
                if feature_type == 'cate':
                    print(col_name + ' already in categorical_columns')
                    return
                else:
                    raise (col_name + ' already in categorical_columns')
            if feature_type == 'num':
                self.numerical_columns.append(col_name)
            elif feature_type == 'cate':
                self.categorical_columns.append(col_name)
        
        elif isinstance(col_name, list):
            alin = []
            for col in col_name:
                if col in self.numerical_columns:
                    if feature_type == 'num':
                        print(col + ' already in numrical')
                    else:
                        raise (col + ' already in numrical')
                elif col in self.categorical_columns:
                    if feature_type == 'cate':
                        print(col + ' already in categorical_columns')
                    else:
                        raise (col + ' already in categorical_columns')
                else:
                    alin.append(col)
            if feature_type == 'num':
                self.numerical_columns += alin
            elif feature_type == 'cate':
                self.categorical_columns += alin
    
    def feature_remove(self, col_name):
        if isinstance(col_name, str):
            if col_name in self.categorical_columns:
                self.categorical_columns.remove(col_name)
            elif col_name in self.numerical_columns:
                self.numerical_columns.remove(col_name)
            else:
                print(col_name + ' not in used feature')
        if isinstance(col_name, list):
            for col in col_name:
                if col in self.categorical_columns:
                    self.categorical_columns.remove(col)
                elif col in self.numerical_columns:
                    self.numerical_columns.remove(col)
                else:
                    print(col + ' not in used feature')
    
    def feature_all(self):
        return self.categorical_columns + self.numerical_columns
    
    def print_fea(self):
        print('categorical_columns:', self.categorical_columns)
        print('numrical_columns:', self.numerical_columns)


def freq_feature(df, count_col, step=None, nan_fill=-1, log=False, norm=False, mode='train', path=None):
    """
    
    :param df: origin dataframe
    :param count_col: count which to be count
    :param step: 步长,尽在连续值时使用
    :param nan_fill:
    :param log: whether use log
    :param norm: whether use norm
    :return: counted dataframe, name of new fea, name of bin
    """
    count_fea_name = count_col + '_cnt'
    bins_name = count_col + '_bins_name'
    
    if mode == 'test':
        map_dict = read_dict_from_file(os.path.join(path, count_fea_name + '.json'))
        df.loc[:,count_fea_name] = df[count_col].fillna(nan_fill).map(map_dict).values
        return df, count_fea_name, bins_name
    
    if step:
        # fanwei = list(range(0, df[count_col].max(), scope))  # 设置区间范围
        fanwei = list(np.arange(df[count_col].min(), df[count_col].max(), step))
        fenzu = pd.cut(df[count_col].fillna(nan_fill).values, fanwei, right=False)  # 分组类别
        df.loc[:,bins_name] = fenzu  # 分组放入原df
        df.loc[:,bins_name]=  df[bins_name].astype('str')
        cnt = df[bins_name].value_counts()
        
        value_count_dict = dict(cnt)
        df.loc[:,count_fea_name] = df[bins_name].map(value_count_dict)  # 得出 count feature
        if log:
            ln = 1 / df[count_col].nunique()
            df.loc[:,count_fea_name] = (df[count_fea_name] + ln).map(math.log)
        if norm:
            df[count_fea_name] = (df[count_fea_name] - df[count_fea_name].min()) / (
                    df[count_fea_name].max() - df[count_fea_name].min())
    else:
        value_count_dict = dict(df[count_col].fillna(nan_fill).value_counts())
        df.loc[:,count_fea_name] = df[count_col].fillna(nan_fill).map(value_count_dict).values
        if log:
            df.loc[:, count_fea_name] = df[count_fea_name].apply(lambda x:np.log(x))
        bins_name = count_col
    if path is not None:
        write_to_file(value_count_dict, os.path.join(path, count_fea_name + '.json'))
    return df, count_fea_name, bins_name


def factor_data(df, col, mode='train',path=None):
    """
    完成类别数字编码
    :param df:
    :param col: 待编码列名称
    :return: encoded dataframe, max id
    """
    ecnode_values, iundex = df.loc[:, col].factorize(sort=True)  # id编码，nan值会编为-1
    # MAKE SMALLEST LABEL 1, RESERVE 0
    if mode == 'train':
        maop_dict = dict(zip(iundex, range(len(iundex))))
        if path is not None:
            write_to_file(maop_dict, os.path.join(path, col+ '_cate_encode.json'))
    else:
        maop_dict = read_dict_from_file(os.path.join(path, col+ '_cate_encode.json'))

    df.loc[:, col] = df[col].map(maop_dict)
    df.loc[df[col].isnull(), col] = -1
    df.loc[:, col] += 1
    # MAKE NAN LARGEST LABEL
    new_code = np.where(df[col] == 0, df[col].max() + 1, df[col])
    df.loc[:, col] = new_code  # 让nan作为最大编码
    max_id = df[col].max() + 1
    return df, max_id


# STATISTICAL CATEGORY ENCODE
def category_filter_by_target(train_df, test_df, col, filter=0.001, zscore=1, tar='HasDetections', m=0.5, verbose=1):
    """
    针对类别特征，过滤掉
    :param train_df: 训练集df
    :param test_df:测试集df
    :param col: 待过滤类别特征列名称
    :param filter:用户控制 滤掉出现次数较少的类别。出现次数多少算少？  < filter*len(train_df) 的算少
    :param zscore:
    :param tar: label标签列名
    :param m:
    :param verbose:
    :return: train_df, test_df, [mx, d2]
    """
    cv = pd.DataFrame(
        train_df[col].value_counts(dropna=False)).reset_index()  # value_counts reset_index后 index就是col  cv[col]为类别样本数
    cv4 = train_df.groupby(col)[tar].mean().reset_index()
    cv4 = cv4.rename(columns={tar: 'rate', col: 'index'})  # group 无法对nan操作，把col重命名为 index 用于合并。
    d1 = set(cv['index'].unique())
    cv = pd.merge(cv, cv4, on='index', how='left')
    if len(cv[cv['index'].isnull()]) != 0:
        cv.loc[cv['index'].isnull(), 'rate'] = train_df.loc[train_df[col].isna(), tar].mean()
    cv = cv[cv[col] > (filter * len(train_df))]  ## 滤掉出现次数较少的类别
    cv['ratec'] = (train_df[tar].sum() - cv['rate'] * cv[col]) / (len(train_df) - cv[col])
    cv['sd'] = zscore * 0.5 / cv[col].map(lambda x: math.sqrt(x))
    cv = cv[(abs(cv['rate'] - m) >= cv['sd']) | (abs(cv['ratec'] - 1 + m) >= cv['sd'])]  # 过滤操作
    d2 = set(cv['index'].unique())
    d = list(d1 - d2)
    if train_df[col].dtype.name == 'category':
        if not 0 in train_df[col].cat.categories:
            train_df[col].cat.add_categories(0, inplace=True)
        else:
            print('###WARNING CAT 0 ALREADY EXISTS IN', col)
    train_df.loc[train_df[col].isin(d), col] = 0  # 将d2范围外的类别置0
    if verbose == 1:
        print('CE encoded', col, '-', len(d2), 'values. Removed', len(d), 'values')
    mx = train_df[col].nunique()
    
    if test_df[col].dtype.name == 'category':
        if not 0 in test_df[col].cat.categories:
            test_df[col].cat.add_categories(0, inplace=True)
        else:
            print('###WARNING CAT 0 ALREADY EXISTS IN', col)
    test_df.loc[~test_df[col].isin(d), col] = 0
    
    return train_df, test_df, [mx, d2]


import numpy as np
import pandas as pd
from tqdm import tqdm


class _Data_Preprocess:
    def __init__(self):
        self.int8_max = np.iinfo(np.int8).max
        self.int8_min = np.iinfo(np.int8).min
        self.int16_max = np.iinfo(np.int16).max
        self.int16_min = np.iinfo(np.int16).min
        self.int32_max = np.iinfo(np.int32).max
        self.int32_min = np.iinfo(np.int32).min
        self.int64_max = np.iinfo(np.int64).max
        self.int64_min = np.iinfo(np.int64).min
        self.float16_max = np.finfo(np.float16).max
        self.float16_min = np.finfo(np.float16).min
        self.float32_max = np.finfo(np.float32).max
        self.float32_min = np.finfo(np.float32).min
        self.float64_max = np.finfo(np.float64).max
        self.float64_min = np.finfo(np.float64).min
    
    '''
    function: _get_type(self,min_val, max_val, types)

       get the correct types that our columns can trans to

    '''
    
    def _get_type(self, min_val, max_val, types):
        if types == 'int':
            if max_val <= self.int8_max and min_val >= self.int8_min:
                return np.int8
            elif max_val <= self.int16_max <= max_val and min_val >= self.int16_min:
                return np.int16
            elif max_val <= self.int32_max and min_val >= self.int32_min:
                return np.int32
            return None
        
        elif types == 'float':
            if max_val <= self.float16_max and min_val >= self.float16_min:
                return np.float16
            if max_val <= self.float32_max and min_val >= self.float32_min:
                return np.float32
            if max_val <= self.float64_max and min_val >= self.float64_min:
                return np.float64
            return None
    
    '''

    function: _memory_process(self,df)
       column data types trans, to save more memory
    '''
    
    def _memory_process(self, df):
        init_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('Original data occupies {} GB memory.'.format(init_memory))
        df_cols = df.columns
        
        for col in tqdm(df_cols):
            try:
                if 'float' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'float')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
                elif 'int' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'int')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
            except:
                print(' Can not do any process for column, {}.'.format(col))
        afterprocess_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('After processing, the data occupies {} GB memory.'.format(afterprocess_memory))
        return df


data_preprocess = _Data_Preprocess()