import pandas as pd
from nn_modules.feature_get import *
import tensorflow as tf
import os
from nn_modules.data_prepare import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing
import shelve

from sklearn.utils import shuffle

folder_name = 'tfrecord_full_706'


def norm(data, cols):
    decom_train_dataset = data[cols]
    stats = decom_train_dataset.describe()
    stats = stats.transpose()
    tmp = (decom_train_dataset - stats['mean']) / (stats['std'] + 0.0001)
    data.drop(cols, axis=1, inplace=True)
    data = data.join(tmp)
    return data


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_tf_example(data, used_fea, resize=None):
    fea_dict = {}
    for col in used_fea.categorical_columns:
        if col not in ['appId_list_encoded']:
            fea_dict[col] = int64_feature(int(data[col]))
    for col in used_fea.numerical_columns + ['age_group']:
        fea_dict[col] = float_feature(data[col])
    fea_dict['appId_list_encoded'] = tf.train.Feature(int64_list=tf.train.Int64List(value=data['appId_list_encoded']))
    usage_appId_list = list(map(lambda x: int(x), data['usage_appId_list'].split('#')))
    usage_duration_list = list(map(lambda x: int(float(x)), data['usage_duration_list'].split('#')))
    usage_times_list = list(map(lambda x: int(float(x)), data['usage_times_list'].split('#')))
    activedapp_cate_list = list(map(lambda x: int(x), data['all_activedApp_cate_list'].split('#')))
    usage_appId_duration_list = list(map(lambda x: int(x), data['usage_appId_duration_list'].split('#')))
    usage_appId_times_list = list(map(lambda x: int(x), data['usage_appId_times_list'].split('#')))
    usage_appId_mean_dura_list = list(map(lambda x: int(x), data['usage_appId_mean_dura_list'].split('#')))
    
    usage_appId_full_list = list(map(lambda x: int(x), data['usage_appId_full_list'].split('#')))
    usage_duration_full_list = list(map(lambda x: int(x), data['usage_duration_full_list'].split('#')))
    usage_time_full_list = list(map(lambda x: int(x), data['usage_time_full_list'].split('#')))

    fea_dict['all_activedApp_cate_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=activedapp_cate_list))
    fea_dict['usage_appId_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_appId_list))
    fea_dict['usage_duration_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_duration_list))
    fea_dict['usage_times_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_times_list))
    fea_dict['usage_appId_duration_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_appId_duration_list))
    fea_dict['usage_appId_times_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_appId_times_list))
    fea_dict['usage_appId_mean_dura_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_appId_mean_dura_list))
    
    fea_dict['usage_appId_full_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_appId_full_list))
    fea_dict['usage_duration_full_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_duration_full_list))
    fea_dict['usage_time_full_list'] = tf.train.Feature(int64_list=tf.train.Int64List(value=usage_time_full_list))

    tf_example = tf.train.Example(features=tf.train.Features(feature=fea_dict))
    
    return tf_example


def czx_write_func(i, train_slice, used_fea):
    writer = tf.python_io.TFRecordWriter("../Data/" + folder_name + "/train/train_" + str(i) + "_record.record")
    for index, row in tqdm(train_slice.iterrows(), desc='train_' + str(i)):
        exemple = create_tf_example(row, used_fea)
        writer.write(exemple.SerializeToString())
    writer.close()


target = 'age_group'
if __name__ == "__main__":
    print('06291926')
    print("folder name####################################:",folder_name)
    data_all, used_fea, max_id_dict = data_prep(old=False)
    print('base readed')
    # usage_df, use_feature_sec, sequence_fea, max_id_dict_sec = usage_fea_get(base_path='/data/Huawei_people_properties/', old=True)
    base_path = '/data/Huawei_people_properties/'
    if sys.platform =='win32':
        base_path = '../Data/'
    usage_df, use_feature_sec, sequence_fea, max_id_dict_sec = usage_fea_get_onetime(base_path=base_path , old=False)
    usage_df.head(500).to_csv("usage_head_626.csv", index=None)
    print('usage finish')
    print('usage info: ', usage_df.info())
    usage_cate = use_feature_sec.categorical_columns
    used_fea.feature_add(usage_cate, 'cate')
    usage_num = use_feature_sec.numerical_columns
    used_fea.feature_add(usage_num, 'num')

    for one_key in list(max_id_dict_sec.keys()):
        max_id_dict[one_key] = max_id_dict_sec[one_key]
    
    data_all = pd.merge(data_all, usage_df, on='uId', how='left')
    
    for col in ['usage_appId_list', 'usage_duration_list', 'usage_times_list','usage_appId_duration_list','usage_appId_times_list','usage_appId_mean_dura_list','usage_appId_full_list','usage_duration_full_list','usage_time_full_list']:
        data_all.loc[data_all[col].isnull(), col] = '0'
    for col in ['max_duration', 'max_duration_app', 'min_duration', 'min_duration_app', 'max_time', 'max_time_app', 'min_time',
                'min_time_app', 'usage_len','mean_duration', 'mean_time','usage_len_full']:
        data_all[col] = data_all[col].fillna(0)
    nul_index = data_all['all_activedApp_cate_list'].isnull()
    data_all.loc[nul_index, 'all_activedApp_cate_list'] = '0'
    data_all['all_activedApp_cate_list_len'] = data_all['all_activedApp_cate_list'].apply(lambda x:len(x.split('#')))
    max_id_dict['max_cate_len'] = data_all['all_activedApp_cate_list_len'].max()

    print('all_activedApp_cate_list_len')
    data_all, max_id = factor_data(data_all, 'all_activedApp_cate_list_len')
    max_id_dict['all_activedApp_cate_list_len'] = max_id
    data_all['all_activedApp_cate_list_len'] = data_all['all_activedApp_cate_list_len'].astype(np.int)
    
    data_all.head(50000).to_csv("data_all_head_626.csv", index=None)
    print("data is null", data_all.isnull().sum())
    print('1 age_test shape: ', data_all[data_all['data_flag'] == 0].shape)
    used_fea.print_fea()
    data_all[target] = data_all[target].fillna(-1).astype('int')
    s = shelve.open('../Data/' + folder_name + '/midlle_file.db')
    s['max_id_dict'] = max_id_dict
    s['used_fea'] = used_fea
    print("finish feature")
    if len(used_fea.numerical_columns):
        for col in used_fea.numerical_columns:
            fill_value = data_all[col].mean()
            data_all[col].fillna(fill_value, inplace=True)
        data_all = norm(data_all, used_fea.numerical_columns)
    print('2 age_test shape: ', data_all[data_all['data_flag'] == 0].shape)
    
    for col in used_fea.numerical_columns:
        data_all[col] = data_all[col].astype(np.float32)
    print('3 age_test shape: ', data_all[data_all['data_flag'] == 0].shape)
    
    train_df = data_all[data_all['data_flag'] == 1]
    test_df = data_all[data_all['data_flag'] == 0]
    del data_all
    print("origin", train_df[target].value_counts())
    # train_df = train_df.sample(frac = 0.1)
    train_df = shuffle(train_df)
    train_df.reset_index(inplace=True)
    print("finish sample")
    print(train_df[target].value_counts())
    train, val = train_test_split(train_df, test_size=0.2, random_state=0)
    # train = train_df
    len_dict = {}
    len_dict['train_len'] = train.shape[0]
    len_dict['val_len'] = val.shape[0]
    len_dict['test_len'] = test_df.shape[0]
    print(len_dict)
    s['len_dict'] = len_dict
    del train_df
    train_len = train.shape[0]
    group_num = 8
    onelen = int(train_len / group_num)
    pool = multiprocessing.Pool(processes=group_num)  # 创建groupnum个进程
    return_list = []
    for i in range(group_num):
        train_slice = train[i * onelen:(i + 1) * onelen]
        return_list.append(pool.apply_async(czx_write_func, (i, train_slice, used_fea,)))
    pool.close()
    pool.join()
    czx_write_func(group_num, train[(i + 1) * onelen:], used_fea)
    # del train
    if val.shape[0]>0:
        val_writer = tf.python_io.TFRecordWriter("../Data/" + folder_name + "/val/val_record.record")
        print("val len: ", val.shape[0])
        for index, row in tqdm(val.iterrows(), desc="val"):
            exemple = create_tf_example(row, used_fea)
            val_writer.write(exemple.SerializeToString())
        val_writer.close()
    
    print("val saved")
    print("test len: ", test_df.shape[0])
    test_writer = tf.python_io.TFRecordWriter("../Data/" + folder_name + "/test/test_record.record")
    for index, row in tqdm(test_df.iterrows(), desc='test'):
        exemple = create_tf_example(row, used_fea)
        test_writer.write(exemple.SerializeToString())
    test_writer.close()
    
    print("test saved")
    s.close()
