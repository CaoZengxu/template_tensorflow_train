from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score

import argparse
from nn_modules.model_define import *
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from tensorflow.python import debug as tf_debug
# tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
# from utils_functions import *
import tensorflow as tf
from nn_modules.eager_act import *
from nn_modules.feature_get import *
import sys
from nn_modules.data_prepare import data_prep
from sklearn.model_selection import train_test_split
from nn_modules.graph_act import graph_context_czxmodel
from nn_modules.eager_act import Modeo_eager
import shelve
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
user_data_filename = 'user_data'
exposureLog_filename = 'totalExposureLog.out'
test_sample_filename = 'test_sample.dat'
ad_feature_filename = 'ad_static_feature.out'
ad_op_filename = 'ad_operation.dat'
if sys.platform == 'linux':
    base_path = '~/CommonData/Tencent_2019/testA/'
else:
    base_path = '../Data/'
base_path = '../Data/'


# tf.enable_eager_execution()


def nn_model_cv(X, Y, test_x, param, NN_Model, kfold_seed=2019, kfolds_num=6):
    start_time = time.time()
    val_pre = np.zeros(X.shape[0])
    result = np.zeros(test_x.shape[0])
    kfolds = KFold(n_splits=kfolds_num, shuffle=True, random_state=kfold_seed)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, Y)):
        train_x, train_y = X.loc[trn_idx], Y.loc[trn_idx]
        val_x, val_y = X.loc[val_idx], Y.loc[val_idx]
        tmp_x = []
        tmp_test = []
        tmp_val = []
        for i in param['used_col']:
            tmp_x.append(train_x[i].values)
            tmp_test.append(test_x[i].values)
            tmp_val.append(val_x[i].values)
        tf.reset_default_graph()
        tf.keras.backend.clear_session()
        # tf.clear_session()
        val_y = val_y.values.astype(np.float32)
        train_y = train_y.values.astype(np.float32)
        reg = NN_Model(param)
        reg.train_with_val(train_x.to_dict('list'), train_y, (val_x.to_dict("list"), val_y))
        val_pre[val_idx] = reg.predict(val_x.to_dict("list"), batch_size=8)
        result += reg.predict(test_x.to_dict("list"), test_x.shape[0]) / kfolds.n_splits
    val_score = roc_auc_score(Y, val_pre)
    print('score: ', val_score)
    end_time = time.time()
    print('one cv round:', end_time - start_time)
    return val_pre, result, val_score


def nn_model_val_train(X, Y, x_test, param):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    train_x, val_x, y_train, y_val = dict(X_train), dict(X_val), y_train.astype(np.float32), y_val.astype(np.float32)
    test_x = dict(x_test)
    # X_train = dict
    
    # train_x, train_y, val_x, val_y = [], [], [], []
    # test_x = []
    # columns = list(X.columns)
    # for col in columns:
    #     test_x.append(x_test[col].values)
    #     train_x.append(X_train[col].values)
    #     val_x.append(X_val[col].values)
    #
    # model = Modeo_eager(param)
    model = graph_context_czxmodel(param)
    model.train(train_x, y_train, (val_x, y_val))
    result = model.predict(test_x)
    return result


def norm(data, cols):
    decom_train_dataset = data[cols]
    stats = decom_train_dataset.describe()
    stats = stats.transpose()
    # print(stats['std'])
    
    tmp = (decom_train_dataset - stats['mean']) / (stats['std'] + 0.0001)
    data.drop(cols, axis=1, inplace=True)
    data = data.join(tmp)
    return data


target = 'label'


def train_with_tfrecords(train_path, val_path, test_path, base_path = '../Data/tfrecords/'):
    s = shelve.open(base_path +'midlle_file.db')
    used_fea = s['used_fea']
    max_id_dict = s['max_id_dict']
    len_dict = s['len_dict']
    s.close()
    used_fea.feature_remove('uId')
    # used_fea.feature_remove(['gender_city', 'color_prodName', 'carrier_ct', 'os_prodName'])
    used_fea.feature_remove(['mean_time', 'mean_duration','mean_time'])

    used_fea.print_fea()

    param_dict = {
        'lr': 0.03,
        'epoch': 100,
        'batch_size': 128,
        'embed_size': 64,
        'feature_name': used_fea,
        'numeric_feature_num': len(used_fea.numerical_columns),
        'cate_feature_num': len(used_fea.categorical_columns),
        'size_of_space': max_id_dict,
        'MLP': [256, 64],
        'drop_rate': 0.001,
        'reg_rate': 0.0015,
        'vector_length': 128,
        'pre_train': False,
        'val_batch_size': 345,
        'len_dict': len_dict,
        'appId_list_encoded_length':888,
        'early_stop': 0,
        'weight_file_path': 'tfrecord_negaTwicepos_no_uid_no_adid.h5',
        'val_before_train': False,
        'cv':False
    }
    
    train_list = [base_path + 'train/' + file for file in os.listdir(train_path)]
    val_list = [base_path + 'val/' + file for file in os.listdir(val_path)]
    test_list = [base_path + 'test/' + file for file in os.listdir(test_path)]
    result_list = []
    val_pre_list = []
    val_y_list = []

    model = graph_context_czxmodel(param_dict)
    val_score, val_pre_value, full_val_y = model.train_with_val(train_tfrecord_file_list=train_list, val_tfrecord_file_list=val_list)
    result = model.predict(test_tfrecord_file_list=test_list)
    if param_dict['cv']:
        result_list.append(result)
        val_pre_list.append(val_pre_value)
        val_y_list.append(full_val_y)
        for i in range(4):
            tf.reset_default_graph()
            tf.keras.backend.clear_session()
            model = graph_context_czxmodel(param_dict)
            this_val = train_list[2*i:(2*i)+2]
            this_train = train_list[:2*i] + train_list[(2*i)+2:]
            val_score, val_pre_value, full_val_y = model.train_with_val(train_tfrecord_file_list=this_train+val_list, val_tfrecord_file_list=this_val)
            result = model.predict(test_tfrecord_file_list=test_list)
            result_list.append(result)
            val_pre_list.append(val_pre_value)
            val_y_list.append(full_val_y)
        val_pre = np.concatenate(val_pre_list)
        val_y = np.concatenate(val_y_list)
        cv_score = accuracy_score(np.argmax(val_y, axis=-1), np.argmax(val_pre, axis=-1))
        print("cv score:", cv_score)

        result = np.mean(result_list,axis=0)
    # model.train(train_tfrecord_file_list=train_list+val_list)
    
    test_df = pd.read_csv('../Data/age_test.csv', names=['uId'])
    print(result.shape)
    result_index = np.argmax(result, axis=-1)
    print(result.shape)
    submission = pd.DataFrame()
    submission['id'] = test_df['uId'].astype('int')
    submission['label'] = result_index + 1
    submission['label'] = submission['label'] .astype(np.int)
    submission[['id', 'label']].to_csv('../sub/nn_submit_20190602.csv', index=None)
    # submission = pd.DataFrame()
    # submission['id'] = test_df['uId'].astype('int')
    # submission['probability'] = result
    # submission[['id', 'probability']].to_csv('../sub/prob_nn_submit_20190602.csv', index=None)
    
    print("all done")
    return result


if __name__ == '__main__':
    base_path = '../Data/tfrecord_full_701/'
    train_with_tfrecords(base_path + 'train/', base_path + 'val/', base_path + 'test/',base_path=base_path)

