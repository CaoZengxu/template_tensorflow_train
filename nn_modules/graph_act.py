import tensorflow as tf;
import time;
# from content_ncf.ncf_param import NcfTraParm,NcfCreParam;
import numpy as np;
import time;
from sklearn.metrics import roc_auc_score
from tensorflow.python import keras;
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras import backend as K;
from tensorflow.python.keras.layers import Input, Lambda, Dense, Concatenate, Dropout
from tensorflow.python.keras.initializers import *  # glorot_uniform, zero, Constant
from tensorflow.python.keras.regularizers import l2;
from tensorflow.python.keras.optimizers import Adagrad;
from tensorflow.python.keras import Model;
from tensorflow.python.keras.callbacks import ModelCheckpoint
from .basic_layers_set import HidFeatLayer, K_Value_numrical_layer, Binteraction, CIN
import pandas as pd
from tensorflow.python.keras import backend as K;
import tensorflow.contrib.eager as tfe
from nn_modules.custom_metrics import f1
from nn_modules.model_define import czx_NN_subclass
import math
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, auc, roc_auc_score


# tf.enable_eager_execution()
def score_get(y_true, y_pre):
    y_mean = K.mean(y_true)
    so = K.sum(K.square(y_pre - y_true))
    mo = K.sum(K.square(y_true - y_mean))
    score = 1 - so / mo
    return score


def f1_np(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=-1)
    n_values = 12
    y_pred = np.eye(n_values, dtype=np.float32)[y_pred]
    
    y_pred = np.round(y_pred)
    true_p = np.sum(y_true * y_pred, axis=0)
    true_n = np.sum((1 - y_true) * (1 - y_pred), axis=0)
    
    false_p = np.sum((1 - y_true) * y_pred, axis=0)
    false_n = np.sum(y_true * (1 - y_pred), axis=0)
    
    acc = (true_p + true_n) / (true_p + true_n + false_p + false_n + 0.0000001)
    weighted = np.sum(y_true, axis=0) / (true_p + true_n + false_p + false_n)
    
    precisions = true_p / (true_p + false_p + 0.0000001)
    recall = true_p / (true_p + false_n + 0.0000001)
    
    f1 = 2 * precisions * recall / (precisions + recall + 0.0000001)
    # f1 = np.where(tf.is_nan(f1), np.zeros_like(f1), f1)
    f1 = np.sum(weighted * f1)
    return f1


def make_iterator(tensors, rep=False, batch_size=32):
    # with tf.device('/device:CPU:0'):
    if rep:
        ds = tf.data.Dataset.from_tensor_slices(tensors).batch(batch_size).repeat()
    else:
        ds = tf.data.Dataset.from_tensor_slices(tensors).batch(batch_size)
    
    return ds.make_one_shot_iterator()
    # return tfe.Iterator(ds)


def train_val_map(used_fea):
    def sub_func(example_proto):
        features = {}
        label = {}
        for col in used_fea.categorical_columns:
            if col not in ['appId_list_encoded']:
                features[col] = tf.FixedLenFeature((1), tf.int64, default_value=0)
        for col in used_fea.numerical_columns:
            features[col] = tf.FixedLenFeature((1), tf.float32, default_value=0.1)
        features['appId_list_encoded'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_appId_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_duration_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_times_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_use_date_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['all_activedApp_cate_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_appId_duration_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_appId_times_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_appId_mean_dura_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_appId_full_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_duration_full_list'] = tf.VarLenFeature(dtype=tf.int64)
        features['usage_time_full_list'] = tf.VarLenFeature(dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto, features)
        parsed_features['appId_list_encoded'] = tf.sparse_tensor_to_dense(parsed_features['appId_list_encoded'])
        parsed_features['usage_appId_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_appId_list'])
        parsed_features['usage_duration_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_duration_list'])
        parsed_features['usage_times_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_times_list'])
        parsed_features['usage_use_date_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_use_date_list'])
        parsed_features['all_activedApp_cate_list'] = tf.sparse_tensor_to_dense(parsed_features['all_activedApp_cate_list'])
        
        parsed_features['usage_appId_duration_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_appId_duration_list'])
        parsed_features['usage_appId_times_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_appId_times_list'])
        parsed_features['usage_appId_mean_dura_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_appId_mean_dura_list'])
        
        parsed_features['usage_appId_full_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_appId_full_list'])
        parsed_features['usage_duration_full_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_duration_full_list'])
        parsed_features['usage_time_full_list'] = tf.sparse_tensor_to_dense(parsed_features['usage_time_full_list'])

        label['age_group'] = tf.FixedLenFeature((1), tf.float32, default_value=1)
        parsed_label = tf.parse_single_example(example_proto, label)
        parsed_label['age_group'] = tf.cast(parsed_label['age_group'], tf.int64)
        label = tf.cast(tf.one_hot(parsed_label['age_group'] - 1, 6, 1, 0), tf.float32)
        label = tf.squeeze(label, squeeze_dims=[0])
        return parsed_features, label
    
    return sub_func


class graph_context_czxmodel():
    def __init__(self, param_dict):
        self.param = param_dict
        print(self.param)
        self.lr = param_dict['lr']
        self.epoch = param_dict['epoch']
        self.batch_size = param_dict['batch_size']
        self.val_batch_size = param_dict['val_batch_size']
        # self.batch_num = param_dict['batch_num']
        # self.embed_feature_size_list = param_dict['embed_faeture_list']
        # self.MLP = param_dict['MLP']
        # self.input_dim = param_dict['input_dim']
        self.drop_rate = param_dict['drop_rate']
        self.reg_rate = param_dict['reg_rate']
        # self.vector_length = param_dict['vector_length']
        # self.subclass = param_dict['subclass']
        self.weight_file_path = param_dict['weight_file_path']
        self.pre_train = param_dict['pre_train']
        self.device = '/gpu:0' if tfe.num_gpus() else '/cpu:0'
        with tf.Session().as_default() as sess:
            self.sess = sess
            self.loss = tf.keras.losses.categorical_crossentropy
            # with tf.Graph().as_default():
            self.model = czx_NN_subclass(param_dict)

        self.padded_dict = {}
        for col in self.param['feature_name'].feature_all():
            if col is not 'appId_list_encoded':
                self.padded_dict[col] = [1]
        self.padded_dict['appId_list_encoded'] = self.param['appId_list_encoded_length']
        self.padded_dict['usage_appId_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['usage_duration_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['usage_times_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['usage_use_date_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['all_activedApp_cate_list'] = self.param['size_of_space']['max_cate_len']
        self.padded_dict['usage_appId_duration_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['usage_appId_times_list'] = self.param['size_of_space']['max_usage_len']
        self.padded_dict['usage_appId_mean_dura_list'] = self.param['size_of_space']['max_usage_len']
        
        self.padded_dict['usage_appId_full_list'] = self.param['size_of_space']['max_usage_full_len']
        self.padded_dict['usage_duration_full_list'] = self.param['size_of_space']['max_usage_full_len']
        self.padded_dict['usage_time_full_list'] = self.param['size_of_space']['max_usage_full_len']

    def _report(self, label, start, num_iters, batch_size):
        avg_time = (time.time() - start) / num_iters
        dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
        name = 'graph_%s_%s_batch_%d' % (label, dev, batch_size)
        extras = {'examples_per_sec': batch_size / avg_time}
        # self.report_benchmark(iters=num_iters, wall_time=avg_time, name=name, extras=extras)
    
    def train_with_val(self, x=None, y=None, val=None, train_tfrecord_file_list=None, val_tfrecord_file_list=None):
        best_score = 0
        no_improve_rounds = 0
        best_val_pre_value = None

        with tf.Graph().as_default():
            with self.sess as sess:
                if val_tfrecord_file_list is not None:
                    val_ds = tf.data.TFRecordDataset(val_tfrecord_file_list).map(
                        train_val_map(self.param['feature_name'])).padded_batch(self.val_batch_size,
                                                                                padded_shapes=(self.padded_dict, [6]))
                else:
                    val_ds = tf.data.Dataset.from_tensor_slices(val[0]).batch(self.val_batch_size)
                if train_tfrecord_file_list is not None:
                    train_ds = tf.data.TFRecordDataset(train_tfrecord_file_list).map(
                        train_val_map(self.param['feature_name'])).padded_batch(self.batch_size,
                                                                                padded_shapes=(self.padded_dict, [6])).repeat()
                    train_iterator = train_ds.make_one_shot_iterator()
                else:
                    train_iterator = make_iterator((x, y), rep=True, batch_size=self.batch_size)  # 生成iterator
                batch_x, batch_y = train_iterator.get_next()
                # loss, batch_pre = self.compute_and_loss_batchpre(batch_x, batch_y)
                batch_pre = self.model(batch_x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(batch_y, batch_pre)
                batch_metric = tf.metrics.auc(batch_y, batch_pre)
                optimizer = tf.train.AdagradOptimizer(self.lr)
                train_op = optimizer.minimize(loss)
                init = tf.global_variables_initializer()
                sess.run(tf.local_variables_initializer())
                sess.run(init)
                if self.pre_train:
                    self.model.load_weights(self.weight_file_path, by_name=True)
                    print("load pre train")
                info = {}
                if self.param['val_before_train']:
                    batch_pre_list = []
                    loss_list = []
                    metric_list = []
                    if train_tfrecord_file_list is not None:
                        full_y = []
                        for i in tqdm(range(int(self.param['len_dict']['train_len'] / self.param['batch_size'])),
                                      desc="train:"):
                            try:
                                loss_np, batch_metric_np, batch_pre_np, batch_y_np = sess.run(
                                    [loss, batch_metric, batch_pre, batch_y])
                                batch_pre_list.append(batch_pre_np)
                                loss_list.append(np.mean(loss_np))
                                metric_list.append(batch_metric_np)
                                full_y.append(batch_y_np)
                            except tf.errors.OutOfRangeError:
                                break
                        y = np.concatenate(full_y)
                    else:
                        for _ in tqdm(range(math.ceil(y.shape[0] / self.batch_size)), desc="train:"):  # batch 循环
                            loss_np, batch_metric_np, batch_pre_np = sess.run([loss, batch_metric, batch_pre])
                            batch_pre_list.append(batch_pre_np)
                            loss_list.append(np.mean(loss_np))
                            metric_list.append(batch_metric_np)
                    val_iterator = val_ds.make_one_shot_iterator()
                    score = self.validate(val_iterator=val_iterator, batch_size=self.val_batch_size)
                    best_score = score
                    train_pre = np.concatenate(batch_pre_list)
                    info['epoch_loss'] = np.mean(loss_list)
                    info['acc_val_score'] = score
                    # info['auc_train_score'] = roc_auc_score(y, train_pre)
                    info['acc_train_score'] = accuracy_score(np.argmax(y, axis=-1), np.argmax(train_pre, axis=-1))
                    print("epoch=%d,loss=%.6f,train_acc=%.6f,val_acc=%.6f,time=%s" % (
                    -1, info['epoch_loss'], info['acc_train_score'], info['acc_val_score'], time.asctime()))
                
                for epoch in range(self.epoch):
                    batch_pre_list = []
                    loss_list = []
                    metric_list = []
                    if train_tfrecord_file_list is not None:
                        full_y = []
                        for i in tqdm(range(math.ceil(self.param['len_dict']['train_len'] / self.param['batch_size'])),
                                      desc="train:"):
                            try:
                                loss_np, batch_metric_np, batch_pre_np, train_op_np, batch_y_np = sess.run(
                                    [loss, batch_metric, batch_pre, train_op, batch_y])
                                batch_pre_list.append(batch_pre_np)
                                loss_list.append(np.mean(loss_np))
                                metric_list.append(batch_metric_np)
                                full_y.append(batch_y_np)
                            except tf.errors.OutOfRangeError:
                                break
                        y = np.concatenate(full_y)
                    else:
                        for _ in tqdm(range(math.ceil(y.shape[0] / self.batch_size)), desc="train:"):  # batch 循环
                            loss_np, batch_metric_np, batch_pre_np, train_op_np = sess.run(
                                [loss, batch_metric, batch_pre, train_op])
                            batch_pre_list.append(batch_pre_np)
                            loss_list.append(np.mean(loss_np))
                            metric_list.append(batch_metric_np)
                    val_iterator = val_ds.make_one_shot_iterator()
                    score, val_pre_values, full_val_y= self.validate(val_iterator=val_iterator, batch_size=self.val_batch_size)
                    train_pre = np.concatenate(batch_pre_list)
                    info['epoch_loss'] = np.mean(loss_list)
                    info['acc_val_score'] = score
                    info['acc_train_score'] = accuracy_score(np.argmax(y, axis=-1), np.argmax(train_pre, axis=-1))
                    
                    print("epoch=%d,loss=%.6f,train_acc=%.6f,val_acc=%.6f,time=%s" % (
                        epoch, info['epoch_loss'], info['acc_train_score'], info['acc_val_score'], time.asctime()))
                    if score > best_score:
                        print('save')
                        self.model.save_weights(self.weight_file_path)
                        no_improve_rounds = 0
                        best_score = score
                        best_val_pre_value = val_pre_values
                    else:
                        no_improve_rounds += 1
                    if no_improve_rounds > self.param['early_stop']:
                        print("early stop at ", epoch)
                        break
        return best_score, best_val_pre_value,full_val_y
    
    def predict(self, x=None, test_tfrecord_file_list=None, batch_size=512):
        
        with tf.Session() as sess:
            if test_tfrecord_file_list is not None:
                test_iterator = tf.data.TFRecordDataset(test_tfrecord_file_list).map(
                    train_val_map(self.param['feature_name'])).padded_batch(batch_size, padded_shapes=(
                self.padded_dict, [6])).make_one_shot_iterator()
                batch_x, batch_y = test_iterator.get_next()
            
            else:
                test_iterator = make_iterator(x, batch_size=batch_size)
                batch_x = test_iterator.get_next()
            full_pre = []
            
            pre = self.model(batch_x, training=False)
            
            init = tf.global_variables_initializer()
            sess.run(tf.local_variables_initializer())
            sess.run(init)
            self.model.load_weights(self.weight_file_path, by_name=True)
            
            # with tf.Graph().as_default():
            if test_tfrecord_file_list is not None:
                for i in tqdm(range(math.ceil(self.param['len_dict']['test_len'] / batch_size)), desc="prediction:"):
                    try:
                        pre_np = sess.run(pre)
                        full_pre.append(pre_np)
                    except tf.errors.OutOfRangeError:
                        break
            else:
                for i in tqdm(range(math.ceil(list(x.values())[0].shape[0] / batch_size)), desc="prediction:"):
                    try:
                        pre_np = sess.run(pre)
                        full_pre.append(pre_np)
                    except tf.errors.OutOfRangeError:
                        break
            full_pre = np.concatenate(full_pre)
        return full_pre
    
    def validate(self, val_iterator, y=None, batch_size=128):
        # test_iterator = make_iterator(x, batch_size=batch_size)
        full_pre = []
        
        if y is None:
            batch_x, batch_y = val_iterator.get_next()
            pre = self.model(batch_x, training=False)
            full_y = []
            for i in tqdm(range(math.ceil(self.param['len_dict']['val_len'] / batch_size)), desc="validation:"):
                try:
                    pre_np, batch_y_np = self.sess.run([pre, batch_y])
                    full_pre.append(pre_np)
                    full_y.append(batch_y_np)
                except tf.errors.OutOfRangeError:
                    break
            y = np.concatenate(full_y)
        else:
            batch_x = val_iterator.get_next()
            pre = self.model(batch_x, training=False)
            for i in tqdm(range(math.ceil(y.shape[0] / batch_size)), desc="validation:"):
                try:
                    pre_np = self.sess.run(pre)
                    full_pre.append(pre_np)
                except tf.errors.OutOfRangeError:
                    break
        full_pre = np.concatenate(full_pre)
        print("print shape: ", y.shape[0], full_pre.shape[0])
        # score = roc_auc_score(y, full_pre)
        score = accuracy_score(np.argmax(y, axis=-1), np.argmax(full_pre, axis=-1))
        # full_pre = np.argmax(full_pre, axis=1)
        # sk_f1 = f1_score(np.argmax(y,axis=1), full_pre, average='weighted')
        return score, full_pre, y
    
    def train(self, x=None, y=None, train_tfrecord_file_list=None):
        
        with tf.Graph().as_default():
            with self.sess as sess:
                
                if train_tfrecord_file_list is not None:
                    train_ds = tf.data.TFRecordDataset(train_tfrecord_file_list).map(
                        train_val_map(self.param['feature_name'])).padded_batch(self.batch_size,
                                                                                padded_shapes=(self.padded_dict, [6])).repeat()
                    train_iterator = train_ds.make_one_shot_iterator()
                else:
                    train_iterator = make_iterator((x, y), rep=True, batch_size=self.batch_size)  # 生成iterator
                batch_x, batch_y = train_iterator.get_next()
                batch_pre = self.model(batch_x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(batch_y, batch_pre)
                batch_metric = tf.metrics.auc(batch_y, batch_pre)
                optimizer = tf.train.AdagradOptimizer(self.lr)
                train_op = optimizer.minimize(loss)
                init = tf.global_variables_initializer()
                sess.run(tf.local_variables_initializer())
                sess.run(init)
                if self.pre_train:
                    self.model.load_weights(self.weight_file_path, by_name=True)
                    print("load pre train")
                info = {}
                if self.param['val_before_train']:
                    batch_pre_list = []
                    loss_list = []
                    metric_list = []
                    if train_tfrecord_file_list is not None:
                        full_y = []
                        for i in tqdm(range(int(self.param['len_dict']['train_len']+self.param['len_dict']['val_len'] / self.param['batch_size'])),
                                      desc="train:"):
                            try:
                                loss_np, batch_metric_np, batch_pre_np, batch_y_np = sess.run(
                                    [loss, batch_metric, batch_pre, batch_y])
                                batch_pre_list.append(batch_pre_np)
                                loss_list.append(np.mean(loss_np))
                                metric_list.append(batch_metric_np)
                                full_y.append(batch_y_np)
                            except tf.errors.OutOfRangeError:
                                break
                        y = np.concatenate(full_y)
                    else:
                        for _ in tqdm(range(math.ceil(y.shape[0] / self.batch_size)), desc="train:"):  # batch 循环
                            loss_np, batch_metric_np, batch_pre_np = sess.run([loss, batch_metric, batch_pre])
                            batch_pre_list.append(batch_pre_np)
                            loss_list.append(np.mean(loss_np))
                            metric_list.append(batch_metric_np)
                    train_pre = np.concatenate(batch_pre_list)
                    info['epoch_loss'] = np.mean(loss_list)
                    info['acc_train_score'] = accuracy_score(np.argmax(y, axis=-1), np.argmax(train_pre, axis=-1))
                    print("epoch=%d,loss=%.6f,train_acc=%.6f,val_acc=%.6f,time=%s" % (-1, info['epoch_loss'], info['acc_train_score'], time.asctime()))
                
                for epoch in range(self.epoch):
                    batch_pre_list = []
                    loss_list = []
                    metric_list = []
                    if train_tfrecord_file_list is not None:
                        full_y = []
                        for i in tqdm(range(math.ceil(self.param['len_dict']['train_len'] / self.param['batch_size'])),
                                      desc="train:"):
                            try:
                                loss_np, batch_metric_np, batch_pre_np, train_op_np, batch_y_np = sess.run(
                                    [loss, batch_metric, batch_pre, train_op, batch_y])
                                batch_pre_list.append(batch_pre_np)
                                loss_list.append(np.mean(loss_np))
                                metric_list.append(batch_metric_np)
                                full_y.append(batch_y_np)
                            except tf.errors.OutOfRangeError:
                                break
                        y = np.concatenate(full_y)
                    else:
                        for _ in tqdm(range(math.ceil(y.shape[0] / self.batch_size)), desc="train:"):  # batch 循环
                            loss_np, batch_metric_np, batch_pre_np, train_op_np = sess.run(
                                [loss, batch_metric, batch_pre, train_op])
                            batch_pre_list.append(batch_pre_np)
                            loss_list.append(np.mean(loss_np))
                            metric_list.append(batch_metric_np)
                    train_pre = np.concatenate(batch_pre_list)
                    info['epoch_loss'] = np.mean(loss_list)
                    info['acc_train_score'] = accuracy_score(np.argmax(y, axis=-1), np.argmax(train_pre, axis=-1))
                    
                    print("epoch=%d,loss=%.6f,train_acc=%.6f,time=%s" % (
                    epoch, info['epoch_loss'], info['acc_train_score'], time.asctime()))

                    self.model.save_weights(self.weight_file_path)
                    print('saved')
                    
    def compute_and_loss_batchpre(self, inputs, labels):
        """
        :param inputs: 输入
        :param labels: 标签
        :return: 梯度，损失值，对batch_x的预测值
        """
        batch_pre = self.model(inputs, training=True)
        loss = self.loss(labels, batch_pre)
        
        return loss, batch_pre
