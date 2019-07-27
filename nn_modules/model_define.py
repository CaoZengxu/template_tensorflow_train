import tensorflow as tf;
import time;
# from content_ncf.ncf_param import NcfTraParm,NcfCreParam;
import numpy as np;
import sys
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
from .basic_layers_set import *
import pandas as pd
from nn_modules.Transformer_modules import *


# tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    
    def on_train_begin(self, logs={}):
        return
    
    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return
    
    def on_batch_begin(self, batch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return


class czx_NN_subclass(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(czx_NN_subclass, self).__init__(name='my_model')
        self.param = param
        self.max_id_dict = param['size_of_space']
        self.feature_name = param['feature_name']
        fea_embed_out_list = []
        embed_size = param['embed_size']
        size_of_space = param['size_of_space']
        fea_num = param['cate_feature_num']  # + param['numeric_feature_num']
        fea_num -= 1
        # fea_num += 887
        MLP = param['MLP']
        reg_rate = param['reg_rate']
        drop_rate = param['drop_rate']
        bias_list = []
        input_dict = {}
        self.full_sequence_cut_len = 1000
        self.sorted_usage_cut_len = 128
        inter_feature_num = len(self.feature_name.categorical_columns) + 3 + 9 +1

        self.embeding_layer_dict = {}
        self.bias_layer_dict = {}
        func_features = ['AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes', 'EFuncTimes', 'FFuncTimes', 'GFuncTimes']
        self.app_cate_fea = ['商务', '实用工具', '影音娱乐', '动作射击', '新闻阅读', '教育', '便捷生活', '休闲益智', '运动健康', '经营策略', '金融理财', '体育竞速', '社交通讯',
                             '购物比价', '角色扮演', '出行导航', '棋牌桌游', '拍摄美化', '儿童', '旅游住宿', '汽车', '美食', '主题个性', '网络游戏', '学习办公', '模拟游戏',
                             '休闲游戏', '棋牌天地', '体育射击', '策略游戏', '角色游戏', '医疗健康', '休闲娱乐', '动作冒险', '主题铃声', '图书阅读', '电子书籍', '益智棋牌',
                             '合作壁纸*', '表盘个性']
        for col in self.feature_name.categorical_columns:
            if col in ['appId_list_encoded']:
                self.embeding_layer_dict[col] = HidFeatLayer(int(self.max_id_dict[col]) + 1, embed_size, zero_lim=True)
                continue
            self.embeding_layer_dict[col] = HidFeatLayer(int(self.max_id_dict[col]) + 1, embed_size)
            self.bias_layer_dict[col] = HidFeatLayer(int(self.max_id_dict[col]) + 1, 6)
        # for col, max_id_key in zip(
        #         ['usage_appId_list', 'usage_duration_list', 'usage_times_list'],
        #         ['appId_encode_list', 'duration_encode_list', 'times_encode_list']):
        #     self.embeding_layer_dict[col] = HidFeatLayer(int(self.max_id_dict[max_id_key]) + 1, embed_size, zero_lim=True)
        #     print("size compare :" + col, self.max_id_dict[max_id_key])
        for col, max_id_key in zip(
                ['usage_appId_full_list', 'usage_duration_full_list', 'usage_times_full_list', 'appcate_list','appId_list_lstm_encoded','usage_appId_duration_list','usage_appId_times_list','usage_appId_mean_dura_list'],
                ['appId_encode_list', 'duration_encode_list', 'times_encode_list', 'all_activedApp_cate_list','appId_encode_list','appId_encode_list','appId_encode_list','appId_encode_list']):
            self.embeding_layer_dict[col] = HidFeatLayer(int(self.max_id_dict[max_id_key]) + 1, embed_size, zero_lim=True)
            print("size compare :" + col, self.max_id_dict[max_id_key])
            
        # ==================================================== A-F功能 ======================================================
        self.func_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.func_trans.add(layer)
            
        # ================================== 激活app 所属 cate ===============================================================
        self.appcate_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.appcate_trans.add(layer)
        self.appcate_conv = Conv_Sequence(filter_size=self.param['size_of_space']['max_cate_len'], embedding_size=1,
                                          num_filters=1, sequence_length=self.param['size_of_space']['max_cate_len'],
                                          pool=False)
        
        # =============================================== cate ont fc ================================================
        self.appcate_onehot_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.appcate_onehot_trans.add(layer)
        
        self.denoise_encoder = CusAutoEncoder([self.max_id_dict['all_activedApp_cate_list'] , 32, 128, self.max_id_dict['all_activedApp_cate_list']], encode_index=1, name='dae')
        self.denoise_trans = tf.keras.layers.Dense(embed_size, activation=tf.nn.sigmoid, use_bias=True,
                              kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
        
        # ==================================================  激活app ========================================================
        self.appembed_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.appembed_trans.add(layer)
        self.active_conv = Conv_Sequence(filter_size=888, embedding_size=1, num_filters=1, sequence_length=888, pool=False)
        
        # ============================================= usage 特征 ============================================================
        self.usage_appId_duration_trans = tf.keras.models.Sequential()  # usage_appId
        for unit in [128, embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appId_duration_trans.add(layer)
        self.usage_appId_duration_conv = Conv_Sequence(filter_size=self.sorted_usage_cut_len, embedding_size=1, num_filters=1, sequence_length=888, pool=False)

        self.usage_appid_times_trans = tf.keras.models.Sequential()  # usage_duration
        for unit in [128, embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_times_trans.add(layer)
        self.usage_appid_times_conv = Conv_Sequence(filter_size=self.sorted_usage_cut_len, embedding_size=1, num_filters = 1, sequence_length=888, pool=False)

        self.usage_appid_mean_duration_trans = tf.keras.models.Sequential()  # usage_times
        for unit in [128, embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_mean_duration_trans.add(layer)
        self.usage_appid_mean_duration_conv = Conv_Sequence(filter_size=self.sorted_usage_cut_len, embedding_size=1, num_filters=1, sequence_length=888,pool=False)
        
        # ===================================== usage full 特征 ===================================
        self.usage_full_appId = tf.keras.models.Sequential()  # usage_appId
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_full_appId.add(layer)
        self.usage_appId_full_conv = Conv_Sequence(filter_size=self.full_sequence_cut_len, embedding_size=1,
                                                   num_filters=1, sequence_length=888, pool=False)
        self.usage_appId_full_attention = Attention(embed_size,self.full_sequence_cut_len)
        
        self.usage_full_duration = tf.keras.models.Sequential()  # usage_duration
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_full_duration.add(layer)
        self.usage_duration_full_conv = Conv_Sequence(filter_size=self.full_sequence_cut_len,
                                                      embedding_size=1,
                                                      num_filters=1, sequence_length=888,
                                                      pool=False)
        self.usage_duration_full_attention = Attention(embed_size,self.full_sequence_cut_len)
        
        self.usage_full_times = tf.keras.models.Sequential()  # usage_times
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_full_times.add(layer)
        self.usage_times_full_conv = Conv_Sequence(filter_size=self.full_sequence_cut_len, embedding_size=1,
                                                   num_filters=1, sequence_length=888,
                                                   pool=False)
        self.usage_times_full_attention = Attention(embed_size,self.full_sequence_cut_len)
        
        # ------------------------usage duration transfomer encoder part ------------------------
        self.usage_appid_duration_encoder = Encoder(n_layers=2, d_model=128, n_heads=8, ddf=256,
                                         input_vocab_size=int(self.max_id_dict['appId_encode_list']) + 1, max_seq_len=256)
        self.usage_appid_duration_encoder_mlp = tf.keras.models.Sequential()
        for unit in [128, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_duration_encoder_mlp.add(layer)
        self.usage_appid_duration_encoder_feature_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_duration_encoder_feature_trans.add(layer)
            
        # ===================================== usage duration LSTM part ==========================================
        self.usage_appid_duration_lstm = tf.keras.models.Sequential()
        if sys.platform == 'win32':
            self.usage_appid_duration_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), merge_mode='ave'))
        else:
            self.usage_appid_duration_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True), merge_mode='ave'))

        self.usage_appid_duration_lstm.add(Attention(embed_size, sequenceLength=self.param['size_of_space']['max_usage_len']))
        # self.usage_appid_duration_lstm.add(tf.keras.layers.Dropout(0.5))
        # self.usage_appid_duration_lstm.add(tf.keras.layers.BatchNormalization())
        # self.usage_appid_duration_lstm.add(tf.keras.layers.TimeDistributed(Dense(embed_size, activation=tf.nn.sigmoid)))
        self.usage_appid_duration_lstm_trans = tf.keras.models.Sequential()
        for unit in [embed_size]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_duration_lstm_trans.add(layer)
        self.usage_appid_duration_lstm_mlp = tf.keras.models.Sequential()
        for unit in [embed_size, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.usage_appid_duration_lstm_mlp.add(layer)
        self.usage_appid_duration_lstm_conv = Conv_Sequence(filter_size=self.param['size_of_space']['max_usage_len'], embedding_size=1, num_filters=1, sequence_length=888, pool=False)

        # ============================================ 公共部分 ====================================================
        self.mine_reshape = tf.keras.layers.Reshape([inter_feature_num, embed_size])
        self.bi = Binteraction()
        self.cin = CIN(vector_length=embed_size,
                       feature_num=inter_feature_num,
                       activition=tf.nn.sigmoid, cross_layer_sizes=[128, 128, 128])
        self.fm = FM_socond_part()
        
        self.cin_fc = tf.keras.models.Sequential()
        for unit in [128, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.cin_fc.add(layer)
        
        self.bi_fc = tf.keras.models.Sequential()
        for unit in [128, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.bi_fc.add(layer)
        
        self.fm_fc = tf.keras.models.Sequential()
        for unit in [128, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.fm_fc.add(layer)
        
        self.mlp = tf.keras.models.Sequential()
        for unit in [256, 128, 6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019), kernel_regularizer=l2(0.01))
            self.mlp.add(layer)
        
        self.last_out_layer = tf.keras.models.Sequential()
        for unit in [6]:
            layer = tf.keras.layers.Dense(unit, activation=tf.nn.sigmoid, use_bias=True,
                                          kernel_initializer=glorot_normal(seed=2019),
                                          kernel_regularizer=l2(0.01))
            self.last_out_layer.add(layer)
        
        self.conv_1 = Conv_Sequence(filter_size=6, embedding_size=1, num_filters=1, sequence_length=888, pool=False)
        self.conv_2 = Conv_Sequence(filter_size=2, embedding_size=1, num_filters=1, sequence_length=888, pool=False)


    def call(self, inputs, training=None, mask=None):
        func_features = ['AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes', 'EFuncTimes', 'FFuncTimes', 'GFuncTimes']
        
        cate_fea_embeding = {}
        cate_fea_bias = []
        
        for col in self.feature_name.categorical_columns:
            if col in ['appId_list_encoded']:
                continue
            cate_fea_bias.append(tf.squeeze(self.bias_layer_dict[col](tf.cast(inputs[col], dtype=tf.int32)), axis=[1, -1]))
            tmp_input = tf.squeeze(self.embeding_layer_dict[col](tf.cast(inputs[col], dtype=tf.int32)), axis=[1, -1])
            cate_fea_embeding[col] = tmp_input
        
        # ========================================= func feature trans part ====================================================
        func_features_embec_list = []
        for col in func_features:
            func_features_embec_list.append(
                tf.squeeze(self.embeding_layer_dict[col](tf.cast(inputs[col], dtype=tf.int32)), axis=[1, -1]))
        concat_func_fea_all = tf.concat(func_features_embec_list, axis=-1)
        concat_func_fea_transed = self.func_trans(concat_func_fea_all)
        # ========================================= app cate feature trans part ================================================
        appcate_list_embed_expanded = self.embeding_layer_dict['appcate_list'](
            tf.cast(inputs['all_activedApp_cate_list'], dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        appId_cate_list_embed = tf.squeeze(appcate_list_embed_expanded, axis=-1)
        appId_cate_list_embed_flatten = tf.layers.flatten(appId_cate_list_embed)
        appId_cate_list_embed = self.appcate_trans(appId_cate_list_embed_flatten)
        appId_cate_list_conved = tf.squeeze(self.appcate_conv(appcate_list_embed_expanded), axis=[1, -1])
        appId_cate_mean = tf.reduce_mean(appcate_list_embed_expanded, axis=1)
        
        # ======================================================= app cate one hot =============================================
        cate_one_hot = tf.one_hot(tf.cast(inputs['all_activedApp_cate_list'], dtype=tf.int32),
                                  self.max_id_dict['all_activedApp_cate_list'], 1, 0)
        cate_one_hot_sum = tf.reduce_sum(cate_one_hot, axis=1)
        zeros_like_cate = tf.zeros_like(cate_one_hot_sum)
        cate_one_hot_ = tf.where(tf.equal(cate_one_hot_sum, 0), zeros_like_cate, zeros_like_cate + 1)
        cate_one_hot_fea = self.appcate_onehot_trans(tf.cast(cate_one_hot_, tf.float32))
        
        # ============================================= active app list ========================================================
        appId_embed_expanded = self.embeding_layer_dict['appId_list_encoded'](
            tf.cast(inputs['appId_list_encoded'], dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        appId_list_embed = tf.squeeze(appId_embed_expanded, axis=-1)
        appId_list_embed_flatten = tf.layers.flatten(appId_list_embed)
        appId_embed = self.appembed_trans(appId_list_embed_flatten)
        active_conved = tf.squeeze(self.active_conv(appId_embed_expanded), axis=[1, -1])
        appId_mean = tf.reduce_mean(appId_list_embed, axis=1)
        # ================================================ usage_appId_list ====================================================
        usage_appId_duration_list = tf.slice(inputs['usage_appId_duration_list'], [0, 0], [-1, self.sorted_usage_cut_len])
        # usage_appId_list = inputs['usage_appId_list']
        usage_appId_duration_embed_expanded = self.embeding_layer_dict['usage_appId_duration_list'](tf.cast(usage_appId_duration_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_appId_duration_list_embed = tf.squeeze(usage_appId_duration_embed_expanded, axis=-1)
        usage_appId_duration_list_embed_flatten = tf.layers.flatten(usage_appId_duration_list_embed)
        usage_appId_duration_embed = self.usage_appId_duration_trans(usage_appId_duration_list_embed_flatten)
        usage_appId_duration_conved = tf.squeeze(self.usage_appId_duration_conv(usage_appId_duration_embed_expanded), axis=[1, -1])
        usage_appId_duration_mean = tf.reduce_mean(appId_list_embed, axis=1)
        # ============================================= usage_duration_list ====================================================
        usage_appid_times_list = tf.slice(inputs['usage_appId_times_list'], [0, 0], [-1, self.sorted_usage_cut_len])
        # usage_duration_list = inputs['usage_duration_list']
        usage_appid_times_embed_expanded = self.embeding_layer_dict['usage_appId_times_list'](
            tf.cast(usage_appid_times_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_appid_times_list_embed = tf.squeeze(usage_appid_times_embed_expanded, axis=-1)
        usage_appid_times_list_embed_flatten = tf.layers.flatten(usage_appid_times_list_embed)
        usage_appid_times_embed = self.usage_appid_times_trans(usage_appid_times_list_embed_flatten)
        usage_appid_times_conved = tf.squeeze(self.usage_appid_times_conv(usage_appid_times_embed_expanded), axis=[1, -1])
        usage_appid_times_mean = tf.reduce_mean(appId_list_embed, axis=1)
        # ============================================== usage_times_list ======================================================
        usage_appid_mean_duration_list = tf.slice(inputs['usage_appId_mean_dura_list'], [0, 0], [-1, self.sorted_usage_cut_len])
        usage_appid_mean_duration_embed_expanded = self.embeding_layer_dict['usage_appId_mean_dura_list'](tf.cast(usage_appid_mean_duration_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_appid_mean_duration_list_embed = tf.squeeze(usage_appid_mean_duration_embed_expanded, axis=-1)
        usage_appid_mean_duration_list_embed_flatten = tf.layers.flatten(usage_appid_mean_duration_list_embed)
        usage_appid_mean_duration_embed = self.usage_appid_mean_duration_trans(usage_appid_mean_duration_list_embed_flatten)
        usage_appid_mean_duration_conved = tf.squeeze(self.usage_appid_mean_duration_conv(usage_appid_mean_duration_embed_expanded), axis=[1, -1])
        usage_appid_mean_duration_mean = tf.reduce_mean(appId_list_embed, axis=1)
        """
        """
        # =========================================== usage_appId_full_list ====================================================
        usage_appId_full_list = tf.slice(inputs['usage_appId_full_list'], [0, 0], [-1, self.full_sequence_cut_len])
        usage_appId_full_embed_expanded = self.embeding_layer_dict['usage_appId_full_list'](
            tf.cast(usage_appId_full_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_appId_full_list_embed = tf.squeeze(usage_appId_full_embed_expanded, axis=-1)
        usage_appId_full_list_embed_flatten = tf.layers.flatten(usage_appId_full_list_embed)
        usage_appId_full_embed = self.usage_full_appId(usage_appId_full_list_embed_flatten)
        usage_appId_full_conved = tf.squeeze(self.usage_appId_full_conv(usage_appId_full_embed_expanded), axis=[1, -1])
        usage_appId_full_mean = tf.reduce_mean(usage_appId_full_list_embed, axis=1)
        # ============================================= usage_duration_full_list ===============================================
        usage_duration_full_list = inputs['usage_duration_full_list']
        usage_duration_full_list = tf.slice(usage_duration_full_list, [0, 0], [-1, self.full_sequence_cut_len])
        usage_duration_full_embed_expanded = self.embeding_layer_dict['usage_duration_full_list'](
            tf.cast(usage_duration_full_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_duration_full_list_embed = tf.squeeze(usage_duration_full_embed_expanded, axis=-1)
        usage_duration_full_list_embed_flatten = tf.layers.flatten(usage_duration_full_list_embed)
        usage_duration_full_embed = self.usage_full_duration(usage_duration_full_list_embed_flatten)
        usage_duration_full_conved = tf.squeeze(self.usage_duration_full_conv(usage_duration_full_embed_expanded), axis=[1, -1])
        usage_duration_full_mean = tf.reduce_mean(usage_duration_full_list_embed, axis=1)
        # ============================================== usage_times_full_list =================================================
        usage_times_full_list = inputs['usage_time_full_list']
        usage_times_full_list = tf.slice(usage_times_full_list, [0, 0], [-1, self.full_sequence_cut_len])
        usage_times_full_embed_expanded = self.embeding_layer_dict['usage_times_full_list'](tf.cast(usage_times_full_list, dtype=tf.int32))
        # active_mask = tf.expand_dims(tf.squeeze(tf.sequence_mask(inputs['appId_list_len'], 888), axis=1), axis=-1)
        # active_mask = tf.tile(active_mask, multiples=[1, 1, 128])
        # active_mask = tf.expand_dims(active_mask, -1)
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_times_full_list_embed = tf.squeeze(usage_times_full_embed_expanded, axis=-1)
        usage_times_list_full_embed_flatten = tf.layers.flatten(usage_times_full_list_embed)
        usage_times_full_embed = self.usage_full_times(usage_times_list_full_embed_flatten)
        usage_times_full_conved = tf.squeeze(self.usage_times_full_conv(usage_times_full_embed_expanded), axis=[1, -1])
        usage_times_full_mean = tf.reduce_mean(usage_times_full_list_embed, axis=1)
        # ========================================== transfomer encoder part ===================================================
        appId_duration_encoder = self.usage_appid_duration_encoder(tf.slice(inputs['usage_appId_duration_list'], [0, 0], [-1, 256]), False,None)
        appId_duration_encoder_flatten = tf.layers.flatten(appId_duration_encoder)
        appId_duration_encoder_feature = self.usage_appid_duration_encoder_feature_trans(appId_duration_encoder_flatten)
        appId_duration_encoder_fc_out = self.usage_appid_duration_encoder_mlp(appId_duration_encoder_flatten)
        # ==================================== LSTM ================================
        usage_appId_duration_embed_expanded = self.embeding_layer_dict['appId_list_lstm_encoded'](
            tf.cast(inputs['usage_appId_duration_list'], dtype=tf.int32))
        # appId_embed_expanded = tf.where(active_mask, appId_embed_expanded, tf.zeros_like(appId_embed_expanded))
        usage_appId_duration_list_embed = tf.squeeze(usage_appId_duration_embed_expanded, axis=-1)
        usage_appid_duration_lstm_out = self.usage_appid_duration_lstm(usage_appId_duration_list_embed)
        usage_appid_duration_lstm_fatten = tf.layers.flatten(usage_appid_duration_lstm_out)
        usage_appid_duration_lstm_transed_out = self.usage_appid_duration_lstm_trans(usage_appid_duration_lstm_fatten)
        usage_appid_duration_lstm_mlp_out = self.usage_appid_duration_lstm_mlp(usage_appid_duration_lstm_fatten)

        # usage_appid_duration_lstm_transed_meanout = tf.reduce_mean(usage_appid_duration_lstm_transed, axis=1)
        # ================================== common ============================================================================
        exced_fea = [usage_appid_times_embed,usage_appid_times_conved,usage_appid_times_mean]  # appId_cate_list_embed
        
        exced_full_fea = [usage_appId_full_embed, usage_appId_full_mean,usage_appId_full_conved,
                          usage_duration_full_embed, usage_duration_full_mean,usage_duration_full_conved,
                          usage_times_full_embed, usage_times_full_conved, usage_times_full_mean
                          ]
        all_fea = list(cate_fea_embeding.values()) + [appId_embed, active_conved, appId_mean, concat_func_fea_transed, cate_one_hot_fea] + exced_full_fea+[usage_appid_duration_lstm_transed_out]#  +exced_fea
        concat_all = tf.concat(all_fea, axis=-1)  # +[num_fea_af_fc]
        # ===================================================   common part ======================================
        concat_all_reshaped = self.mine_reshape(concat_all)
        cin_out = self.cin(concat_all_reshaped)
        bi_out = self.bi(concat_all_reshaped)
        fm_out = self.fm(concat_all_reshaped)
        mlp_out = self.mlp(concat_all)
        cin_fc_out = self.cin_fc(cin_out)
        bi_fc_out = self.bi_fc(bi_out)
        fm_fc_out = self.fm_fc(fm_out)
        
        out = tf.concat([
                         tf.expand_dims(mlp_out, 1),
                         tf.expand_dims(fm_fc_out, 1),
                         tf.expand_dims(bi_fc_out, 1),
                         tf.expand_dims(cin_fc_out, 1),
                         tf.expand_dims(usage_appid_duration_lstm_mlp_out, 1),
                         tf.expand_dims(tf.nn.softmax(tf.reduce_mean(cate_fea_bias, axis=0)), 1)], axis=1)
                         # # tf.expand_dims(activeapp_encoder_fc_out, 1),
                         # tf.expand_dims(usage_appid_duration_lstm_mlpout, 1),

        out = tf.squeeze(self.conv_1(tf.expand_dims(out, axis=-1)), axis=[1, -1])
        
        # out = tf.concat([
        #     tf.expand_dims(usage_appid_duration_lstm_transed_out, 1),
        #     tf.expand_dims(out, 1)], axis=1)
        # out = tf.squeeze(self.conv_2(tf.expand_dims(out, axis=-1)), axis=[1, -1])
        out = tf.nn.softmax(usage_appid_duration_lstm_mlp_out)
        
        return out
    
    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = 1
        return tf.TensorShape(shape)
