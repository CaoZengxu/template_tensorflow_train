import tensorflow as tf;
from tensorflow.python.keras.initializers import *  # glorot_uniform, zero, Constant
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras import backend as K;
from tensorflow.python.keras.regularizers import l2;


class HidFeatLayer(Layer):
    def __init__(self, space_size, out_dim, zero_lim=False, cus_name=None, name=None, **kwargs):
        self.output_dim = out_dim;
        self.space_size = space_size;
        self.cus_name = cus_name
        self.zero_lim = zero_lim
        super(HidFeatLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        
        self.ker = self.add_weight(name=self.cus_name + '_' + 'hid_feat' if self.cus_name is not None else 'hid_feat',
                                   shape=(self.space_size, self.output_dim),
                                   initializer=glorot_normal(seed=2019),
                                   dtype='float32',
                                   trainable=True)
        super(HidFeatLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        choosed = tf.expand_dims(tf.nn.embedding_lookup(self.ker, x), -1)
        return choosed
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class HidFeatLayer_3D(Layer):
    def __init__(self, space_size, out_dim, field, zero_lim=False, cus_name=None, name=None, **kwargs):
        self.output_dim = out_dim
        self.space_size = space_size
        self.field = field
        self.cus_name = cus_name
        self.zero_lim = zero_lim
        super(HidFeatLayer_3D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        
        self.ker = self.add_weight(name=self.cus_name + '_' + 'hid_feat' if self.cus_name is not None else 'hid_feat',
                                   shape=(self.space_size, self.output_dim, self.field),
                                   initializer=glorot_normal(seed=2019),
                                   dtype='float32',
                                   trainable=True)
        super(HidFeatLayer_3D, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        choosed = tf.expand_dims(tf.nn.embedding_lookup(self.ker, x), -1)
        return choosed
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.field)


class K_Value_numrical_layer(Layer):
    def __init__(self, output_dim, key_nums, hops=None, **kwargs):
        self.output_dim = output_dim;
        self.hops = hops
        self.key_nums = key_nums
        self.m_values_num = key_nums
        self.m_values_featture_size = output_dim
        self.r_list = []
        super(K_Value_numrical_layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        index = [(i + 0.5) / self.key_nums for i in range(self.key_nums)]
        self.index = tf.constant(index)
        self.m_values = self.add_weight(name='m_values',
                                        shape=(self.key_nums, self.output_dim),
                                        initializer=glorot_normal(seed=2019),
                                        dtype='float32',
                                        trainable=True)
    
    def call(self, input):
        distance = 1 / (tf.abs(input[:, None] - self.index[None, :]) + 0.00001)
        weights = tf.nn.softmax(distance, -1)
        weights = tf.squeeze(weights)
        out = K.math_ops.matmul(weights, self.m_values)
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Binteraction(Layer):
    def __init__(self, **kwargs):
        super(Binteraction, self).__init__(**kwargs)
    
    def build(self, input_shape):
        pass
    
    def call(self, input):
        sumed = tf.keras.backend.sum(input, axis=1)
        sum_square = tf.keras.backend.square(sumed)
        squared = tf.keras.backend.square(input)
        squared_sum = tf.keras.backend.sum(squared, axis=1)
        out = sum_square - squared_sum
        
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CIN(Layer):
    def __init__(self, vector_length, feature_num, activition, cross_layer_sizes, bias=True, direct=True, res=True,
                 **kwargs):
        self.vec_len = vector_length
        self.feature_num = feature_num
        self.activition = activition
        self.bias = bias
        self.cross_layer_sizes = cross_layer_sizes
        self.direct = direct
        self.res = res
        super(CIN, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.filter_list = []
        field_nums = []
        field_nums.append(self.feature_num)
        for idx, layer_size in enumerate(self.cross_layer_sizes):
            # tmp = tf.(name="f_" ,shape=[1, field_nums[-1] * field_nums[0], layer_size],dtype=tf.float32)
            filters = self.add_weight(name='f_' + str(idx), shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                      initializer=glorot_normal(seed=2019), dtype='float32', trainable=True)
            # filters = tf.get_variable(name="f_" + str(idx),shape=[1, field_nums[-1] * field_nums[0], layer_size],initializer=glorot_normal(seed=2019),dtype=tf.float32)
            field_nums.append(int(layer_size))
            self.filter_list.append(filters)
        pass
    
    def call(self, inputs, **kwargs):
        final_len = 0
        field_nums = []
        hidden_nn_layers = []
        final_result = []
        hidden_nn_layers.append(inputs)
        field_nums.append(self.feature_num)
        split_tensor0 = tf.split(hidden_nn_layers[0], self.vec_len * [1], 2)
        split_tensor0 = split_tensor0
        for idx, layer_size in enumerate(self.cross_layer_sizes):
            split_tensor = tf.split(hidden_nn_layers[-1], self.vec_len * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[self.vec_len, -1, field_nums[0] * field_nums[-1]])  # 撸平
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
            # dot_result = tf.transpose(dot_result_o, perm=[1, 2, 0])
            
            dot_result = dot_result
            filter = self.filter_list[idx]
            curr_out = tf.nn.conv1d(dot_result, filters=filter, stride=1, padding='VALID')
            # curr_out = tf.keras.layers.SeparableConv1D(filters=64, kernel_size=field_nums[-1] * field_nums[0], strides=1, padding='VALID', data_format='channels_last')(dot_result)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            if self.direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_nums.append(int(layer_size))
            else:
                if idx != len(self.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))
            
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)
        return result
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], sum(self.cross_layer_sizes))


class Conv_Sequence(Layer):
    def __init__(self, filter_size, embedding_size, num_filters, sequence_length=None, pool=True, **kwargs):
        super(Conv_Sequence, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.pool = pool
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=[self.filter_size, self.embedding_size, 1, self.num_filters],
                                 initializer=glorot_normal(seed=2019), name='conv_w')
        self.b = self.add_weight(shape=[self.num_filters], initializer=glorot_normal(seed=2019), name='conv_bias')
    
    def call(self, input):
        conv = tf.nn.conv2d(input, self.w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, self.b), name="relu")
        out = h
        if self.pool:
            pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - self.filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")
            out = pooled
        return out
    
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[-1])


class FM_socond_part(Layer):
    def __init__(self, dropout_rate=1, cus_name=None, name=None, **kwargs):
        self.dropout_rate = dropout_rate
        self.cus_name = cus_name
        super(FM_socond_part, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(FM_socond_part, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        # sum_square part
        summed_features_emb = tf.reduce_sum(x, axis=1)  # None * K
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K
        
        # square_sum part
        squared_features_emb = tf.square(x)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)  # None * K
        
        # second order
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
        y_second_order = tf.nn.dropout(y_second_order, self.dropout_rate)  # None * K
        return y_second_order
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Attention(Layer):
    
    def __init__(self, output_dim,sequenceLength,drop_out_rate = 0, **kwargs):
        self.output_dim = output_dim
        self.sequenceLength = sequenceLength
        self.drop_out_rate =drop_out_rate
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.hiddenSize = int(input_shape[2])
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=[3, int(input_shape[2]), self.output_dim],
        #                               initializer='uniform',
        #                               trainable=True)

        self.wei = self.add_weight(name='kernel',
                                      shape=[self.hiddenSize],
                                      initializer='uniform',
                                      trainable=True)

        super(Attention, self).build(input_shape)  # 一定要在最后调用它
    
    def call(self, x):
        M = tf.tanh(x)
        newM = tf.matmul(tf.reshape(M, [-1, self.hiddenSize]), tf.reshape(self.wei, [-1, 1]))
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.sequenceLength])
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        r = tf.matmul(tf.transpose(x, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r,axis=-1)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = sentenceRepren
        if self.drop_out_rate>0:
            output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def Encoder(x, structure, encode_index,name=None):
    out = x
    encoed_out = None
    used_name = None
    for i, unit in enumerate(structure):
        # out = tf.keras.layers.BatchNormalization()(out)
        if i == len(structure)-1:
            used_name = name
        out = tf.keras.layers.Dense(unit, activation=tf.keras.activations.relu, kernel_initializer=glorot_uniform(seed=2019),
                    kernel_regularizer=l2(0), name=used_name)(out)
        # if i < encode_index:
        #   out = Dropout(0.005)(out)
        if i == encode_index:
            encoed_out = out
    return encoed_out, out


class CusAutoEncoder(Layer):
    def __init__(self, structure, encode_index, dropout_rate=1, cus_name=None, name=None, **kwargs):
        self.dropout_rate = dropout_rate
        self.cus_name = cus_name
        self.structure =structure
        self.encode_index = encode_index
        super(CusAutoEncoder, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.layer_list = []
        for i, unit in enumerate(self.structure):
            # out = tf.keras.layers.BatchNormalization()(out)
            layer= tf.keras.layers.Dense(unit, use_bias=True,activation=tf.keras.activations.relu,
                                        kernel_initializer=glorot_uniform(seed=2019),
                                        kernel_regularizer=l2(0.01))
            # if i < encode_index:
            #   out = Dropout(0.005)(out)
            self.layer_list.append(layer)
            
        super(CusAutoEncoder, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x):
        out = x
        encoded_out= None
        for i, dense_layer in enumerate(self.layer_list):
            out = dense_layer(out)
            if i == self.encode_index:
                encoded_out = out
        return encoded_out, out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.structure[self.encode_index]),(input_shape[0],self.structure[-1])