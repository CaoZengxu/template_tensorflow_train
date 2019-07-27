import tensorflow as tf
import tensorflow.contrib.eager as tfe
from .model_define import *
from nn_modules.custom_metrics import f1
from tensorflow.python.eager import tape
# tf.enable_eager_execution()


def make_iterator(tensors, batch_size=32):
    with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_tensor_slices(tensors).batch(batch_size)  # .repeat()
    return ds.make_one_shot_iterator()
    # return tfe.Iterator(ds)


def compute_gradients_and_loss(model, inputs, labels):
    with tf.GradientTape() as grad_tape:
        batch_pre = model(inputs, training=True)
        loss = tf.keras.losses.binary_crossentropy(batch_pre, labels)
        # tf.summary.scalar(name='loss', tensor=loss)

    # TODO(b/110991947): We can mistakenly trace the gradient call in
    # multi-threaded environment. Explicitly disable recording until
    # this is fixed.
    # with tape.stop_recording():
    grads = grad_tape.gradient(loss, model.variables)
    return grads,loss,batch_pre


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables), global_step=tf.train.get_or_create_global_step())


class Modeo_eager():
    def __init__(self, param_dict):
        self.lr = param_dict['lr']
        self.epoch = param_dict['epoch']
        self.batch_size = param_dict['batch_size']
        # self.embed_feature_size_list = param_dict['embed_faeture_list']
        # self.MLP = param_dict['MLP']
        # self.input_dim = param_dict['input_dim']
        self.drop_rate = param_dict['drop_rate']
        self.reg_rate = param_dict['reg_rate']
        # self.vector_length = param_dict['vector_length']
        # self.subclass = param_dict['subclass']
        self.device = '/gpu:0' if tfe. num_gpus() else '/cpu:0'
        self.model = czx_NN_subclass(param_dict)

    def train(self, x, y, val, execution_mode=None):
        device = '/gpu:0' if tfe.num_gpus() else '/cpu:0'
        log = {'epoch_list':[],'train_binary_crossentropy':[],'train_roc':[],'val_roc':[]}
        # with tfe.execution_mode(execution_mode):
        optimizer = tf.train.AdagradOptimizer(self.lr)
        weight_file = '/nn_modules/best_eager_MLP_weight'
        no_improve = 0 # no_improve 技计数 用于early_stoping
        max_score = 0
        with tf.device(device): # 指定硬件
            for epoch in range(self.epoch): # epoch 寻环
                train_iterator = make_iterator((x, y), batch_size=self.batch_size) # 生成iterator
                loss_history = []
                full_y_pred = []
                while True: # batch 寻环，利用try停止epoch
                    try:
                        batch_x, batch_y = train_iterator.get_next()
                        grads, loss, batch_pre = self.compute_gradients_and_loss(batch_x, batch_y)
                        
                        self.apply_gradients(optimizer, grads)
                        loss_history.append(loss.numpy())
                        full_y_pred.append(batch_pre.numpy())
                        # tfe.async_wait()
                    except tf.errors.OutOfRangeError:
                        break
                full_y_pred = np.concatenate(full_y_pred)
                val_score = self.validate(val[0],val[1])
                if val_score > max_score:
                    max_score = val_score
                    no_improve = 0
                    self.model.save_weights(weight_file)
                else:
                    no_improve += 1
                    if no_improve > 10:
                        print("early stop at epoch %d"%(no_improve))
                        break
                self.model.load_weights(weight_file)
                train_score = f1(y, full_y_pred)
                epoch_loss = np.mean(loss_history)
                log['epoch_list'].append(epoch)
                log['train_binary_crossentropy'].append(epoch_loss)
                log['train_roc'].append(train_score)
                log['val_roc'].append(val_score)
                print("epoch=%d,loss=%.6f,train_roc=%.6f,val_roc=%.6f,time=%s" % (epoch, epoch_loss, train_score, val_score, time.asctime()))

        datafe = pd.DataFrame(log)
        datafe.to_csv('performance_log.csv')
        
    def predict(self,x, batch_size=32):
        test_iterator = make_iterator(x, batch_size=batch_size)
        full_pre = []
        with tf.device(self.device):
            while True:
                try:
                    batch_x = test_iterator.get_next()
                    pre = self.model(batch_x, training=False)
                    full_pre.append(pre.numpy())
                except tf.errors.OutOfRangeError:
                    break
        full_pre = np.concatenate(full_pre)
        return full_pre
    
    def validate(self, x, y, batch_size = 32):
        full_pre = self.predict(x,batch_size = 32)
        score = f1(y, full_pre)
        return score

    def compute_gradients_and_loss(self,inputs, labels):
        """
        :param inputs: 输入
        :param labels: 标签
        :return: 梯度，损失值，对batch_x的预测值
        """
        with tf.GradientTape() as grad_tape:
            batch_pre = self.model(inputs, training=True)
            loss = tf.keras.losses.categorical_crossentropy(labels, batch_pre)
            # tf.summary.scalar(name='loss', tensor=loss)
    
        # TODO(b/110991947): We can mistakenly trace the gradient call in
        # multi-threaded environment. Explicitly disable recording until
        # this is fixed.
        # with tape.stop_recording():
        grads = grad_tape.gradient(loss, self.model.variables)
        return grads, loss, batch_pre

    def apply_gradients(self, optimizer, gradients):
        """‘
        根据梯度优化优化
        :param optimizer:指定优化器
        :param gradients: 指定梯度值
        """
        optimizer.apply_gradients(zip(gradients, self.model.variables), global_step=tf.train.get_or_create_global_step())

