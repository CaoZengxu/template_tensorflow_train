import tensorflow as tf


def f1(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_pred = tf.one_hot(y_pred, 12, 1, 0)
    y_pred = tf.cast(y_pred, tf.float32)
    true_p = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    true_n = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    
    false_p = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    false_n = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
    acc = (true_p + true_n) / (true_p + true_n + false_p + false_n + 0.0000001)
    weighted = tf.reduce_sum(y_true, axis=0) / (true_p + true_n + false_p + false_n)
    
    precisions = true_p / (true_p + false_p + 0.0000001)
    recall = true_p / (true_p + false_n + 0.0000001)
    
    f1 = 2 * precisions * recall / (precisions + recall + 0.0000001)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = tf.reduce_sum(weighted * f1)
    return f1


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_mean(
            (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
    
    return focal_loss_fixed

def focal_loss_fixed(y_true, y_pred,gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_mean(
        (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))