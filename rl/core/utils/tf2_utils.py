import tensorflow as tf
import numpy as np

tf_float = tf.float32
tf_int = tf.int32


""" Conversion between tf.Tensor(s) and np.ndarray(s) """
def ts_to_array(ts_x):
    # convert tf.Tensor(s) to np.ndarray(s)
    if type(ts_x) is list:
        return [ts_xx.numpy() for t in ts_x]
    else:
        return ts_x.numpy()

def array_to_ts(x):
    # convert np.ndarray(s) to tf.Tensor(s)
    if type(x) is list:
        return [tf.constant(xx,dtype=tf_float) for xx in x]
    else:
        return tf.constant(x, dtype=tf_float)
