import numpy as np
import tensorflow as tf
from abc import abstractmethod
from functools import wraps
from rl.core.utils.misc_utils import zipsame, flatten
from rl.core.function_approximators.normalizers import NormalizerStd
from rl.core.oracles.oracle import Oracle
from rl.core.utils.tf2_utils import tf_float, ts_to_array

class tfOracle(Oracle):
    """ A minimal wrapper of tensorflow functions. """
    def __init__(self, tf_fun, **kwargs):
        self.ts_fun = ts_fun  # a function that returns tf.Tensor(s)
        self.ts_loss = None
        self.ts_g = None

    @ts_to_array
    def fun(self, x=None, **kwargs):
        """ If x is not provided, the cached value from the previous call of
        `fun` or `grad` will be returned. """
        if x is not None:
            x = tf.constant(x, dtype=tf_float)
            self.ts_loss = self.ts_fun(x, **kwargs)
        return self.ts_loss

    @ts_to_array
    def grad(self, x=None, **kwargs):
        """ If x is not provided, the cached value from the previous call of
         `grad` will be returned. """
        if x is None:
            x = tf.constant(x, dtype=tf_float)
            with tf.GradientTape() as tape:
                tape.watch(x)
                self.ts_loss = self.ts_fun(x, **kwargs)
            self.ts_g = tape.gradient(self.ts_loss, x)
        return self.ts_g


class tfLikelihoodRatioOracle(tfOracle):
    """
    An Oracle based on the loss function below: if use_log_loss is True

        E_{x} E_{y ~ q | x} [ w * log p(y|x) * f(x, y) ]

    otherwise, it uses

        E_{x} E_{y ~ q | x} [ p(y|x)/q(y|x) * f(x, y) ]

    where p is the variable distribution, q is a constant
    distribution, and f is a scalar function.

    When w = p/q, then the gradients of two loss functions are equivalent.

    The expectation is approximated by unbiased samples from q. To minimize
    the variance of sampled gradients, the implementation of 'grad' is
    based on a normalizer, which can shift, rescale, or clip f.

    """
    def __init__(self, ts_logp_fun, nor=None, biased=False,
                 use_log_loss=False, normalized_is=False):
        assert use_log_loss in (True, False, None)
        self._ts_logp_fun = ts_logp_fun
        self._biased = biased
        self._use_log_loss = use_log_loss
        self._normalized_is = normalized_is  # normalized importance sampling
        if nor is None:
            if biased:  # use the current samples
                self._nor = NormalizerStd((1,), unscale=True, clip_thre=None, momentum=0.0)
            else:  # use a moving average
                self._nor = NormalizerStd((1,), unscale=True, clip_thre=None, momentum=None)
        else:
            self._nor = nor
        self._ts_loss = None
        super().__init__(ts_fun=self.ts_loss)

    def ts_loss(self, ts_x):
        """ Return the loss function as tf.Tensor and a list of tf.plyeholders
        required to evaluate the loss function. """
        ts_logp = self._ts_logp_fun(ts_x)  # the function part
        if tf.equal(self._use_log_loss, True):  # ts_w_or_logq is w
            ts_w = ts_w_or_logq
            ts_loss = tf.reduce_sum(ts_w * ts_f * ts_logp)
        elif tf.equal(self._use_log_loss, False): # ts_w_or_logq is logq
            ts_w = tf.exp(ts_logp - ts_w_or_logq)
            ts_loss = tf.reduce_sum(ts_w*ts_f)
        else:  # ts_w_or_logq is logq
            # The function value is pointwise as self._use_log_loss==True, but
            # its gradient behaves like self._use_log_loss==True.
            ts_w = tf.stop_gradient(tf.exp(ts_logp - ts_w_or_logq))
            ts_loss = tf.reduce_sum(ts_w * ts_f * ts_logp)
        if self._normalized_is:  # normalized importance sampling
            return ts_loss / tf.reduce_sum(ts_w)
        else: # regular importance sampling
            return ts_loss / tf.cast(ts_x.shape[0], tf_float)

    def update(self, f, w_or_logq, update_nor=True):
        """ Update the function with Monte-Carlo samples.

            f: sampled function values
            w_or_logq: importance weight or the log probability of the sampling distribution
            update_nor: whether to update the normalizer using the current sample
        """
        if self._biased:  # always update
            self._nor.update(f)
        f_normalized = self._nor.normalize(f)  # np.ndarray
        if self._use_log_loss:  # ts_w_or_logq is w
            assert np.all(w_or_logq >= 0)
        # these are treated as constants
        self._ts_f = tf.constant(f_normalized, dtype=tf_float)
        self._ts_w_or_logq = tf.constant(w_or_logq, dtype=tf_float)
        if not self._biased and update_nor:
            self._nor.update(f)
