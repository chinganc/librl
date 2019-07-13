import collections
import tensorflow as tf
import numpy as np
from functools import wraps
from abc import abstractmethod

from rl.tools.function_approximators.tf2_function_approximators import tf2FuncApp, tf2RobustFuncApp, KerasFuncApp

tf_float = U.tf_float


class tf2Policy(tf2FuncApp, Policy):

    def __init__(self, x_shape, y_shape, name='tf2_policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # `predict` has been defined by tf2FuncApp
    def compute_logp(self, xs, ys, **kwargs)
        return self.ts_predict(tf.constant(xs, dtype=tf_float),
                               tf.constant(ys, dtype=tf_float),
                               **kwargs).numpy()

    @online_compatible
    def kl(self, other, xs, reversesd=False, **kwargs):
        return self.ts_kl(other, tf.constant(xs, dtype=tf_float, reversesd=reversesd)

    @online_compatible
    def fvp(self, xs, gs, **kwargs):
        return self.ts_fvp(tf.constant(xs, dtype=tf_float),
                           tf.constant(gs, dtype=tf_float),
                           **kwargs)

    # required implementations
    @abstractmethod
    def ts_predict(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """

    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """

    @abstractmethod
    def ts_compute_logp(self, xs, ys):
        """ Define the tf operators for compute_logp """

    # Some useful functions
    def ts_kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def ts_fvp(self, xs, gs, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        raise NotImplementedError


class tfGaussianPolicy(tfPolicy):
    """
    An abstract class which is namely tfPolicy with Gaussian distribution based
    on tfFunctionApproximator.
    """
    @staticmethod
    def _build_logp(dim, rvs, mean, logstd):  # log probability of Gaussian
        return (-0.5 * U.squared_sum((rvs - mean) / tf.exp(logstd), axis=1) -
                tf.reduce_sum(logstd) - 0.5 * np.log(2.0 * np.pi) * tf.to_float(dim))

    @classmethod
    def build_kl(cls, p1, p2, p1_sg=False, p2_sg=False, w=None):
        """KL(p1, p2). p1_sg: whether stop gradient for p1."""
        def get_attr(p, sg):
            logstd, mean = p.ts_logstd, p.ts_mean
            if sg:
                logstd, mean = tf.stop_gradient(logstd), tf.stop_gradient(mean)
            std = tf.exp(logstd)
            return logstd, std, mean
        logstd1, std1, mean1 = get_attr(p1, p1_sg)
        logstd2, std2, mean2 = get_attr(p2, p2_sg)
        kl = logstd2 - logstd1 - 0.5
        kl += (tf.square(std1) + tf.square(mean1 - mean2)) / (2.0 * tf.square(std2))
        kl = tf.reduce_sum(kl, axis=-1)  # reduce over ac dimension
        if w is None:
            kl = tf.reduce_mean(kl)  # average over x
        else:  # weighted average over x
            kl = tf.reduce_sum(kl * w) / tf.reduce_sum(w)
        return kl


def _tfGaussianPolicyDecorator(cls):
    """
    A decorator for defining tfGaussianPolicy via tfFunctionApproximator.
    It is mainly tfGaussianPolicy but uses cls' _build_func_apprx.
    """
    assert issubclass(cls, tfFunctionApproximator)

    class decorated_cls(tfGaussianPolicy, cls):

        @tfObject.save_init_args()
        def __init__(self, x_dim, y_dim, init_logstd,
                     name='tfGaussianPolicy', seed=None,
                     build_nor=None, max_to_keep=None,
                     min_std=0.1, **kwargs):
            # Since tfPolicy only modifies the __init__ of tfFunctionApproximator
            # in adding _kl_cache, we reuse the __init__ of cls.

            # new attributes
            self.ts_mean = self.ts_logstd = self._ts_stop_std_grad = None
            self._init_logstd = init_logstd
            self._min_std = min_std

            # construct a tfFunctionApproximator
            cls.__init__(self, x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep,
                         **kwargs)

            # add the cache of tfPolicy
            self._kl_cache = collections.defaultdict(lambda: None)

        @property
        def std(self):  # for convenience
            return self._std()

        @std.setter
        def std(self, val):
            self._set_logstd(np.log(val))

        def stop_std_grad(self, cond=True):
            self._set_stop_std_grad(cond)

        def _build_dist(self, ts_nor_x, ph_y):
            # mean and std
            self.ts_mean = cls._build_func_apprx(self, ts_nor_x)  # use the tfFunctionApproximator to define mean
            self._ts_logstd = tf.get_variable(
                'logstd', shape=[self.y_dim], initializer=tf.constant_initializer(self._init_logstd))
            self._ts_stop_std_grad = tf.get_variable('stop_std_grad', initializer=tf.constant(False), trainable=False)
            _ts_logstd = tf.cond(self._ts_stop_std_grad,  # whether to stop gradient
                                 true_fn=lambda: tf.stop_gradient(self._ts_logstd),
                                 false_fn=lambda: self._ts_logstd)
            # make sure the distribution does not degenerate
            self.ts_logstd = tf.maximum(tf.to_float(np.log(self._min_std)), _ts_logstd)
            ts_std = tf.exp(self.ts_logstd)
            self._std = U.function([], ts_std)
            self._set_logstd = U.build_set([self._ts_logstd])
            self._set_stop_std_grad = U.build_set([self._ts_stop_std_grad])

            # pi
            self.ts_noise = tf.random_normal(tf.shape(ts_std), stddev=ts_std, seed=self.seed)
            ts_pi = self.ts_mean + self.ts_noise
            ts_pid = self.ts_mean
            # logp
            ts_logp = self._build_logp(self.y_dim, ph_y, self.ts_mean, self.ts_logstd)
            return ts_pi, ts_logp, ts_pid

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfGaussianPolicyDecorator
class tfGaussianMLPPolicy(tfMLPFunctionApproximator):
    pass
