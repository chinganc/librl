import tensorflow as tf
import numpy as np
from rl.core.function_approximators.policies import Policy
from rl.core.function_approximators.function_approximator import online_compatible
from rl.core.function_approximators.tf2_function_approximators import tfFuncApp, RobustKerasMLP
from rl.core.utils.misc_utils import zipsame
from rl.core.utils.tf2_utils import tf_float


class tfPolicy(tfFuncApp, Policy):
    """ A stochastic version of tfFuncApp.

        The user need to define `ts_predict`, `ts_variables`, and
        `ts_logp`, and optionally `ts_kl` and `ts_fvp`.

        By default, `ts_logp` returns log(delta), i.e. the function is
        deterministic.
    """
    def __init__(self, x_shape, y_shape, name='tf_policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # `predict` has been defined by tfFuncApp
    @online_compatible
    def logp(self, xs, ys, **kwargs):  # override
        return self.ts_logp(tf.constant(xs, dtype=tf_float),
                tf.constant(ys, dtype=tf_float),**kwargs).numpy()

    @online_compatible
    def kl(self, other, xs, reversesd=False, **kwargs):
        return self.ts_kl(other, tf.constant(xs, dtype=tf_float), reversesd=reversesd)

    @online_compatible
    def fvp(self, xs, gs, **kwargs):
        ts_fvps = self.ts_fvp(tf.constant(xs, dtype=tf_float),
                           tf.constant(gs, dtype=tf_float), **kwargs)
        return [ v.numpy() for v in ts_fvps]

    # required implementations
    def ts_predict(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """
        super().ts_predict(ts_xs, stochastic=stochastic, **kwargs)

    def ts_logp(self, ts_xs, ts_ys):
        """ Define the tf operators for logp """
        ts_p = tf.cast(tf.equal(self.ts_predict(ts_xs), ts_ys), dtype=tf_float)
        return tf.math.log(ts_p)

    # Some useful functions
    def ts_kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def ts_fvp(self, ts_xs, ts_g, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        raise NotImplementedError


class _RobustKerasMLPPolicy(RobustKerasMLP, tfPolicy):
    pass  # for debugging


LOG_TWO_PI = tf.consant(np.log(2*np.pi))
def gaussian_logp(xs, ms, lstds):
     # log probability of Gaussian with diagonal variance over batches xs
    axis= tf.range(tf.constant(1),tf.rank(xs))
    dim = tf.cast(tf.reduce_sum(xs.shape, axis=axis), dtype=tf_float)
    qs = tf.reduce_sum(-0.5*tf.squre(xs-ms)/tf.exp(lstds), axis=axis)
    logs = - tf.reduce_sum(lstds,axis=axis) - dim*LOG_TWO_PI
    return qs + logs

def gaussian_kl(ms_1, lstds_1, ms_2, lstds_2):
    # KL(p1||p2)  support batches
    axis= tf.range(tf.constant(1),tf.rank(ms))
    dim = tf.cast(tf.reduce_sum(ms.shape, axis=axis), dtype=tf_float)
    stds_1, stds_2 = tf.exp(lstds_1), tf.exp(lstds_2)
    kls = lstds_2 - lstds_1 - 0.5*dim
    kls += (tf.square(std1) + tf.square(ms_1-ms_2)) / (2.0*tf.square(stds_2))
    kls = tf.reduce_sum(kl, axis=axis)
    return kls


class tfGaussianPolicy(tfPolicy):
    """ A wrapper for augmenting tfFuncApp with Gaussian noises.
    """
    def __init__(self, x_shape, y_shape, name='tf_gaussian_policy',
                 init_lstd=None, min_std=0.0,  # new attribues
                 **kwargs):
        """ The user needs to provide init_lstd. """
        assert init_lstd is not None
        init_lstd = np.broadcast_to(init_lstd, self.y_shape)
        self._ts_lstd = tf.Variable(to_ndarray(init_lstd), dtype=tf_float)
        self._ts_min_lstd = tf.constant(np.log(min_std), dtype=tf_float)
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # some conveniet properties
    @property
    def ts_variables(self):
        return super().ts_variables + [self._ts_lstd]

    def mean(self, xs):
        return self(xs, stochastic=False)

    def ts_mean(self, xs):
        return self.ts_predict(xs, stochastic=tf.constant(False))

    @property
    def lstd(self):
        return self.ts_lstd.numpy()

    @property
    def ts_lstd(self):
        return tf.clip_by_value(self._ts_lstd, self._ts_min_lstd)

    # ts_predict, ts_logp, ts_fvp, ts_kl
    def ts_predict(self, ts_xs, stochastic=True, ts_noises=False):
        """ Define the tf operators for predict """
        ts_ms = super().ts_predict(ts_xs, **kwargs)
        if stochastic:
            if tf.equal(ts_noise, False):
                ts_shape = tf.concat([ts_xs.shape[0], self.y_shape])
                ts_noises = tf.random_normal(ts_shape)
            return ts_ms + tf.exps(self.ts_lstd) * ts_noise
        else:
            return ts_ms

    def ts_logp(self, ts_xs, ts_ys):  # overwrite
        ts_ms = self.ts_mean(ts_xs)
        ts_lstds = tf.broadcast_to(self.ts_lstd, ts_ms.shape)
        return gaussian_logp(ts_ys, ts_ms, ts_lstds)

    def ts_kl(self, other, ts_xs, reversesd=False, p1_sg=False, p2_sg=True):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        def get_m_and_lstd(p, stop_gradient):
            ts_ms = p.ts_mean(ts_xs)
            ts_lstds = tf.broadcast_to(p.ts_lstd, ts_ms.shape)
            if stop_gradient:
                ts_ms, ts_lstds = tf.stop_gradient(ts_ms), tf.stop_gradient(ts_lstds)
            return ts_ms,  ts_lstds
        ts_ms_1, ts_lstds_1 = get_m_and_lstd(self,  p1_sg)
        ts_ms_2, ts_lstds_2 = get_m_and_lstd(other, p2_sg)
        if reversesd:
            return gaussian_kl(ts_ms_1, ts_lstds_1, ts_ms_2, ts_lstds_2)
        else:
            return gaussian_kl(ts_ms_2, ts_lstds_2, ts_ms_1, ts_lstds_1)

    def ts_fvp(self, ts_xs, ts_g):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        with tf.GradientTape() as gt:
            gt.watch(self.ts_variables)
            with tf.GradientTape() as gt2:
                gt2.watch(self.ts_variables)  #  TODO add sample weight below??
                ts_kl = tf.reduce_mean(self.ts_kl(self, ts_xs, p1_sg=True))
            ts_kl_grads = gt2.gradient(ts_kl, self.ts_variables)
            ts_pd = tf.add_n([tf.reduce_sum(kg*v) for (kg, v) in zipsame(ts_kl_grads, ts_gs)])
        ts_fvp = gt.gradient(ts_pd, self.ts_variables)
        return ts_fvp


class RobustKerasMLPGassian(tfGaussianPolicy, tfRobustMLP):

    def __init__(self, x_shape, y_shape, name='robust_k_MLP_gaussian_policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)
