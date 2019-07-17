import tensorflow as tf
import numpy as np
from rl.core.function_approximators.policies import Policy
from rl.core.function_approximators.function_approximator import online_compatible
from rl.core.function_approximators.tf2_function_approximators import tfFuncApp, RobustKerasMLP, KerasFuncApp, RobustKerasFuncApp
from rl.core.utils.misc_utils import zipsame
from rl.core.utils.tf2_utils import tf_float, array_to_ts, ts_to_array
from rl.core.utils.misc_utils import flatten, unflatten


class tfPolicy(tfFuncApp, Policy):
    """ A stochastic version of tfFuncApp.

        The user need to define `ts_predict`, `ts_variables`, and optionally
        `ts_logp`, `ts_kl` and `ts_fvp`.

        By default, `ts_logp` returns log(delta), i.e. the function is assumed
        to be deterministic. Therefore, it can be used a wrapper of subclass of
        `tfFuncApp`. For example, for a subclass `A`, one can define

            class B(tfPolicy, A):
                pass

        which creates a deterministic tfPolicy.
    """
    def __init__(self, x_shape, y_shape, name='tf_policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # `predict` has been defined by tfFuncApp
    @online_compatible
    def logp(self, xs, ys, **kwargs):  # override
        return self.ts_logp(array_to_ts(xs), array_to_ts(ys), **kwargs).numpy()

    def kl(self, other, xs, reversesd=False, **kwargs):
        """ Return the KL divergence for each data point in the batch xs. """
        return self.ts_kl(other, array_to_ts(xs), reversesd=reversesd)

    def fvp(self, xs, g, **kwargs):
        """ Return the product between a vector g (in the same formast as
        self.variable) and the Fisher information defined by the average
        over xs. """
        gs = unflatten(g, shapes=self.var_shapes)
        ts_fvp = self.ts_fvp(array_to_ts(xs), array_to_ts(gs), **kwargs)
        return flatten([v.numpy() for v in ts_fvp])

    # required implementations
    def ts_predict(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """
        return super().ts_predict(ts_xs, stochastic=stochastic, **kwargs)

    def ts_logp(self, ts_xs, ts_ys):
        """ Define the tf operators for logp """
        ts_p = tf.cast(tf.equal(self.ts_predict(ts_xs), ts_ys), dtype=tf_float)
        return tf.math.log(ts_p)

    # Some useful functions TODO
    def ts_kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def ts_fvp(self, ts_xs, ts_gs, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        raise NotImplementedError


class _RobustKerasPolicy(RobustKerasFuncApp, tfPolicy):
    pass  # for debugging


class _RobustKerasMLPPolicy(RobustKerasMLP, tfPolicy):
    pass  # for debugging


LOG_TWO_PI = tf.constant(np.log(2*np.pi), dtype=tf_float)
def gaussian_logp(xs, ms, lstds):
     # log probability of Gaussian with diagonal variance over batches xs
    axis= tf.range(1,tf.rank(xs))
    qs = tf.reduce_sum(-0.5*tf.square(xs-ms)/tf.exp(2.*lstds), axis=axis)
    logs = tf.reduce_sum(-lstds -0.5*LOG_TWO_PI,axis=axis)
    return qs + logs

def gaussian_kl(ms_1, lstds_1, ms_2, lstds_2):
    # KL(p1||p2)  support batches
    axis= tf.range(1,tf.rank(ms_1))
    vars_1, vars_2 = tf.exp(lstds_1*2.), tf.exp(lstds_2*2.)
    kls = lstds_2 - lstds_1 - 0.5
    kls += (vars_1 + tf.square(ms_1-ms_2)) / (2.0*vars_2)
    kls = tf.reduce_sum(kls, axis=axis)
    return kls


class tfGaussianPolicy(tfPolicy):
    """ A wrapper for augmenting tfFuncApp with Gaussian noises.
    """
    def __init__(self, x_shape, y_shape, name='tf_gaussian_policy',
                 init_lstd=None, min_std=0.0,  # new attribues
                 **kwargs):
        """ The user needs to provide init_lstd. """
        assert init_lstd is not None
        init_lstd = np.broadcast_to(init_lstd, y_shape)
        self._ts_lstd = tf.Variable(array_to_ts(init_lstd), dtype=tf_float)
        self._ts_min_lstd = tf.constant(np.log(min_std), dtype=tf_float)
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # some conveniet properties
    @property
    def ts_variables(self):
        return super().ts_variables + [self._ts_lstd]

    @ts_variables.setter
    def ts_variables(self, ts_vals):
        [var.assign(val) for var, val in zip(self.ts_variables, ts_vals)]


    def mean(self, xs):
        return self(xs, stochastic=False)

    def ts_mean(self, xs):
        return self.ts_predict(xs, stochastic=False)

    @property
    def lstd(self):
        return self.ts_lstd.numpy()

    @property
    def ts_lstd(self):
        return tf.maximum(self._ts_lstd, self._ts_min_lstd)

    # ts_predict, ts_logp, ts_fvp, ts_kl
    def ts_predict(self, ts_xs, stochastic=True, ts_noises=False, **kwargs):
        """ Define the tf operators for predict """
        ts_ms = super().ts_predict(ts_xs, **kwargs)
        if stochastic:
            if tf.equal(ts_noises, False):
                shape = [ts_xs.shape[0]]+list(self.y_shape)
                ts_noises = tf.random.normal(shape)
            return ts_ms + tf.exp(self.ts_lstd) * ts_noises
        else:
            return ts_ms

    def ts_logp(self, ts_xs, ts_ys):  # overwrite
        ts_ms = self.ts_mean(ts_xs)
        ts_lstds = tf.broadcast_to(self.ts_lstd, ts_ms.shape)
        return gaussian_logp(ts_ys, ts_ms, ts_lstds)

    def ts_kl(self, other, ts_xs, reversesd=False, p1_sg=False, p2_sg=False):
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

    def ts_fvp(self, ts_xs, ts_gs):
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


class RobustKerasMLPGassian(tfGaussianPolicy, RobustKerasMLP):

    def __init__(self, x_shape, y_shape, name='robust_k_MLP_gaussian_policy', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)

