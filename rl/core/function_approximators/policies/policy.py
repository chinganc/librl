from abc import abstractmethod
import numpy as np
from rl.core.function_approximators.function_approximator import FunctionApproximator, online_compatible


class Policy(FunctionApproximator):
    """ An abstract interface that represents conditional distribution \pi(a|s).

        A policy is namely a stochastic FunctionApproximator.
    """

    def __init__(self, x_shape, y_shape, name='policy', **kwargs):
        super().__init__(x_dim, y_dim, name=name, **kwargs)

    def predict(self, xs, stochastic=True, **kwargs):
        """ Predict the values on batches of xs. """
        return super().predict(ts_xs, stochastic=stochastic, **kwargs)

    @online_compatible
    def logp(self, xs, ys, **kwargs):
        """ Compute the log probabilities on batches of (xs, ys)."""
        return np.log(self.predict(xs, **kwargs)==ys)  # default behavior

    # Some useful functions
    def kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def fvp(self, xs, gs, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        raise NotImplementedError
