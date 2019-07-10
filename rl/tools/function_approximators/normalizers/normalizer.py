import numpy as np
import copy
import pickle
import os

from rl.tools.function_approximators.function_approximator import FunctionApproximator
from rl.tools.utils.mvavg import ExpMvAvg, PolMvAvg
from rl.tools.utils.misc_utils import deepcopy_from_list


class Normalizer(FunctionApproximator):
    """ A normalizer that adapts to streaming observations.

        Given input x, it computes x_cooked = clip((x-bias)/scale, thre)

        By default, Normalizer is just an identity map.  The user needs to
        implement `update` (and `reset`) to create the desired behaviors.
    """

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None, name='normalizer'):
        """ It overloads the signature of FunctionApproximator, since the input
            and output are always in the same shape and there is no randomness.

            `clip_thre` can be a non-negative float/nd.array; in this case, the
            thresholds are -clip_thre and clip_thre. Alternatively, `clip_thre`
            can be a list/tuple of two non-negative float/nd.arrays, which directly
            specifies the upper and lower bounds.
        """
        super().__init__(shape, shape, name=name)
        # new attributes
        self.bias = np.zeros(shape)
        self.scale = np.ones(shape)
        self.unscale = unscale
        self.unbias = unbias
        if isinstance(clip_thre, list) or isinstance(clip_thre, tuple):
            assert len(clip_thre)==2
            clip_thre = np.array(clip_thre)
        else:
            if clip_thre is not None:
                assert np.all(clip_thre>=0)
                clip_thre = np.array((-clip_thre, clip_thre))
        self.thre = clip_thre
        self._initialized = False

    def predict(self, x):
        """ Normalization in batches

            Given input x, it computes
                x_cooked = clip((x-bias)/scale, thre)
            If unscale/unbias is True, it removes the scaling/bias after clipping.
        """
        if not self._initialized:
            return x

        # do something
        if self.thre is None:
            if not self.unbias:
                x = x - self.bias
            if not self.unscale:
                x = x / self.scale
        else:
            # need to first scale it before clipping
            x = (x - self.bias) / self.scale
            x = np.clip(x, self.thre[0], self.thre[1])
            # check if we need to scale it back
            if self.unscale:
                x = x * self.scale
                if self.unbias:
                    x = x + self.bias
            else:
                if self.unbias:
                    x = x + self.bias / self.scale
        return x

    def normalize(self, x):  # acronym
        return self.predict(x)

    @property
    def variables(self):
        return []

    @variables.setter
    def variables(self, vals):
        pass

    def update(self, *args, **kwargs):
        """ Update bias and scale"""
        self._initialized = True

    def reset(self):
        """ Reset bias and scale """
        self.bias = np.zeros(self.x_shape)
        self.scale = np.ones(self.x_shape)
        self._initialized = False

    def assign(self, other):
        assert isinstance(self, type(other))
        deepcopy_from_list(self, other, self.__dict__.keys())

    def save(self, path):
        path = os.path.join(path, self.name)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def restore(self, path):
        with open(path, 'rb') as pickle_file:
            saved = pickle.load(pickle_file)
        self.assign(saved)


class NormalizerStd(Normalizer):
    """ An online normalizer based on whitening. """

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6, name='normalizer_std'):
        """
            Args:
                shape: None or an tuple specifying each dimension
                momentum: None for moving average
                          [0,1) for expoential average
                          1 for using instant update
                rate: decides the weight of new observation as itr**rate
        """
        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre, name=name)
        if momentum is None:
            self._mvavg_init = lambda: PolMvAvg(np.zeros(shape), power=rate)
        else:
            assert momentum <= 1.0 and momentum >= 0.0
            self._mvavg_init = lambda: ExpMvAvg(np.zeros(shape), rate=momentum)
        # new attributes
        self._mean = self._mvavg_init()
        self._mean_of_sq = self._mvavg_init()
        self._eps = eps

    def update(self, x):
        # x can be an intance of a batch
        if x.shape == self.x_shape:
            x = x[None,:]
        assert len(x.shape)>1
        assert x[0,:].shape == self.x_shape
        # observed stats
        new_mean = np.mean(x, axis=0)
        new_mean_of_sq = np.mean(np.square(x), axis=0)
        weight = x.shape[0]
        self._mean.update(new_mean, weight=weight)
        self._mean_of_sq.update(new_mean_of_sq, weight=weight)
        # update bias and scale
        self.bias = self._mean.val
        variance = self._mean_of_sq.val - np.square(self._mean.val)
        std = np.sqrt(np.maximum(variance, np.zeros_like(variance)))
        self.scale = np.maximum(std, self._eps)
        super().update()

    def reset(self):
        super().reset()
        self._mean = self._mvavg_init()
        self._mean_of_sq = self._mvavg_init()


class NormalizerMax(Normalizer):
    """ An online normalizer based on'non-shrinking' upper and lower bounds.

        It uses a NormalizerStd to give an instantaneous estimate of upper and
        lower bounds. It then compares that to the current bounds, and accept
        the new bounds only if that expands the current ones. The shifting bias
        is centered at the centered with respect to the upper and lower bounds.
    """


    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6, name='normalizer_max'):
        """ The signature requires to create NormalizerStd.

            Args:
                shape: None or an tuple specifying each dimension
                momentum: None for moving average
                          [0,1) for expoential average
                          1 for using instant update
                rate: decides the weight of new observation as itr**rate
         """
        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre, name=name)
        # new attributes
        self._norstd = NormalizerStd(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre,
                                     rate=rate, momentum=momentum, eps=eps)
        self._upper_bound = None
        self._lower_bound = None
        self._eps = eps

    def update(self, x):
        # update stats
        self._norstd.update(x)
        # update clipping
        upper_bound_candidate = self._norstd.bias + self._norstd.scale
        lower_bound_candidate = self._norstd.bias - self._norstd.scale

        if not self._initialized:
            self._upper_bound = upper_bound_candidate
            self._lower_bound = lower_bound_candidate
        else:
            self._upper_bound = np.maximum(self._upper_bound, upper_bound_candidate)
            self._lower_bound = np.minimum(self._lower_bound, lower_bound_candidate)

        self.bias = 0.5*self._upper_bound + 0.5*self._lower_bound
        self.scale = np.maximum(self._upper_bound-self.bias, self._eps)
        super().update()

    def reset(self):
        super().reset()
        self._norstd.reset()
        self._upper_bound = None
        self._lower_bound = None
