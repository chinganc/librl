import numpy as np
import copy
import pickle

from rl.tools.function_approximators.function_approximator import FunctionApproximator


from rl.tools.utils.mvavg import ExpMvAvg, PolMvAvg
from rl.tools.utils.misc_utils import deepcopy_from_list


class Normalizer(FunctionApproximator):
    """ A normalizer that adapts to streaming observations. 
        
        Given input x, it computes x_cooked = clip((x-bias)/scale, thre)

        By default, Normalizer is just an identity map.  The user needs to
        implement `update` to create desired behaviors.
    """

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None):
        """ It overloads the signature of FunctionApproximator, since the input
            and output are always in the same shape and there is no randomness.

            `clip_thre` can be a non-negative float/nd.array; in this case, the
            thresholds are -clip_thre and clip_thre. Alternatively, `clip_thre`
            can be a list/tuple of two non-negative float/nd.arrays, which directly
            specifies the upper and lower bounds.  
        """
        super().__init__(shape, shape, name='online_normalizer')
        # new attributes
        self.bias = np.zeros(shape)
        self.scale = np.ones(shape)
        self.unscale = unscale
        self.unbias = unbias    
        if isinstance(clip_thre, list) or isinstance(clip_thre, tuple):
            assert len(clip_thre)==2
        else:
            if clip_thre is not None: 
                assert np.all(clip_thre>=0)
                clip_thre = (-clip_thre, clip_thre)                      
        self.clip_thre = clip_thre

    def predict(self, x):
        """ Normalization in batches
        
            Given input x, it computes
                x_cooked = clip((x-bias)/scale, thre)
            If unscale/unbias is True, it removes the scaling/bias after clipping.
        """
        # do something
        if self.clip_thre is None:
            if not self.unbias:
                x = x - self.bias
            if not self.unscale:
                x = x / self.scale
        else:
            # need to first scale it before clipping
            x = (x - self.bias) / self.scale
            x = np.clip(x, self.clip_thre[0], self.clip_thre[1])
            # check if we need to scale it back
            if self.unscale:
                x = x * self.scale
                if self.unbias:
                    x = x + self.bias
            else:
                if self.unbias:
                    x = x + self.bias / self.scale
        return x

    @property
    def variables(self):
        return []

    @variables.setter
    def variables(self, vals):
        pass

    def update(self, *args, **kwargs):
        """ Update bias, scale, clip_thre """

    def reset(self):
        """ Reset bias and scale """
        self.bias = np.zeros(shape)
        self.scale = np.ones(shape)

    def assign(self, other):
        assert type(self) == type(other)
        deepcopy_from_list(self, other, self.__dict__.keys())

    def save(self, path):
        pickle.dump(self, path)

    def restore(self, path):
        saved = pickle.load(path)
        self.assign(saved)


class NormalizerStd(OnlineNormalizer):

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6):
        """
            An online normalizer based on whitening.

            shape: None or an tuple specifying each dimension
            momentum: None for moving average
                      [0,1) for expoential average
                      1 for using instant update
            rate: decides the weight of new observation as itr**rate
        """
        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre)
        if momentum is None:
            self._mvavg_init = lambda: PolMvAvg(np.zeros(self._shape), power=rate)
        else:
            assert momentum <= 1.0 and momentum >= 0.0
            self._mvavg_init = lambda: ExpMvAvg(np.zeros(self._shape), rate=momentum)

        # new attributes
        self._mean = self._mvavg_init()
        self._mean_of_sq = self._mvavg_init()
        self._eps = eps
    
    def reset(self):
        super().reset()
        self._mean = self._mvavg_init()
        self._mean_of_sq = self._mvavg_init()

    def update(self, x):
        if x.shape == self.x_shape:
            x = x[None,:]
        # observed stats
        new_mean = np.mean(x, axis=0)
        new_mean_of_sq = np.mean(np.square(x), axis=0)
        self._mean.update(new_mean)
        self._mean_of_sq.update(new_mean_of_sq)

        self.bias = self._mean.val
        variance = self._mean_of_sq.val - np.square(self._mean.val)
        std = np.sqrt(variance)

        self.scale = np.maximum(std, self._eps)



class NormalizerMax(OnlineNormalizer):

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6):
        # Args:
        #   momentum: None for moving average
        #             [0,1) for expoential average
        #             1 for using instant update
        #   rate: decide the weight of new observation as itr**rate
        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre)
        self._norstd = NormalizerStd(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre,
                                     rate=rate, momentum=momentum, eps=eps)
        self.reset()
        self._eps = eps

    def _reset(self):
        self._norstd.reset()
        self._upper_bound = None
        self._lower_bound = None

    @property
    def bias(self):
        return 0.5 * self._upper_bound + 0.5 * self._lower_bound

    @property
    def scale(self):
        return np.maximum(self._upper_bound - self.bias, self._eps)

    def _update(self, x):
        # update stats
        self._norstd.update(x)
        # update clipping
        scale_candidate = self._norstd.std
        upper_bound_candidate = self._norstd.bias + self._norstd.scale
        lower_bound_candidate = self._norstd.bias - self._norstd.scale

        if not self._initialized:
            self._upper_bound = upper_bound_candidate
            self._lower_bound = lower_bound_candidate
        else:
            self._upper_bound = np.maximum(self._upper_bound, upper_bound_candidate)
            self._lower_bound = np.minimum(self._lower_bound, lower_bound_candidate)
