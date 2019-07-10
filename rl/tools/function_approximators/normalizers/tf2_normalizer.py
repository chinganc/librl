import tensorflow as tf
import numpy as np
import copy
from abc import ABC, abstractmethod

from rl.tools.function_approximators.normalizers.normalizer import Normalizer, NormalizerStd, NormalizerMax
from rl.tools.utils.tf_utils import tf_float


def _tf2NormalizerDecorator(cls):
    """ A decorator for adding a tf operator equivalent of Normalizer.predict

        It reuses all the functionalties of the original Normalizer and
        additional tf.Variables for defining the tf operator.
    """
    assert issubclass(cls, Normalizer)

    class decorated_cls(cls):

        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)
            # add additional tf.Variables
            self.ts_bias = tf.Variable(self.bias, dtype=tf_float)
            self.ts_scale = tf.Variable(self.scale, dtype=tf_float)
            self.ts_unscale = tf.Variable(self.unscale, dtype=tf.bool)
            self.ts_unbias = tf.Variable(self.unbias, dtype=tf.bool)
            self._ts_initialized = tf.Variable(self._initialized, dtype=tf.bool)
            self._ts_clip = tf.Variable(self.thre is not None)
            if self._ts_clip:
                self.ts_thre = tf.Variable(self.thre, dtype=tf_float)
            else:
                self.ts_thre = tf.Variable((0.,0.), dtype=tf_float)

        # make sure the tf.Variables are synchronized
        def update(self, x):
            super().update(x)
            self._update_tf_vars()

        def reset(self, x):
            super().reset(x)
            self._update_tf_vars()

        def assign(self, other):
            super().assign(other)
            self._update_tf_vars()

        def save(self, path):
            super().save(path)

        def restore(self, path):
            super().restore(path)
            self._update_tf_vars()

        def _update_tf_vars(self):
            # synchronize the tf.Variables
            self.ts_bias.assign(self.bias)
            self.ts_scale.assign(self.scale)
            self.ts_unscale.assign(self.unscale)
            self.ts_unbias.assign(self.unbias)
            self._ts_initialized.assign(self._initialized)
            self._ts_clip.assign(self.thre is not None)
            if self._ts_clip:
                self.ts_thre.assign(self.thre)

        # add convenient interface
        # @tf.function
        def ts_predict(self, ts_x):
            """ A tf operator that mimics the behavior of `predict`. """
            ts_x = tf.cast(ts_x, tf_float)

            if tf.logical_not(self._ts_initialized):
                return ts_x

            # do something
            if tf.logical_not(self._ts_clip):
                if tf.logical_not(self.ts_unbias):
                    ts_x = ts_x - self.ts_bias
                if tf.logical_not(self.ts_unscale):
                    ts_x = ts_x / self.ts_scale
            else:
                # need to first scale it before clipping
                ts_x = (ts_x - self.ts_bias) / self.ts_scale
                ts_x = tf.clip_by_value(ts_x, self.ts_thre[0], self.ts_thre[1])
                # check if we need to scale it back
                if self.ts_unscale:
                    ts_x = ts_x * self.ts_scale
                    if self.ts_unbias:
                        ts_x = ts_x + self.ts_bias
                else:
                    if self.ts_unbias:
                        ts_x = ts_x + self.ts_bias / self.ts_scale
            return ts_x

        def ts_normalize(self, ts_x):  # alias
            return self.ts_predict(ts_x)

    # make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tf2NormalizerDecorator
class tf2NormalizerStd(NormalizerStd):
    pass

@_tf2NormalizerDecorator
class tf2NormalizerMax(NormalizerMax):
    pass
