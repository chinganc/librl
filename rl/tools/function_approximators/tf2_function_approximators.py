import copy
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.keras import layers
from rl.tools.function_approximators.function_approximator import FunctionApproximator, assert_shapes
from rl.tools.function_approximators.normalizers import tf2NormalizerMax


class tf2FuncApp(FunctionApproximator):
    """ A minimal wrapper for tensorflow 2 operators.

        The user needs to define `ts_predict`and `ts_variables`.

        (Everything else should work out of the box, because of tensorflow 2.)
    """
    def __init__(self, x_shape, y_shape, name='tf2_func_app', seed=None):
        super().__init__(x_shape, y_shape, name=name, seed=seed)
        self(np.zeros(x_shape))  # make sure everything is initialized

    def predict(self, xs, **kwargs):
        return self.ts_predict(xs, **kwargs).numpy()

    @property
    def variables(self):
        return [var.numpy() for var in self.ts_variables]

    @variables.setter
    def variables(self, vals):
        return [var.assign(val) for var, val in zip(self.ts_variables,vals) ]

    @abstractmethod
    def ts_predict(self, ts_xs, **kwargs):
        """ Define the tf operators for predict """

    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """


class tf2RobustFuncApp(tf2FuncApp):
    """ A function approximator with input and output normalizers.

        The user needs to define `_ts_predict`and `ts_variables`.
    """

    def __init__(self, x_shape, y_shape, name='tf2_robust_func_app', seed=None,
                 build_x_nor=None, build_y_nor=None):

        build_x_nor = build_x_nor or (lambda : tf2NormalizerMax(x_shape, unscale=False, \
                                        unbias=False, clip_thre=5.0, rate=0., momentum=None))
        build_y_nor = build_y_nor or (lambda: tf2NormalizerMax(y_shape, unscale=True, \
                                        unbias=True, clip_thre=5.0, rate=0., momentum=None))
        self._x_nor = build_x_nor()
        self._y_nor = build_y_nor()
        super().__init__(x_shape, y_shape, name=name, seed=seed)

    def ts_predict(self, ts_xs, clip=True, **kwargs):
        ts_xs = self._x_nor.ts_predict(ts_xs)
        ts_ys = self._ts_predict(ts_xs, **kwargs)
        if clip:
            return self._x_nor.ts_predict(ts_ys)
        else:
            return ts_ys

    def update(self, xs, ys=None):
        self._x_nor.update(xs)
        if ys is not None:
            self._y_nor.update(ys)

    @abstractmethod
    def _ts_predict(self, ts_xs, **kwargs):
        """ define the tf operators for predict """

    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """


class tf2RobustMLP(tf2RobustFuncApp):

    def __init__(self, x_shape, y_shape, units=(), activation='tanh', **kwargs):

        self._kmodel = tf.keras.Sequential()
        for unit in units:
            self._kmodel.add(tf.keras.layers.Dense(unit, activation=activation))
        self._kmodel.add(tf.keras.layers.Dense(y_shape[0], activation='linear'))
        super().__init__(x_shape, y_shape, **kwargs)

    def _ts_predict(self, ts_xs):
        return self._kmodel(ts_xs)

    @property
    def ts_variables(self):
        return self._kmodel.trainable_variables

