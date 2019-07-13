import copy
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from rl.tools.function_approximators.function_approximator import FunctionApproximator
from rl.tools.function_approximators.normalizers import tf2NormalizerMax
from rl.tools.utils.tf_utils import tf_float


class tf2FuncApp(FunctionApproximator):
    """ A minimal wrapper for tensorflow 2 operators.

        The user needs to define `ts_predict`and `ts_variables`.

        (Everything else should work out of the box, because of tensorflow 2.)
    """
    def __init__(self, x_shape, y_shape, name='tf2_func_app', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, **kwargs):
        return self.ts_predict(tf.constant(xs, dtype=tf_float), **kwargs).numpy()

    @property
    def variables(self):
        return [var.numpy() for var in self.ts_variables]

    @variables.setter
    def variables(self, vals):
        return [var.assign(val) for var, val in zip(self.ts_variables,vals) ]

    # required implementation
    @abstractmethod
    def ts_predict(self, ts_xs, **kwargs):
        """ Define the tf operators for predict """

    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """


class KerasFuncApp(tf2FuncApp):
    """
        A wrapper of tf.keras.Model.

        It is a FunctionApproximator with an additional attribute `kmodel`,
        which is a tf.keras.Model, so we can reuse existing functionality in
        tf.keras, such as batch prediction, etc.

        When inheriting this class, users can choose to implement the
        `_build_kmodel` method, for ease of implementation. `build_kmodel` can be
        used to create necessary tf.keras.Layer or tf.Tensor to help defining
        the kmodel. Note all attributes created, if any, should be deepcopy
        compatible.

        Otherwise, a tf.keras.Model or a method, which shares the same
        signature of `_build_kmodel`, can be passed in __init__ .

    """
    def __init__(self, x_shape, y_shape, name='keras_func_app',
                 build_kmodel=None, **kwargs): # a keras.Model or a method that shares the signature of `_build_kmodel`
        super().__init__(x_shape, y_shape, name=name, **kwargs)
        # decide how to build the kmodel
        """ Build an initialized tf.keras.Model as the overall function
            approximator.
        """
        if isinstance(build_kmodel, tf.keras.Model):
            self.kmodel = build_kmodel
        else:
            build_kmodel = build_kmodel or self._build_kmodel
            self.kmodel = build_kmodel(self.x_shape, self.y_shape)
        # make sure the model is constructed
        ts_x = tf.zeros([1]+list(self.x_shape))
        self.ts_predict(ts_x)

    def _build_kmodel(self, x_shape, y_shape):
        """ Build the default kmodel.

            Users are free to create additional attributes, which are
            tf.keras.Model, tf.keras.Layer, tf.Variable, etc., to help
            construct the overall function approximator. At the end, the
            function should output a tf.keras.Model, which is the overall
            function approximator.
        """
        raise NotImplementedError

    # required methods of tf2FuncApp
    def ts_predict(self, ts_xs):
        return self.kmodel(ts_xs)

    @property
    def ts_variables(self):
        return self.kmodel.trainable_variables

    # utilities
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['kmodel']
        d['kmodel_config'] = self.kmodel.get_config()
        d['kmodel_weights'] = self.kmodel.get_weights()
        return d

    def __setstate__(self, d):
        d = dict(d)
        weights = d['kmodel_weights']
        config = d['kmodel_config']
        del d['kmodel_weights']
        del d['kmodel_config']
        self.__dict__.update(d)
        try:
            self.kmodel = tf.keras.Model.from_config(config)
        except KeyError:
            self.kmodel = tf.keras.Sequential.from_config(config)
        # intialize the weights (keras bug...)
        ts_x = tf.zeros([1]+list(self.x_shape))
        self.ts_predict(ts_x)
        self.kmodel.set_weights(weights)


class tf2RobustFuncApp(tf2FuncApp):
    """ A function approximator with input and output normalizers.

        This class can be viewed as a wrapper in inheritance.  For example, for
        any subclass `A` of `tf2FuncApp`, we can create a robust subclass `B` by
        simply defining

            class B(tf2RobustFuncApp, A):
                pass
    """

    def __init__(self, x_shape, y_shape, name='tf2_robust_func_app',
                 build_x_nor=None, build_y_nor=None, **kwargs):

        build_x_nor = build_x_nor or (lambda : tf2NormalizerMax(x_shape, unscale=False, \
                                        unbias=False, clip_thre=5.0, rate=0., momentum=None))
        build_y_nor = build_y_nor or (lambda: tf2NormalizerMax(y_shape, unscale=True, \
                                        unbias=True, clip_thre=5.0, rate=0., momentum=None))
        self._x_nor = build_x_nor()
        self._y_nor = build_y_nor()
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def ts_predict(self, ts_xs, clip_y=tf.constant(True)):
        # include also input and output normalizeations
        ts_xs = self._x_nor.ts_predict(ts_xs)
        ts_ys = super().ts_predict(ts_xs)
        if clip_y:
            return self._y_nor.ts_predict(ts_ys)
        else:
            return ts_ys

    def update(self, xs=None, ys=None, *args, **kwargs):
        print('Update normalizers of {}'.format(self.name))
        if xs is not None:
            self._x_nor.update(xs)
        if ys is not None:
            self._y_nor.update(ys)
        return super().update(xs=xs, ys=ys, *args, **kwargs)


class KerasRobustFuncApp(tf2RobustFuncApp, KerasFuncApp):

    def __init__(self, x_shape, y_shape, name='k_robust_func_app', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)


class KerasRobustMLP(KerasRobustFuncApp):

    def __init__(self, x_shape, y_shape, name='k_robust_mlp', units=(),
                 activation='tanh', **kwargs):
        self.units, self.activation = units, activation
        super().__init__(x_shape, y_shape, **kwargs)

    def _build_kmodel(self, x_shape, y_shape):
        kmodel = tf.keras.Sequential()
        for unit in self.units:
            kmodel.add(tf.keras.layers.Dense(unit, activation=self.activation))
        kmodel.add(tf.keras.layers.Dense(y_shape[-1], activation='linear'))
        return kmodel

