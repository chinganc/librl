import copy
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from rl.core.function_approximators.function_approximator import FunctionApproximator
from rl.core.function_approximators.normalizers import tfNormalizerMax
from rl.core.utils.tf2_utils import array_to_ts
from rl.core.utils.misc_utils import flatten, unflatten

# NOTE ts_* methods are in batch mode
#      ts_variables is a list of tf.Variables

class tfFuncApp(FunctionApproximator):
    """ A minimal wrapper for tensorflow 2 operators.

        The user needs to define `ts_predict`and `ts_variables`.

        (Everything else should work out of the box, because of tensorflow 2.)
    """
    def __init__(self, x_shape, y_shape, name='tf_func_app', **kwargs):
        self._var_shapes = None  # cache
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, **kwargs):
        return self.ts_predict(array_to_ts(xs), **kwargs).numpy()

    @property
    def variable(self):
        return flatten([var.numpy() for var in self.ts_variables])

    @variable.setter
    def variable(self, val):
        if self._var_shapes is None:
            self._var_shapes = [var.shape.as_list() for var in self.ts_variables]
        vals = unflatten(val, shapes=self._var_shapes)
        [var.assign(val) for var, val in zip(self.ts_variables, vals)]

    # required implementation
    @abstractmethod
    def ts_predict(self, ts_xs, **kwargs):
        """ Define the tf operators for predict """

    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """


class KerasFuncApp(tfFuncApp):
    """
        A wrapper of tf.keras.Model.

        It is a FunctionApproximator with an additional attribute `kmodel`,
        which is a tf.keras.Model.

        It adds a new method `k_predict` which calls tf.keras.Model.preidct.

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

    def k_predict(self, xs, **kwargs):
        return self.kmodel.predict(xs, **kwargs)

    # required methods of tfFuncApp
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


class tfRobustFuncApp(tfFuncApp):
    """ A function approximator with input and output normalizers.

        This class can be viewed as a wrapper in inheritance.  For example, for
        any subclass `A` of `tfFuncApp`, we can create a robust subclass `B` by
        simply defining

            class B(tfRobustFuncApp, A):
                pass
    """

    def __init__(self, x_shape, y_shape, name='tf_robust_func_app',
                 x_nor=None, y_nor=None, **kwargs):

        self._x_nor = x_nor or tfNormalizerMax(x_shape, unscale=False, \
                                    unbias=False, clip_thre=5.0, rate=0., momentum=None)
        self._y_nor = y_nor or tfNormalizerMax(y_shape, unscale=True, \
                                    unbias=True, clip_thre=5.0, rate=0., momentum=None)
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, clip_y=True, **kwargs):
        return super().predict(xs, clip_y=clip_y, **kwargs)

    def ts_predict(self, ts_xs, clip_y=True):
        # include also input and output normalizeations
        ts_xs = self._x_nor.ts_predict(ts_xs)
        ts_ys = super().ts_predict(ts_xs)
        return self._y_nor.ts_predict(ts_ys) if clip_y else ts_ys

    def update(self, xs=None, ys=None, *args, **kwargs):
        print('Update normalizers of {}'.format(self.name))
        if xs is not None:
            self._x_nor.update(xs)
        if ys is not None:
            self._y_nor.update(ys)
        return super().update(xs=xs, ys=ys, *args, **kwargs)


class RobustKerasFuncApp(tfRobustFuncApp, KerasFuncApp):

    def __init__(self, x_shape, y_shape, name='robust_k_func_app', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def k_predict(self, xs, clip_y=True, **kwargs):  # take care of this new method
        xs = self._x_nor(xs)
        ys = super().k_predict(xs)
        return self._y_nor(ts_ys) if clip_y else ys


# Some examples
class RobustKerasMLP(RobustKerasFuncApp):

    def __init__(self, x_shape, y_shape, name='robust_k_mlp', units=(),
                 activation='tanh', **kwargs):
        self.units, self.activation = units, activation
        super().__init__(x_shape, y_shape, **kwargs)

    def _build_kmodel(self, x_shape, y_shape):
        kmodel = tf.keras.Sequential()
        for unit in self.units:
            kmodel.add(tf.keras.layers.Dense(unit, activation=self.activation))
        kmodel.add(tf.keras.layers.Dense(y_shape[-1], activation='linear'))
        return kmodel

