import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from rl.tools.function_approximators.function_approximator import FunctionApproximator, assert_shapes


class KerasFuncApp(FunctionApproximator):
    """
        A wrapper of tf.keras.Model.

        It is a FunctionApproximator with an additional attribute `model`,
        which is a tf.keras.Model, so we can reuse existing functionality in
        tf.keras, such as batch prediction, etc.

        When inheriting this class, users can choose to implement the
        `_build_model` method, for ease of implementation. `build_model` can be
        used to create attributes which are not deepcopy compatible, such as
        tf.keras.Layer. These attributes will be built new when performing
        deepcopy, and then updated indirectly through self.model.

        Otherwise, a tf.keras.Model or a method, which shares the same
        signature of `_build_model`, can be passed in __init__ .

    """
    def __init__(self, x_shape, y_shape, name='tf2_func_approx', seed=None,
                 build_model=None): # a keras.Model or a method that shares the signature of `_build_model`
        super().__init__(x_shape, y_shape, name, seed)
        # decide how to build the model
        if isinstance(build_model, tf.keras.Model):
            _model = build_model
            build_model = lambda *args, **kwargs: _model
        assert build_model is None or callable(build_model)
        self.build_model = build_model
        # build the overall tf.keras.Model
        self.build()

    def _build_model(self, x_shape, y_shape, seed=None):
        """ Build the default model.

            Users are free to create additional attributes, which are
            tf.keras.Model, tf.keras.Layer, tf.Variable, etc., to help
            construct the overall function approximator. At the end, the
            function should output a tf.keras.Model, which is the overall
            function approximator.
        """
        raise NotImplementedError

    def build(self):
        """ Build an initialized tf.keras.Model as the overall function
            approximator.
        """
        build_model = self.build_model or self._build_model
        self.model = build_model(self.x_shape, self.y_shape, seed=self.seed)
        # force to the model to initialize
        if isinstance(self.x_shape, list):
            ts_x = [tf.keras.Input(shape) for shape in self.x_shape]
        else:
            ts_x = tf.keras.Input(self.x_shape)
        self.model(ts_x)

    def predict(self, xs, **kwargs):
        # kwargs contains parameters for tf.keras.Model.predict.
        return self.model.predict(xs, **kwargs)

    @property
    def variables(self):
        return [var.numpy() for var in self.model.trainable_variables]

    @variables.setter
    def variables(self, vals):
        return [var.assign(val) for var, val in zip(self.model.trainable_variables,vals)]

    # utilities
    def assign(self, other):
        assert_shapes(self.x_shape, other.x_shape)
        assert_shapes(self.y_shape, other.y_shape)
        self.seed = other.seed
        self.model.set_weights(other.model.get_weights())

    def save(self, path):
        self.model.save_weights(path)

    def restore(self, path):
        self.model.load_weights(path)

    def __deepcopy__(self, memo):
        """ Make tf.keras.Model support deepcopy. """
        cls = type(self)  # create a new instance
        new = cls.__new__(cls)
        memo[id(self)] = new # prevent forming a loop
        for k, v in self.__dict__.items():
            try:
                setattr(new, k, copy.deepcopy(v, memo))
            except:
                pass
        new.build()  # this should work
        new.assign(self)
        return new

