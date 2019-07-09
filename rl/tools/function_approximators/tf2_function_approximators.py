import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from rl.tools.function_approximators.function_approximator import FunctionApproximator, online_compatible, assert_shapes


class tf2FunctionApproximator(FunctionApproximator):
    """
        A wrapper of tf.keras.Model

        The user should implement _build_model
    """
    def __init__(self, x_shapes, y_shapes, name='tf2_func_approx', seed=None,
                 build_model=None,  # ah-hoc keras.Model
                 build_input_nor=None,  # input normalizer
                 build_output_nor=None):  # output normalizer

        x = tf.keras.Input(shape=x_shape)

        build_input_nor = build_input_nor or tfNormalizerMax
        self._in_nor = build_input_nor(x_shapes)

        build_model = build_model or self._build_model
        self.model =  build_model(x_shapes, y_shapes, seed)

        build_output_nor = build_output_nor or tfNormalizerMax
        self._out_nor = build_output_nor(y_shapes)


        super().__init__(x_shapes, y_shapes, name, seed)

    def _build_model(self, x_shapes, y_shapes, seed=None):
        """ Default model
        
            return a tf.keras.Model 
        """
        raise NotImplementedError

    def predict(self, xs, **kwargs):
        # kwargs contains parameters for tf.keras.Model.predict.
        xs = self._in_nor.predict(xs)
        return model.predict(xs, **kwargs)

    @property
    def variables(self):
        return self.model.get_weights()

    @variables.setter
    def variables(self, vals):
        self.model.set_weights(vals)

    # utilities
    def assign(self, other):
        assert_shapes(self.x_shapes, other.x_shapes)
        assert_shapes(self.y_shapes, other.y_shapes)
        self.seed = other.seed
        self.variables = other.variables

    def save(self, path):
        self.model.save(path)

    def restore(self, path):
        self.model = tf.keras.models.load_model(path)

    def __deepcopy__(self, memo):
        """ Overloaded because tf.keras.Model cannot be deepcopy """
        # deepcopy everything except for model
        cls = type(self)  # create a new instance
        new = cls.__new__(cls)
        memo[id(self)] = new # prevent forming a loop
        for k, v in self.__dict__.items():
            if k != 'model':
                setattr(new, k, copy.deepcopy(v, memo))
        new.model = tf.keras.models.clone_model(self.model)
        return new


if __name__=='__main__':


    def build_model1(x_shape, y_shape, seed=None):
        # function approximator based on tf.keras.Model
        model = tf.keras.Sequential()
        # Adds a densely-connected layer with 64 units to the model:
        model.add(layers.Dense(64, activation='relu'))
        # Add another:
        model.add(layers.Dense(64, activation='relu'))
        # Add a softmax layer with 10 output units:
        model.add(layers.Dense(y_shape, activation='softmax'))
        x = np.zeros(x_shape)
        model.predict(x)  # create the weights
        return model


    def build_model2(x_shape, y=_shape, seed=None):
        x = tf.keras.Input(shape=x_shape)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        y = layers.Dense(y_shape, activation='softmax')(x)
        model = tf.keras.Model(inputs=x, outputs=y)
        return model

    def test_fun(fun):
        new_fun = copy.deepcopy(fun)
        new_fun.variable = fun.variable
        new_fun.variables = fun.variables

        x = np.random.random(fun.x_shape)
        fun.predict(x)

        x = np.random.random([10,]+list(fun.x_shape))
        fun.predict(x)

    x_shapes = 10
    y_shapes = 3
    fun1 = tf2FunctionApproximator(x_shapes, y_shapes, build_model=build_model1)
    #fun2 = tf2FunctionApproximator(x_shapes, y_shapes, build_model=build_model2)



