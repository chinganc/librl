import unittest
import copy
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from rl.tools.function_approximators.tf2_function_approximators import KerasFuncApp


def build_model1(x_shape, y_shape, seed=None):
    # function approximator based on tf.keras.Model
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(y_shape[0], activation='softmax'))

    return model

def build_model2(x_shape, y_shape, seed=None):
    inputs = tf.keras.Input(shape=x_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    y = layers.Dense(y_shape[0], activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model


class TestKerasFuncApp(unittest.TestCase):

    def test_build_model(self):
        x_shape = (10,2,3)
        y_shape = (3,)
        fun1 = KerasFuncApp(x_shape, y_shape, build_model=build_model1)
        fun2 = KerasFuncApp(x_shape, y_shape, build_model=build_model2)
        func3 = KerasFuncApp(x_shape, y_shape, build_model=build_model1(x_shape, y_shape))

    def test_copy(self):
        x_shape = (10,2,3)
        y_shape = (3,)
        fun = KerasFuncApp(x_shape, y_shape, build_model=build_model1)

        new_fun = copy.deepcopy(fun)
        new_fun.variables = fun.variables
        new_fun.variable = fun.variable+1
        assert all([np.all(np.isclose(v1-v2,1.0)) for v1, v2 in zip(new_fun.variable,fun.variable)])

    def test_predict(self):
        x_shape = (10,2,3)
        y_shape = (3,)
        fun = KerasFuncApp(x_shape, y_shape, build_model=build_model1)

        x = np.random.random(fun.x_shape)
        fun(x)
        xs = np.random.random([10,]+list(fun.x_shape))
        fun.predict(xs)

    def test_save_and_restore(self):
        x_shape = (10,2,3)
        y_shape = (3,)
        fun1 = KerasFuncApp(x_shape, y_shape, build_model=build_model1)
        fun2 = KerasFuncApp(x_shape, y_shape, build_model=build_model1)
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            fun1.save(path)
            fun2.restore(path)
            assert all([np.all(np.isclose(v1-v2,0.0)) for v1, v2 in zip(fun1.variable,fun2.variable)])


if __name__ == '__main__':
    unittest.main()
