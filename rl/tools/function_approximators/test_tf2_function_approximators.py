import unittest
import copy
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from rl.tools.function_approximators.tf2_function_approximators import KerasFuncApp, RobustKerasFuncApp

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-8))


def build_model1(x_shape, y_shape, seed=None):
    # function approximator based on tf.keras.Model
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(y_shape[0]))

    return model

def build_model2(x_shape, y_shape, seed=None):
    inputs = tf.keras.Input(shape=x_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    y = layers.Dense(y_shape[0])(x)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model

def test_build_model(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun1 = cls(x_shape, y_shape, build_model=build_model1)
    fun2 = cls(x_shape, y_shape, build_model=build_model2)
    func3 = cls(x_shape, y_shape, build_model=build_model1(x_shape, y_shape))

def test_copy(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape, build_model=build_model1)

    new_fun = copy.deepcopy(fun)
    new_fun.variables = fun.variables
    new_fun.variable = fun.variable+1
    assert all([np.all(np.isclose(v1-v2,1.0)) for v1, v2 in zip(new_fun.variable,fun.variable)])

def test_predict(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape, build_model=build_model1)

    x = np.random.random(fun.x_shape)
    fun(x)
    xs = np.random.random([10,]+list(fun.x_shape))
    fun.predict(xs)

def test_save_and_restore(cls):
    x_shape = (10,2,3)
    y_shape = (1,)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun1 = cls(x_shape, y_shape, build_model=build_model1)
    fun1.update(xs)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun2 = cls(x_shape, y_shape, build_model=build_model1)
    fun2.update(xs)

    import tempfile
    with tempfile.TemporaryDirectory() as path:
        fun1.save(path)
        fun2.restore(path)
        assert all([np.all(np.isclose(v1-v2,0.0)) for v1, v2 in zip(fun1.variable,fun2.variable)])
        import pdb; pdb.set_trace()
        assert_array(fun1.predict(xs), fun2.predict(xs))


class Testsp(unittest.TestCase):

    #  def test_keras_func_app(self):
    #      cls = KerasFuncApp
    #      test_build_model(cls)
    #      test_copy(cls)
    #      test_predict(cls)
    #      test_save_and_restore(cls)

    def test_robust_keras_func_app(self):
        cls = RobustKerasFuncApp
        test_build_model(cls)
        test_copy(cls)
        test_predict(cls)
        test_save_and_restore(cls)

        x_shape = (10,2,3)
        y_shape = (3,)
        xs = np.random.random([100]+list(x_shape))
        ys = np.random.random([100]+list(y_shape))

        fun = cls(x_shape, y_shape, build_model=build_model1)
        fun.update(xs, ys)

        assert_array(fun._in_nor._bias, np.mean(xs, axis=0))
        assert_array(fun._out_nor._bias, np.mean(ys, axis=0))

        fun.predict(xs)

if __name__ == '__main__':
    unittest.main()
