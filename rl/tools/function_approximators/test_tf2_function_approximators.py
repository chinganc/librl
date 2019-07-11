import unittest
import copy
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from rl.tools.function_approximators.tf2_function_approximators import tf2RobustMLP

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-8))

def test_copy(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape)

    new_fun = copy.deepcopy(fun)
    new_fun.variables = fun.variables
    new_fun.variable = fun.variable+1
    assert all([np.all(np.isclose(v1-v2,1.0)) for v1, v2 in zip(new_fun.variable,fun.variable)])

def test_predict(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape)

    x = np.random.random(fun.x_shape)
    fun(x)
    xs = np.random.random([10,]+list(fun.x_shape))
    fun.predict(xs)

def test_save_and_restore(cls):
    x_shape = (10,2,3)
    y_shape = (1,)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun1 = cls(x_shape, y_shape)
    fun1.update(xs)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun2 = cls(x_shape, y_shape)
    fun2.update(xs)

    import tempfile
    with tempfile.TemporaryDirectory() as path:
        fun1.save(path)
        fun2.restore(path)
        assert all([np.all(np.isclose(v1-v2,0.0)) for v1, v2 in zip(fun1.variable,fun2.variable)])
        assert_array(fun1.predict(xs), fun2.predict(xs))


class Tests(unittest.TestCase):


    def test_func_app(self):
        cls = tf2RobustMLP
        test_copy(cls)
        test_predict(cls)
        test_save_and_restore(cls)

        x_shape = (10,2,3)
        y_shape = (3,)
        xs = np.random.random([100]+list(x_shape))
        ys = np.random.random([100]+list(y_shape))

        fun = cls(x_shape, y_shape, build_kmodel=build_kmodel1)
        fun.update(xs, ys)

        assert_array(fun._in_nor._bias, np.mean(xs, axis=0))
        assert_array(fun._out_nor._bias, np.mean(ys, axis=0))

        fun.predict(xs)

if __name__ == '__main__':
    unittest.main()
