import unittest
import os
import copy
import numpy as np
import tensorflow as tf
import rl.tools.function_approximators.normalizers as Nor

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-5))

class Tests(unittest.TestCase):

    @staticmethod
    def _test(cls, tf_cls):
        shape = (1,2,3)
        nor = cls(shape)
        tf_nor = tf_cls(shape)

        for _ in range(1000):
            xs = np.random.random([10]+list(shape))
            nor.update(xs)
            tf_nor.update(xs)

            assert nor._initialized is True
            assert tf_nor._initialized is True

            xs = np.random.random([10]+list(shape))
            nxs1 = nor.normalize(xs)
            nxs2 = tf_nor.normalize(xs)
            nxs3 = tf_nor.ts_normalize(tf.constant(xs,dtype=tf.float32)).numpy()
            assert_array(nxs1, nxs2)
            assert_array(nxs1, nxs3)

    def test_tf2_normalizers(self):

        self._test(Nor.NormalizerStd, Nor.tf2NormalizerStd)
        self._test(Nor.NormalizerMax, Nor.tf2NormalizerMax)



if __name__ == '__main__':

    unittest.main()
