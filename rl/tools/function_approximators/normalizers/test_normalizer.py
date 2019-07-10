import unittest
import os
import copy
import numpy as np
import rl.tools.function_approximators.normalizers as Nor
from rl.tools.function_approximators.normalizers.normalizer import Normalizer


class Tests(unittest.TestCase):

    def test_base_normalizer(self):
        shape = (1,3,4)
        nor = Normalizer(shape)
        xs = np.random.random([10]+list(shape))
        nxs = nor.normalize(xs)
        assert np.all(np.isclose(xs,nxs))

        x = np.random.random(shape)
        nx = nor(x)
        assert np.all(np.isclose(x,nx))

        import tempfile
        with tempfile.TemporaryDirectory() as path:
            nor.save(path)
            nor2 = copy.deepcopy(nor)
            nor2.save(path)

    def test_normalizer_std(self):
        shape = (1,2,3)
        nor = Nor.NormalizerStd(shape)
        xs = np.random.random([10]+list(shape))
        nor.update(xs)
        assert np.all(np.isclose(nor.bias-np.mean(xs,axis=0),0.0))
        assert np.all(np.isclose(nor.scale-np.std(xs,axis=0),0.0))
        assert nor._initialized  is True

        xs2 = np.random.random([10]+list(shape))
        xs = np.concatenate((xs,xs2))
        nor.update(xs2)
        assert np.all(np.isclose(nor.bias-np.mean(xs,axis=0),0.0))
        assert np.all(np.isclose(nor.scale-np.std(xs,axis=0),0.0))

        # single instance
        nor(xs[0])
        nor.update(xs[0])

        # reset
        nor.reset()
        assert nor._initialized is False


if __name__ == '__main__':

    unittest.main()
