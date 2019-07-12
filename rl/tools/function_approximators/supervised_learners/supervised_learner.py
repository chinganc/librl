from abc import abstractmethod
from functools import wraps
import numpy as np

from rl.tools import function_approximators as fa
from rl.tools.datasets import Dataset, data_namedtuple


Data = data_namedtuple('Data', 'x y w')
class SupervisedLearner(fa.FunctionApproximator):
    """
    FunctionApproximator trained in a supervised learning manner on aggregated
    data.

    Its predict method can be decorated with predict_in_batches decorator for
    considering memory constraint.
    """
    # NOTE this class is meant to be a wrapper so we use **kwargs
    def __init__(self, x_shape, y_shape, name='supervised_learner',
                 max_n_samples=0,
                 max_n_batches=0,
                 batch_size_for_prediction=2048
                 **kwargs):

        super().__init__(x_dim, y_dim, name=name, **kwargs)
        self._dataset = Dataset(max_n_samples=max_n_samples,
                                max_n_batches=max_n_batches)
        self._batch_size_for_prediction=batch_size_for_prediction

    def update(self, xs, ys, w=1.0, **kwargs):
        """ Update the function approximator through supervised learning, where
        x, y, and w are inputs, outputs, and the weight on each datum.  """
        ws = np.ones(x.shape[0])*w if type(w) is not np.ndarray else w
        assert xs.shape[0] == ys.shape[0] == ws.shape[0]
        data = Data(x=xs, y=ys, w=ws)
        self._dataset.append(data)
        super().update(xs, ys, ws, **kwargs)
        self._update_func_approx(xs, ys, ws, **kwargs)  # user-defined

    # Methods to be implemented
    @abstractmethod
    def _update_func_approx(self, xs, ys, ws, **kwargs):
        """ Update the function approximator based on the current data (x, y,
        w) or through self._agg_data which is up-to-date with (x, y, w). """


def make_robust_keras_supervised_learner(cls):
    assert isinstance(cls, fa.KerasRobustFuncApp):
    class decorated_cls(SupervisedLearner, cls):

        def __init__(self, x_shape, y_shape, name='k_robust_super_learner', **kwargs):
            super().__init__(self, x_shape, y_shape, name=name, **kwargs)

        def update(self, xs, ys, w=1.0, lr=0.001,
                   **kwargs): # for keras.Model.fit
            xs = self._x_nor(xs)  # need to normalize
            super().update(xs, ys, w=1.0, lr=lr, **kwargs)  # we do not consider

        def _update_func_approx(self, xs, ys, ws, lr, **kwargs):
            self._kfun.kmodel.compile(optimizer=tf.train.AdamOptimizer(lr),
                                      loss='mse',
                                      metrics=['mae'])
            self._kfun.kmodel.fit(self._dataset['x'], self._dataset['y'], **kwargs)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls



@make_robust_keras_supervised_learner
class superKerasRobustFuncApp(fa.KerasRobustFuncApp):
    pass


@make_robust_keras_supervised_learner
class superKerasRobustMLP(fa.KerasRobustMLP):
    pass


