from abc import abstractmethod
import numpy as np
from rl.core.function_approximators.function_approximator import FunctionApproximator
from rl.core.datasets import Dataset, data_namedtuple
from rl.core.utils.math_utils import compute_explained_variance

Data = data_namedtuple('Data', 'xs ys ws')
class SupervisedLearner(FunctionApproximator):
    """ FunctionApproximator trained on aggregated data. """

    def __init__(self, x_shape, y_shape, name='supervised_learner',
                 max_n_samples=0,  # number of samples to keep
                 max_n_batches=0,  # number of batches to keep
                 **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)
        self._dataset = Dataset(max_n_samples=max_n_samples,
                                max_n_batches=max_n_batches)

    def update(self, xs, ys, ws=1.0, **kwargs):
        """ Update the function approximator through supervised learning

            xs, ys, and ws are inputs, outputs, and weights.
        """
        assert len(xs.shape)>1 and len(ys.shape)>1
        super().update(xs, ys, ws, **kwargs)
        # update dataset
        ws = np.ones(xs.shape[0])*ws if type(ws) is not np.ndarray else ws
        assert xs.shape[0] == ys.shape[0] == ws.shape[0]
        self._dataset.append(Data(xs=xs, ys=ys, ws=ws))

        # update function approximator
        ev0 = compute_explained_variance(self(xs), ys)
        results = self.update_funcapp(**kwargs)  # return logs, if any
        ev1 = compute_explained_variance(self(xs), ys)
        return results, ev0, ev1

    @abstractmethod
    def update_funcapp(self, **kwargs):
        """ Update the function approximator based on the aggregated dataset.

            Logs of the training results can be returned.
        """
