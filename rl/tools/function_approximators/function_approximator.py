from abc import ABC, abstractmethod
from functools import wraps

from rl.tools.utils.misc_utils import flatten, unflatten

def assert_shapes(s1, s2):
    assert type(s1)==type(s2)
    if isinstance(s1, list):
        assert all([ss1==ss2 for ss1,ss2 in zip(s1,s2)])
    else:
        assert s1==s2

class FunctionApproximator(ABC):
    """ An abstract interface of function approximators.

        Generally a function approximator has
            1) "variables" that are amenable to gradient-based updates,
            2) "parameters" that works as the hyper-parameters.

        The user needs to implement the following
            `predict`, `variables` (getter and setter), `assign`, `save`, `restore`
            `update` (optional)

        In addition, the class should be copy.deepcopy compatible.
    """
    def __init__(self, x_shape, y_shape, name='func_approx', seed=None):
        self.name = name
        self.x_shape = x_shape  # a nd.array or a list of nd.arrays
        self.y_shape = y_shape  # a nd.array or a list of nd.arrays
        self.seed = seed

    @abstractmethod
    def predict(self, xs, **kwargs):
        """ Predict the values over batches of xs. """

    def __call__(self, x, **kwargs):
        """ Predict the value at x """
        x = [xx[None,:] for xx in x] if isinstance(x, list) else x[None,:]  # add an extra dimension
        y = self.predict(x, **kwargs)
        y = [yy[0] for yy in y] if isinstanc(y, list) else y[0]  # remove the extra dimension
        return y

    def update(self, *args, **kwargs):
        """ Perform update the parameters.

            This can include updating internal normalizers, etc.
        """

    @property
    @abstractmethod
    def variables(self):
        """ Return the variables as a list of nd.ndarrays. """

    @variables.setter
    @abstractmethod
    def variables(self, vals):
        """ Set the variables as val, which is in the same format as self.variables."""

    @property
    def variable(self):
        """ Return a np.ndarray of the variables. """
        return flatten(self.variables)

    @variable.setter
    def variable(self, val):
        """ Set the variables as val, which is a np.ndarray in the same format as self.variable. """
        self.variables = unflatten(val, template=self.variables)

    # utilities
    @abstractmethod
    def assign(self, other):
        """ Set both the variables and the parameters as other. """

    @abstractmethod
    def save(self, path, *args, **kwargs):
        """ Save the instance in path. """

    @abstractmethod
    def restore(self, path, *args, **kwargs):
        """ restore the saved instance in path. """
