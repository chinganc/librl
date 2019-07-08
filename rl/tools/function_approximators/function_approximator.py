from abc import ABC, abstractmethod
from functools import wraps

from rl.tools.utils.misc_utils import flatten, unflatten

def online_compatible(f):
    # A decorator to make f, designed for batch inputs and outputs, support both single and
    # batch predictions
    @wraps(f)
    def wrapper(self, x, **kwargs):
        x = [xx[None,:] for xx in x] if type(x) is list else x[None,:]  # add an extra dimension
        y = f(x, **keras_kwargs)
        y = [yy[0] for yy in y] if type(y) is list else y[0]  # remove the extra dimension
    return wrapper


def assert_shapes(s1, s2):
    assert type(s1)==type(s2)
    if type(s1) is list:
        assert all([ss1==ss2 for ss1,ss2 in zip(s1,s2)])
    else:
        assert s1==s2



class FunctionApproximator(ABC):
    """
        An abstract interface of function approximators.

        Generally a function approximator has "variables" that are amenable to gradient-based updates,
        and "parameters" that works as the hyper-parameters.

        The user needs to implement the following
            predict, variables (getter and setter), save, restore
            pre_callback, post_callback (optional)

        In addition, the class should be copy.deepcopy compatible.
    """

    def __init__(self, x_shapes, y_shapes, name='func_approx', seed=None):
        self.name = name
        self.x_shapes = x_shapes  # a nd.array or a list of nd.arrays
        self.y_shapes = y_shapes  # a nd.array or a list of nd.arrays
        self.seed = seed

    @abstractmethod
    def predict(self, x, online=False):
        """ Predict the values at x.

            When online is False, the first dimension of x is treated as the batch size so should
            the first dimension of the output. When online is True, an instance of x is provided and
            the output should matches the size of an instance of prediction (not including the batch
            size).
        """
    def pre_callback(self, *args, **kwargs):
        """ Perform pre-processing before the update of variables.

            This can include updating internal normalizers, etc.
        """
    def post_callback(self, *args, **kwargs):
        """ Perform post-processing after the update of varaibles.

            This can include logging, etc.
        """
    @property
    @abstractmethod
    def variables(self):
        """ Return the variables as a list of nd.ndarrays """

    @variables.setter
    @abstractmethod
    def variables(self, vals):
        """ Set the variables as val, which is in the same format as self.variables."""

    @property
    def variable(self):
        """ Return a np.ndarray of the variables """
        return flatten(self.variables)

    @variable.setter
    def variable(self, val):
        """ Set the variables as val, which is a np.ndarray in the same format as self.variable."""
        self.variables = unflatten(val, template=self.variables)

    # utilities
    @abstractmethod
    def assign(self, other):
        """ Set both the variables and the parameters as other.

            In general, this may be different from copy.deepcopy; with assign, self can have
            different, e.g., update behaviors from other.
        """
    @abstractmethod
    def save(self, path, *args, **kwargs):
        """ Save the instance in path"""

    @abstractmethod
    def restore(self, path, *args, **kwargs):
        """ restore the saved instance in path """
