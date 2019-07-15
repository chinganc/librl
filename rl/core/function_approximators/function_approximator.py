from abc import abstractmethod
from functools import wraps
import os, pickle, copy
from rl.core.oracles.oracle import Oracle


def online_compatible(f):
    @wraps(f)
    def decorated_f(self, x, *args, **kwargs):
        if x.shape==self.x_shape:  # single instance
            x = [xx[None,:] for xx in x] if isinstance(x, list) else x[None,:]  # add an extra dimension
            new_args =[[aa[None,:] for aa in a] if isinstance(a, list) else a[None,:] for a in args ]  # add an extra dimension
            y = f(self, x, *new_args, **kwargs)
            y = [yy[0] for yy in y] if isinstance(y, list) else y[0]  # remove the extra dimension
        else:
            y = f(self, x, *args, **kwargs)
        return y
    return decorated_f

# TODO
def predict_in_batches(fun):
    """ for wrapping a predit method of FunctionApproximator objects """
    @wraps(fun)
    def wrapper(self, x):
        return minibatch_utils.apply_in_batches(lambda _x: fun(self, _x),
                                                x, self._batch_size_for_prediction, [self.y_dim])
    return wrapper


class FunctionApproximator(Oracle):
    """ An abstract interface of function approximators.

        Generally a function approximator has
            1) "variables" that are amenable to gradient-based updates,
            2) "parameters" that works as the hyper-parameters.

        This is realized by adding `variables` property to `Oracle`. In
        addition, here we require function calls to be compatible with both
        single-instance and batch queries.

        We also provide basic `assign`, `save`, `restore` functions, based on
        deepcopy and pickle, which should work for nominal python objects. But
        they might need be overloaded when more complex objects are used (e.g.,
        tf.keras.Model) as attributes.

        The user needs to implement the following
            `predict`, `variables` (getter and setter), and `update` (optional)

        In addition, the class should be copy.deepcopy compatible.
    """
    def __init__(self, x_shape, y_shape, name='func_app', **kwargs):
        self.name = name
        self.x_shape = x_shape  # a nd.array or a list of nd.arrays
        self.y_shape = y_shape  # a nd.array or a list of nd.arrays

    @abstractmethod
    def predict(self, xs, **kwargs):
        """ Predict the values on batches of xs. """

    @online_compatible
    def __call__(self, xs, **kwargs):
        return self.predict(xs, **kwargs)

    def fun(self, x, **kwargs):  # alias
        return self(x, **kwargs)

    def update(self, *args, **kwargs):  # needs to be callable
        """ Perform update the parameters.

            This can include updating internal normalizers, etc.
            Return a report, in any.
        """
    @property
    @abstractmethod
    def variable(self):
        """ Return the variable as a np.ndarray. """

    @variable.setter
    @abstractmethod
    def variable(self, val):
        """ Set the variable as val, which is a np.ndarray in the same format as self.variable. """

    # utilities
    def assign(self, other, excludes=()):
        """ Set both the variables and the parameters as other. """
        assert type(self)==type(other)
        self.__dict__.update(copy.deepcopy(other).__dict__)

    def save(self, path):
        """ Save the instance in path. """
        path = os.path.join(path, self.name)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def restore(self, path):
        """ restore the saved instance in path. """
        path = os.path.join(path, self.name)
        with open(path, 'rb') as pickle_file:
            saved = pickle.load(pickle_file)
        self.assign(saved)
