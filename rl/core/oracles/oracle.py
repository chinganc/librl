from abc import ABC, abstractmethod
import copy

class Oracle(ABC):
    """ An abstract interface of functions.

        `Oracle` provides a unified interface for defining optimization
        objectives, or building function approximators, etc.

        The user would want to implement the following methods:

            `fun`  returns the function value given an input.
            `grad` returns the gradient with respect to an input.
            `hess` returns the Hessian with respect to an input.
            `hvp`  returns the Hessia-vector-product with respect to an input.
            `update` redefines the function.

        Implementation of all these methods is not mandatory. For example, the
        gradient of the function might not be defined.

        Finally, a subclass of `Oracle` should be copy.deepcopy compatible. For
        convenience, we overload __deepcopy__ to include an `exclude` list, in
        order not to deepcopy some attributes.
    """
    def fun(self, x, **kwargs):
        """ Return the function value given an input. """
        raise NotImplementedError

    def grad(self, x, **kwargs):
        """ Return the gradient with respect to an input as np.ndarray(s). """
        raise NotImplementedError

    def hess(self, x, **kwargs):
        """ Return the Hessian with respect to an input as np.ndarray(s). """
        raise NotImplementedError

    def hvp(self, x, g, **kwargs):
        """ Return the product between Hessian and a vector `g` with respect to
            an input as np.ndarray(s). """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """ Redefine the function. """
        raise NotImplementedError

    def assign(self, other, excludes=()):
        """ Set the parameters as others. """
        assert type(self)==type(other)
        self.__dict__.update(copy.deepcopy(other).__dict__)
        # if hasattr(other,'__getstate__'):
        #     d = other.__getstate__()
        # else:
        #     d = other.__dict__
        # d = dict(d)
        # for k in excludes:
        #     del d[k]
        # if hasattr(self,'__setstate__'):
        #     self.__setstate__(d)
        # else:
        #     self.__dict__.update(d)

    def __deepcopy__(self, memo, excludes=()):
        """ __deepcopy__ but with an exclusion list
            excludes is a list of attribute names (string) that is to be shallow copied.
        """
        new = copy.copy(self)
        memo[id(self)] = new  # prevent loop
        if hasattr(self,'__getstate__'):
            d = self.__getstate__()
        else:
            d = self.__dict__
        # don't deepcopy the items in `excludes`
        d = dict(d)
        for k in excludes:
            del d[k]
        # deepcopy others
        d = copy.deepcopy(d, memo)
        if hasattr(new,'__setstate__'):
            new.__setstate__(d)
        else:
            new.__dict__.update(d)
        return new
