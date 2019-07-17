from abc import abstractmethod
from rl.core.oracles import Oracle


class rlOracle(Oracle):
    """ rlOracle interacts in terms of policies and `ro` which is a Dataset. """

    # These functions should now take a Policy instance as input.
    def fun(self, policy, **kwargs):
        """ Return the function value given an input. """
        raise NotImplementedError

    def grad(self, policy, **kwargs):
        """ Return the gradient with respect to an input as np.ndarray(s). """
        raise NotImplementedError

    def hess(self, policy, **kwargs):
        """ Return the Hessian with respect to an input as np.ndarray(s). """
        raise NotImplementedError

    @abstractmethod
    def update(ro, *args, **kwargs):
        """The update method should take ro as the first argument."""

    @property
    @abstractmethod
    def ro(self):
        """Return the effective ro that defines this oracle."""

