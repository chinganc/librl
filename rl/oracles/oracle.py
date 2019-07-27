from abc import abstractmethod
from rl.core.oracles import Oracle


class rlOracle(Oracle):
    """ rlOracle interacts in terms of `ro` which is a Dataset. """

    # `fun`, `grad`, `hess` should now take policy.variable as input.

    @property
    @abstractmethod
    def ro(self):
        """Return the effective ro that defines this oracle."""

