from abc import abstractmethod, ABC
from rl.core.online_learners import OnlineLearner

class Algorithm(ABC):
    """ An abtract interface required by Experimenter. """

    # Outcome of an algorithm
    @property
    @abstractmethod
    def policy(self):
        """ Return a Policy object which is the outcome of the algorithm. """

    # For update
    @abstractmethod
    def pretrain(self, gen_ro):
        """ Pretrain the policy. """

    @abstractmethod
    def update(self, ro):
        """ Update the policy based on rollouts. """

    # For performance evaluation
    @abstractmethod
    def pi(self, ob, t, done):
        """ Target policy used in online querying. """

    # For data collection
    @abstractmethod
    def pi_ro(self, ob, t, done):
        """ Behavior policy used in online querying. """

    @abstractmethod
    def logp(self, obs, acs):
        """ Log probability of the behavior policy, which will be called at
            end, once, of each rollout.

            Need to support batch querying. """

    # For parallel data collection
    def child_alg(self):
        """ Return an Algorithm for parallel data collection. """
        raise NotImplementedError

    def batch_update(self, batch_ro):
        """ Perform `update` given results collected by child algorithms.

            `batch_ro` is a list of dicts, in which each dict has key `child`
            and key `ro`.
        """
        raise NotImplementedError


