from abc import abstractmethod
from rl.core.online_learners import OnlineLearner

class Algorithm(OnlineLearner):

    @property
    @abstractmethod
    def policy(self):
        """ Return a Policy object """

    @abstractmethod
    def pretrain(self, gen_ro):
        """ Pretraining. """

    @abstractmethod
    def update(self, ro):
        """ Update the policy based on rollouts. """

    @abstractmethod
    def pi(self, ob, t, done):
        """ Target policy for online querying. """

    @abstractmethod
    def pi_ro(self, ob, t, done):
        """ Behavior policy for online querying. """

    @abstractmethod
    def logp(self, obs, acs):
        """ Log probability of the behavior policy.
            Need to support batch querying. """

    @property
    def decision(self):
        return self.policy
