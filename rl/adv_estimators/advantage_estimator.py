from abc import abstractmethod
import numpy as np
import copy
from .performance_estimate import PerformanceEstimate as PE
from rl.policies import Policy
from rl.core.function_approximators import FunctionApproximator
from rl.core.datasets import Dataset

class AdvantageFunction(FunctionApproximator)
    """ An abstract advantage function estimator based on replay buffer.

        The user needs to implement
            `update`, `predict`, variable (setter and getter) from FunctionApproximator
        and the new methods
            `advs`, `qfns`, `vfns`
    """
    # We overload the interface to work with policies and rollouts.

    def __init__(self, ref_policy, name='advantage_function',
                 max_n_samples=0,  # number of samples to keep
                 max_n_batches=0,  # number of batches/rollouts to keep
                 **kwargs):
        self._ro = Dataset(max_n_batches=max_n_batches, max_n_samples=max_n_samples)  # replay buffer
        self._ref_policy = ref_policy  # reference policy
        super().__init__([policy.x_shape, policy.y_shape], (1,), name=name, **kwargs)

    @abstractmethod
    def update(self, ro, *args, **kwargs):
        """ based on rollouts """

    @abstractmethod
    def advs(self, ro, *args, **kwargs):  # advantage function
        """ Return a list of nd.arrays, one for each rollout. """

    @abstractmethod
    def qfns(self, ro, *args, **kwargs):  # Q function
        """ Return a list of nd.arrays, one for each rollout. """

    @abstractmethod
    def vfns(self, ro, *args, **kwargs):  # value function
        """ Return a list of nd.arrays, one for each rollout. """

    # _ref_policy should not be deepcopy or saved
    def __getstate__(self):
        if hasattr(super(), '__getstate__'):
            d = super().__getstate__()
        else:
            d = self.__dict__
        d = dict(d)
        del d['_ref_policy']
        return d


class ValueBasedAdvantageEstimator(AdvantageFunction):
    """ An estimator based on value function. """

    def __init__(self, ref_policy,  # the reference ref_policy of this estimator
                 vfn,  # value function estimator (SupervisedLearner)
                 gamma,  # discount in the problem definition (e.g. 0. for undiscounted problem)
                 delta,  # additional discount to make value function learning well-behave, or to reduce variance
                 lambd,  # mixing rate of different K-step qfun estimates (e.g. 0 for actor-critic, 0.98 GAE)
                 default_v,  # value function of the absorbing states
                 v_target,  # target of learning value function
                 # whether to use one-step importance weight (only for value function learning)
                 onestep_weighting=False,
                 multistep_weighting=False,  # whether to use multi-step importance weight
                 data_aggregation=False,  # whether to keep previous data
                 max_n_rollouts=None,  # maximal number of rollouts to keep
                 n_updates=5  # number of iterations in policy evaluation
                 ):
        """ Create an advantage estimator wrt ref_policy. """
        assert isinstance(ref_policy, Policy)
        self._ref_policy = ref_policy  # Policy object
        self._ob_dim = ref_policy.ob_dim
        self._ac_dim = ref_policy.ac_dim
        # helper object to compute estimators for Bellman-like objects
        self._pe = PE(gamma=gamma, lambd=lambd, delta=delta, default_v=default_v)
        # importance sampling
        self._multistep_weighting = multistep_weighting
        self._onestep_weighting = onestep_weighting
        # policy evaluation
        self._v_target = v_target
        if v_target == 'monte-carlo' or np.isclose(v_target, 1.0):
            n_updates = 1
        assert n_updates >= 1, 'Policy evaluation needs at least one udpate.'
        self._n_updates = n_updates
        # replay buffer
        self._ro = None
        self._max_n_rollouts = max_n_rollouts
        self._data_aggregation = data_aggregation
        # SupervisedLearner for regressing the value function of ref_policy
        assert isinstance(vfn, SupervisedLearner)
        self._vfn = vfn
        if hasattr(self._vfn, 'n_batches'):
            self._vfn.n_batches /= self._n_updates

    def update(self, ro, to_log=False, log_prefix=''):
        # check if replay buffer needs to be udpated
        if self._data_aggregation:
            if self._ro is None:
                self._ro = copy.deepcopy(ro)
                self._ro.max_n_rollouts = self._max_n_rollouts
            else:
                self._ro.append(ro.rollouts)
            ro = self._ro

        # different ways to construct the target in regression
        if self._v_target == 'monte-carlo':
            lambd = 1.0  # using Monte-Carlo samples
        elif self._v_target == 'td':
            lambd = 0.  # one-step td error
        elif self._v_target == 'same':
            lambd = None  # default lambda-weighted td error
        elif type(self._v_target) is float:
            lambd = self._v_target  # user-defined lambda-weighted td error
        else:
            raise ValueError('Unknown target {} for value function update.'.format(self._v_target))

        # compute the target for regression, the expected Q function wrt self._ref_policy
        if self._onestep_weighting:
            w = np.concatenate(self.weights(ro)).reshape([-1, 1])
        else:
            w = 1.0
        for i in range(self._n_updates):
            expected_qfn = w * np.concatenate(self.qfns(ro, lambd)).reshape([-1, 1])  # target
            if i < self._n_updates - 1:
                self._vfn.update(ro.obs, expected_qfn, to_log=False)
            else:  # only log the last iteration
                self._vfn.update(ro.obs, expected_qfn, to_log=to_log, log_prefix=log_prefix)

    # helper functions (which can be overloaded for different classes)
    def weights(self, ro, policy=None):
        policy = self._ref_policy if policy is None else policy
        assert isinstance(policy, Policy)
        return [np.exp(policy.logp(rollout.obs[:-1], rollout.acs) - rollout.lps) for rollout in ro.rollouts]

    def advs(self, ro, lambd=None, ref_policy=None):  # advantage function
        """
        Compute adv (evaluated at ro) wrt to ref_policy, which may be different from the data collection
        ref_policy. Note ref_policy is only considered when self._multistep_weighting is True; in this case,
        if ref_policy is None, it is wrt to self._ref_policy. Otherwise, when self._multistep_weighting is
        False, the adv is biased toward the data collection ref_policy.
        """
        vfns = self.vfns(ro)
        if self._multistep_weighting:
            ws = self.weights(ro, ref_policy)  # importance weight
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, w, lambd)
                    for rollout, vf, w in zip(ro.rollouts, vfns, ws)]
        else:
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, 1.0, lambd) for rollout, vf in zip(ro.rollouts, vfns)]
        return advs, vfns

    def qfns(self, ro, lambd=None, ref_policy=None):  # Q function
        advs, vfns = self.advs(ro, lambd, ref_policy)
        qfns = [adv + vfn[:-1] for adv, vfn in zip(advs, vfns)]
        return qfns

    def vfns(self, ro):  # value function
        return [np.squeeze(self._vfn.predict(rollout.obs)) for rollout in ro.rollouts]
