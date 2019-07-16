from abc import abstractmethod
import numpy as np
import copy
from .performance_estimate import PerformanceEstimate as PE
from rl.policies import Policy
from rl.core.function_approximators import FunctionApproximator
from rl.core.datasets import Dataset


class AdvantageFuncApp(FunctionApproximator)
    """ An abstract advantage function estimator based on replay buffer.

        The user needs to implement methods required by `FunctionApproximator`
            `update`, `predict`, `variable` (setter and getter)
        and the new methods
            `advs`, `qfns`, `vfns`
    """

    # NOTE We overload the interfaces here to work with policies and rollouts.
    # This class can no longer use as a wrapper of usual `FunctionApproximator`.
    def __init__(self, ref_policy, name='advantage_func_app',
                 max_n_samples=0,  # number of samples to keep
                 max_n_batches=0,  # number of batches (i.e. rollouts) to keep
                 **kwargs):
        self.ro = Dataset(max_n_batches=max_n_batches, max_n_samples=max_n_samples)  # replay buffer
        assert isinstance(ref_policy, Policy)
        self._ref_policy = ref_policy  # reference policy
        self._ob_shape = ref_policy.x_shape
        self._ac_shape = ref_policy.y_shape
        super().__init__([self._ob_shape, self._ac_shape], (1,), name=name, **kwargs)

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

   # helper functions
    def weights(self, ro, policy=None):  # importance weight
        # ro is a Dataset or list of rollouts
        policy = policy or self._ref_policy
        assert isinstance(policy, Policy)
        return [np.exp(policy.logp(rollout.obs[:-1], rollout.acs) - rollout.lps) for rollout in ro]

    # _ref_policy should not be deepcopy or saved
    def __getstate__(self):
        if hasattr(super(), '__getstate__'):
            d = super().__getstate__()
        else:
            d = self.__dict__
        d = dict(d)
        del d['_ref_policy']
        return d


class ValueBasedAdvFuncApp(AdvantageFuncApp):
    """ An estimator based on value function. """

    def __init__(self, ref_policy,  # the reference ref_policy of this estimator
                 vfn,  # value function estimator (SupervisedLearner)
                 gamma,  # discount in the problem definition (e.g. 0. for undiscounted problem)
                 delta,  # additional discount to make value function learning well-behave, or to reduce variance
                 lambd,  # mixing rate of different K-step qfun estimates (e.g. 0 for actor-critic, 0.98 GAE)
                 default_v,  # value function of the absorbing states
                 v_target,  # target of learning value function
                 use_is=False,  # whether to use one-step importance weight, for value function learning
                 n_updates=5  # number of iterations in policy evaluation
                 name='value_based_adv_func_app',
                 **kwargs):
        """ Create an advantage estimator wrt ref_policy. """
        self._ref_policy = ref_policy  # Policy object
        # helper object to compute estimators for Bellman-like objects
        self._pe = PE(gamma=gamma, lambd=lambd, delta=delta, default_v=default_v)
        # importance sampling
        assert use_is in ['one', 'multi', False, None]
        self._use_is = use_is
        # policy evaluation
        if v_target == 'monte-carlo':
            self.pe_lambd = 1.0  # using Monte-Carlo samples
        elif v_target == 'td':
            self.pe_lambd = 0.  # one-step td error
        elif v_target == 'same':
            self.pe_lambd = None  # default self.pe_lambda-weighted td error
        elif type(v_target) is float:
            self.pe_lambd = v_target  # user-defined self.pe_lambda-weighted td error
        else:
            raise ValueError('Unknown target {} for value function update.'.format(v_target))
        if v_target == 'monte-carlo' or np.isclose(self.pe_lambd, 1.0):
            n_updates = 1
        assert n_updates >= 1, 'Policy evaluation needs at least one udpate.'
        self._n_updates = n_updates
        # SupervisedLearner for regressing the value function of ref_policy
        assert isinstance(vfn, SupervisedLearner)
        self._vfn = vfn
        self._vfn._dataset.max_n_samples=0
        self._vfn._dataset.max_n_batches=0
        super().__init__(ref_policy, name=name, **kwargs)


    def update(self, ro):
        """ Policy evaluation """
        # update the replay buffer
        self.ro.extend(ro)
        # compute the target for regression, the expected Q function wrt self._ref_policy
        if use_is is 'one':
            w = np.concatenate(self.weights(self.ro))[:,None]
        elif use_is is 'multi'
            ws = self.weights(ro, ref_policy)  # importance weight
        else: 
            w= 1.0
        for i in range(self._n_updates):
            expected_qfn = w * np.concatenate(self.qfns(self.ro, lambd)).reshape([-1, 1])  # target
            self._vfn.update(self.ro.obs, expected_qfn)


    def advs(self, ro, lambd=None, ref_policy=None):  # advantage function
        """
        Compute adv (evaluated at ro) wrt to ref_policy, which may be different from the data collection
        ref_policy. Note ref_policy is only considered when self._multistep_is is True; in this case,
        if ref_policy is None, it is wrt to self._ref_policy. Otherwise, when self._multistep_is is
        False, the adv is biased toward the data collection ref_policy.
        """
        vfns = self.vfns(ro)
        if use_is is 'one':
            w = np.concatenate(self.weights(self.ro))[:,None] if self._onestep_is else 1.0
        elif use_is is 'multi'
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
        return [self._vfn.predict(rollout.obs) for rollout in ro.rollouts]

