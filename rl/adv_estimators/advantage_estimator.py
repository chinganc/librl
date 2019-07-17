from abc import abstractmethod
import numpy as np
import copy
from .performance_estimate import PerformanceEstimate as PE
from rl.core.function_approximators.policies import Policy
from rl.core.function_approximators import FunctionApproximator
from rl.core.datasets import Dataset


class AdvantageFuncApp(FunctionApproximator):
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
        return [np.exp(policy.logp(rollout.obs_short, rollout.acs) - rollout.lps) for rollout in ro]

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
                 v_target,  # target of learning value function
                 use_is_pe=False,  # for value function learning
                 use_is=False,  # for value function learning
                 n_updates=5,  # number of iterations in policy evaluation
                 name='value_based_adv_func_app',
                 **kwargs):
        """ Create an advantage estimator wrt ref_policy. """
        self._ref_policy = ref_policy  # Policy object
        # helper object to compute estimators for Bellman-like objects
        self._pe = PE(gamma=gamma, lambd=lambd, delta=delta)
        # importance sampling  #TODO
        assert use_is in ['one', 'multi', False, None]
        self.use_is = use_is
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
        # compute the target for regression, the expected Q function wrt self._ref_polic
        w = np.concatenate(self.weights(self.ro))[:,None] if self.use_is else 1.0
        for i in range(self._n_updates):
            v_hat = w*np.concatenate(self.qfns(self.ro, self.pe_lambd)).reshape([-1, 1])  # target
            self._vfn.update(self.ro.obs_short, v_hat)

    def advs(self, ro, lambd=None, use_is=None, ref_policy=None):  # advantage function
        """ Compute adv (evaluated at ro) wrt to ref_policy.

            Note `ref_policy` argument is only considered when `self.use_is`
            is True; in this case, if `ref_policy` is None, it is wrt to
            `self._ref_policy`. Otherwise, when `self.use_is`_is is False, the
            adv is biased toward the behavior policy that collected the data.
        """
        use_is = use_is or self.use_is
        vfns = self.vfns(ro)
        if use_is is 'multi':
            ws = self.weights(ro, ref_policy)  # importance weight
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, w=w, lambd=lambd)
                    for rollout, vf, w in zip(ro.rollouts, vfns, ws)]
        else:
            advs = [self._pe.adv(rollout.rws, vf, rollout.done, w=1.0, lambd=lambd)
                    for rollout, vf in zip(ro.rollouts, vfns)]
        return advs, vfns

    def qfns(self, ro, lambd=None, use_is=None, ref_policy=None):  # Q function
        advs, vfns = self.advs(ro, lambd=lambd, use_is=use_is, ref_policy=ref_policy)
        qfns = [adv + vfn[:-1] for adv, vfn in zip(advs, vfns)]
        return qfns

    def vfns(self, ro):  # value function
        return [self._vfn.predict(rollout.obs) for rollout in ro.rollouts]


class QBasedAdvFuncApp(ValueBasedAdvFuncApp):
    pass
