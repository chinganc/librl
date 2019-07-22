import numpy as np
from rl.algorithms.algorithm import Algorithm
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient
from rl.core.online_learners import base_algorithms as balg
from rl.core.online_learners import BasicOnlineOptimizer
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz

class PolicyGradient(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, vfn, lr=1e-3,
                 gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2,
                 warm_up_itrs=None,
                 n_pretrain_interactions=1):
        self.vfn = vfn
        self._policy = policy
        # create online learner
        x0 = self.policy.variable
        scheduler = PowerScheduler(lr)
        self.learner = BasicOnlineOptimizer(balg.Adam(x0, scheduler))
        # create oracle
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd,
                               use_is='one', max_n_batches=max_n_batches)
        self.oracle = ValueBasedPolicyGradient(policy, self.ae)

        self._n_pretrain_interactions = n_pretrain_interactions
        if warm_up_itrs is None:
            warm_up_itrs = float('Inf')
        self._warm_up_itrs =warm_up_itrs
        self._itr = 0

    @property
    def policy(self):
        return self._policy

    def pi(self, ob, t, done):
        return self.policy(ob)

    def pi_ro(self, ob, t, done):
        return self.policy(ob)

    def logp(self, obs, acs):
        return self.policy.logp(obs, acs)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_interactions):
                ro = gen_ro(self.pi_ro, logp=self.logp)
                self.oracle.update(ro, self.policy)
                self.policy.update(xs=ro['obs_short'])

    def update(self, ro):
        # Update input normalizer for whitening
        if self._itr < self._warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        # Correction Step (Model-free)
        with timed('Update oracle'):
            _, ev0, ev1 = self.oracle.update(ro, self.policy)

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy)

        with timed('Policy update'):
            self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
        logz.log_tabular('ExplainVarianceAfter(AE)', ev1)

        self._itr +=1
