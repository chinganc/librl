import numpy as np
from rl.algorithms.algorithm import Algorithm
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.tf2_rl_oracles import tfValueBasedExpertGradient
from rl.core.online_learners import base_algorithms as balg
from rl.core.online_learners import BasicOnlineOptimizer
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.datasets import Dataset


class AggreVaTeD(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, expert, expert_vfn, horizon,
                 lr=1e-3,
                 gamma=1.0, delta=None, lambd=0.9,
                 max_n_batches=1000,
                 n_pretrain_interactions=4):
        self.policy = policy
        self.expert = expert
        self.expert_vfn = expert_vfn
        # create online learner
        x0 = self.policy.variable
        scheduler = PowerScheduler(lr)
        self.learner = BasicOnlineOptimizer(balg.Adam(x0, scheduler))
        # create oracle
        self.ae = ValueBasedAE(expert, expert_vfn,  # wrt expert
                               gamma=gamma, delta=delta, lambd=lambd,
                               use_is='one', max_n_batches=max_n_batches)
        self.oracle = tfValueBasedExpertGradient(policy, self.ae)

        self._n_pretrain_interactions = n_pretrain_interactions

        # for sampling
        if horizon is None: horizon=float('Inf')
        assert horizon<float('Inf') or gamma<1.
        self._horizon = horizon
        self._gamma = gamma
        self._reset_pi_ro()  # should be called for each iteration

    def _reset_pi_ro(self):
        self._sample=True  #  randomly sample a switching time step
        self._ro_for_cv = False  # in the phase of cv rollout
        self._t_switch = []  # switching time
        self._scale = None  # extra scaling factor due to switching
        self._noises = []  # noises used in the

    def pi(self, ob, t, done):
        return self.policy(ob)

    # It alternates between two phases
    #   1) roll-in learner and roll-out expert for unbiased gradient
    #   2) roll fully the learner for control variate
    #
    # NOTE The below implementation assume that `logp` is called "ONLY ONCE" at
    # the end of each rollout.

    def pi_ro(self, ob, t, done):
        if self._ro_for_cv:  # just run the learner
            if len(self._noises)>0:
                ns = self._noises[0]
                del self._noises[0]
                return self.policy.derandomize(ob, ns)
            else:
                return self.policy(ob)
        else:  # roll-in policy and roll-out expert
            if t==0:
                assert self._sample
                # At the begining of the two-phased rollouts
                # sample t_switch in [1, horizon]
                if self._horizon < float('Inf'):
                    p0 = self._gamma**np.arange(self._horizon)
                    sump0 = np.sum(p0)
                    p = p0/sump0
                    ind = np.random.multinomial(1,p)
                    self._t_switch.append(np.where(ind==1)[0][0]+1)
                    self._scale=sump0
                else:
                    self._t_switch.append(np.random.geometric(p=1-self._gamma)[0])
                    self._scale=1/(1-self._gamma)
                self._sample=False
                self._noises = []

            if t<self._t_switch[-1]:  # roll-in
                ac = self.policy(ob)
                ns = self.policy.noise(ob, ac)
                self._noises.append(ns)
                return ac
            else:
                return self.expert(ob)

    def logp(self, obs, acs):
        if self._ro_for_cv:
            self._ro_for_cv = False
            return self.policy.logp(obs, acs)
        else: # roll-in policy and roll-out expert
            assert len(self._noises)<=self._t_switch[-1]
            self._ro_for_cv = True  # do cv rollout the next time
            self._sample = True
            t_switch = self._t_switch[-1]
            logp0 = self.policy.logp(obs[:t_switch], acs[:t_switch])
            logp1 = self.expert.logp(obs[t_switch:], acs[t_switch:])
            return np.concatenate([logp0, logp1])

    def pretrain(self, gen_ro):
        pi_exp = lambda ob, t, done: self.expert(ob)
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_interactions):
                ro = gen_ro(pi_exp, logp=self.expert.logp)
                self.oracle.update(ro_pol=ro, policy=self.policy)
                # maybe also update policy
        self._reset_pi_ro()

    def update(self, ro):
        # Update input normalizer for whitening
        self.policy.update(ro['obs_short'])

        # Mirror descent
        with timed('Update oracle'):
            # Split ro into two phases
            rollouts = ro.to_list()[:int(len(ro)/2)*2]  # even length
            ro_mix = rollouts[0:][::2]  # ro with random switch
            assert len(ro_mix)==len(self._t_switch) or len(ro_mix)==len(self._t_switch)-1
            # if a rollout too short, it is treated as zero
            ro_exp = [r[t-1:] for r, t in zip(ro_mix, self._t_switch) if len(r)>=t]
            [ setattr(r,'scale',self._scale) for r in ro_exp ]
            ro_exp = Dataset(ro_exp)
            ro_pol = Dataset(rollouts[1:][::2])

            _, ev0, ev1 = self.oracle.update(ro_exp=ro_exp,
                                             ro_pol=ro_pol,
                                             policy=self.policy)

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
        logz.log_tabular('NumberOfRandomRollouts', len(ro_exp))
        logz.log_tabular('NumberOfCVRollouts', len(ro_pol))

        # reset
        self._reset_pi_ro()

