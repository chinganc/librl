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
                 gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2):
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

        # for sampling
        if horizon is None: horizon=float('Inf')
        self._horizon = horizon
        self._gamma = gamma
        self._reset_pi_ro()

    def _reset_pi_ro(self):
        self._sample=True
        self._ro_for_cv = False
        self._t_switch = []
        self._s_switch = []

    def pi(self, ob, t, done):
        return self.policy(ob)

    def pi_ro(self, ob, t, done):
        # TODO share randomness
        if self._ro_for_cv:
            return self.policy(ob)
        else:  # roll-in policy and roll-out expert
            if t==0:
                assert self._sample
                if self._horizon < float('Inf'):
                    p0 = self._gamma**np.arange(self._horizon)
                    sump0 = np.sum(p0)
                    p = p0/sump0
                    ind = np.random.multinomial(1,p)
                    self._t_switch.append(np.where(ind==1)[0][0])
                    self._s_switch.append(sump0)
                else:
                    self._t_switch.append(np.random.geometric(p=1-self._gamma)[0]-1)
                    self._s_switch.append(1/(1-self._gamma))
                self._sample=False
            if t<self._t_switch[-1]:  # roll-in
                return self.policy(ob)
            else:
                return self.expert(ob)

    def logp(self, obs, acs):
        if self._ro_for_cv:
            self._ro_for_cv = False
            return self.policy.logp(obs,acs)
        else:
            self._ro_for_cv = True
            self._sample = True
            t_switch = self._t_switch[-1]
            logp0 = self.policy.logp(obs[:t_switch], acs[:t_switch])
            logp1 = self.expert.logp(obs[t_switch:], acs[t_switch:])
            return np.concatenate([logp0, logp1])

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            ro = gen_ro(self.expert, logp=self.expert.logp)
            self.oracle.update(ro_pol=ro, policy=policy)

    def update(self, ro):
        # Update input normalizer for whitening
        self.policy.update(ro['obs_short'])

        # Correction Step (Model-free)
        with timed('Update oracle'):
            # split ro
            rollouts = ro.to_list()
            ro_mix = rollouts[0:][::2]
            #[setattr(r,'scale',s) for r, s in zipsame(ro_mix, self._s_switch)]
            #ro_exp = [r[t:] for r,t in zipsame(ro_mix, self._t_switch) if len(r)>t]

            ro_exp = []
            try: 
                for r,t,s in zipsame(ro_mix, self._t_switch, self._s_switch):
                    if len(r)>t:
                        r = r[t:]
                        r.scale=s
                        ro_exp.append(r)
            except: 
                import pdb; pdb.set_trace()
            ro_exp = Dataset(ro_exp)
            ro_pol = Dataset(rollouts[1:][::2])
            _, ev0, ev1 = self.oracle.update(ro_exp=ro_exp, ro_pol=ro_pol,
                                             policy=self.policy)
            self._reset_pi_ro()

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
