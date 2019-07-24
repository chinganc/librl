import copy
import numpy as np
from functools import partial
from rl.algorithms.algorithm import Algorithm
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient as Oracle
from rl.core.function_approximators.policies import Policy
from rl.core.online_learners import base_algorithms as balg
from rl.core.online_learners import BasicOnlineOptimizer
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.datasets import Dataset
from rl.core.utils.mvavg import PolMvAvg, ExpMvAvg


from rl.core.function_approximators import FunctionApproximator, online_compatible
from rl.core.function_approximators.supervised_learners import SupervisedLearner


class Bandit:
    """ Maximization """
    def update(self, *args, **kwargs):
        pass
    def decision(self, xs, explore=False, **kwargs):
        pass

class ContextualEpsilonGreedy(Bandit):

    def __init__(self, x_shape,  models, eps):
        self.x_shape = x_shape
        self.eps = eps
        self.models = models

    def update(self, k, *args, **kwargs):
        return self.models[k].update(*args, **kwargs)

    @online_compatible
    def decision(self, xs, explore=False, **kwargs):
        # epsilon-greedy choice
        N = len(xs)
        K = len(self.models)
        vals = [m(xs, **kwargs) for m in self.models]
        vals = np.concatenate(vals, axis=1)
        k_star = np.argmax(vals, axis=1)
        val_star = vals.flatten()[k_star+np.arange(N)*K]
        val_star = np.reshape(val_star,[-1,1])
        if explore:
            ind = np.where(np.random.rand(N)<=self.eps)
            k_star[ind] = np.random.randint(0,K,size=(len(ind),))
            return k_star
        else:
            return k_star, val_star


class MaxValueFunction(SupervisedLearner):
    """ Statewise maximum over a set of value functions """
    def __init__(self, vfns, eps=0.5, name='max_vfn'):
        super().__init__(vfns[0].x_shape, vfns[0].y_shape,
                         name=name)
        self.bandit = ContextualEpsilonGreedy(self.x_shape, vfns, eps=eps)
        self.vfns = vfns

    def predict(self, xs, **kwargs):
        _, v_star = self.bandit.decision(xs, explore=False, **kwargs)
        return v_star

    def update(self, *args, k=None, **kwargs):  # overload
        # As vfns are already SupervisedLearners, we don't need to aggregate
        # data here.
        assert not k is None
        return self.update_funcapp(*args, k=k, **kwargs)

    def update_funcapp(self, *args, k=None, **kwargs):
        return self.bandit.update(k, *args, **kwargs)

    @property
    def variable(self):
        return np.concatenate([v.variable for v in self.vfns])

    @variable.setter
    def variable(self, vals):
        [setattr(v,'variable',val) for v, val in zip(self.vfns, vals)]


class PolicyImprovementFromExperts(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, experts, vfn, horizon,
                 lr=1e-3,
                 gamma=1.0, delta=None, lambd=0.9,
                 eps=0.5,
                 max_n_batches=2,
                 max_n_batches_experts=1000,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=5,
                 use_policy_as_expert=True,
                 sampling_rule='exponential', # define how random switching time is generated
                 cyclic_rate=2): # the rate of forward training, relative to the number of iterations
        assert isinstance(policy, Policy)
        self._policy = policy
        self._vfn = vfn
        # Define max over value functions
        vfns = [copy.deepcopy(vfn) for _ in range(len(experts))]
        if use_policy_as_expert:
            experts += [policy]
            vfns += [vfn]
        self.experts = experts
        self.vfn_max = MaxValueFunction(vfns, eps=eps)  # max over values
        if use_policy_as_expert:
            print('Using {} experts, including its own policy'.format(len(vfns)))
        else:
            print('Using {} experts'.format(len(vfns)))

        # Create online learner
        x0 = self.policy.variable
        scheduler = PowerScheduler(lr)
        self.learner = BasicOnlineOptimizer(balg.Adam(x0, scheduler))
        # Create oracle  # TODO max_n_batches
        self.ae = ValueBasedAE(policy, self.vfn_max,  # TODO importance weight
                               gamma=gamma, delta=delta, lambd=lambd,
                               use_is='one', max_n_batches=max_n_batches)
        self.ae.update = None
        self.oracle = Oracle(policy, self.ae)

        # Create ae for updating value functions
        create_ae = partial(ValueBasedAE, gamma=gamma, delta=delta, lambd=lambd,
                            use_is='one')
        aes = []
        for i, (e,v) in enumerate(zip(experts, vfns)):
            if use_policy_as_expert and i==(len(experts)-1):
                aes.append(create_ae(e, v, max_n_batches=max_n_batches))
            else:
                aes.append(create_ae(e, v, max_n_batches=max_n_batches_experts))
        self.aes = aes
        self._use_policy_as_expert = use_policy_as_expert
        # Misc configs
        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs =n_warm_up_itrs
        self._itr = 0

        # for sampling random switching time
        if horizon is None: horizon=float('Inf')
        assert horizon<float('Inf') or gamma<1.
        self._horizon = horizon
        self._gamma = gamma  # discount factor of the original problem
        assert sampling_rule in ['exponential','cyclic','uniform']
        self._sampling_rule = sampling_rule
        self._cyclic_rate = cyclic_rate
        self._avg_n_steps = PolMvAvg(1,weight=1)  # the number of steps that the policy can survive so far
        self._reset_pi_ro()

    def _reset_pi_ro(self):
        # NOTE Should be called for each iteration
        self._locked = False  #  randomly sample a switching time step
        self._ro_with_policy = True  # in the phase of gradient rollout
        self._t_switch = []  # switching time
        self._scale = []  # extra scaling factor due to switching
        self._k_star = []

        self._n_ro = 0
        self._ind_ro_pol = []
        self._ind_ro_mix = []

    @property
    def policy(self):
        return self._policy

    def pi(self, ob, t, done):
        return self.policy(ob)

    # It alternates between two phases
    #   1) roll-in learner and roll-out expert for updating value function
    #   2) roll fully the learner for computing gradients
    #
    # NOTE The below implementation assume that `logp` is called "ONLY ONCE" at
    # the end of each rollout.

    def pi_ro(self, ob, t, done):
        if t==0:  # make sure logp has been called
            assert not self._locked
            self._locked=True

        if self._ro_with_policy:  # just run the learner
            return self.policy(ob)
        else:  # roll-in policy and roll-out expert
            # update self._t_switch, self._scale, self._k_star
            if t==0:  # sample t_switch in [1, horizon]
                self._sample_t_switch()

            if t<self._t_switch[-1]:  # roll-in
                return self.policy(ob)
            else:
                if t==self._t_switch[-1]: # select the expert to rollout
                    k_star = self.vfn_max.bandit.decision(ob, explore=True)
                    self._k_star.append(k_star)
                    # print('k_star', k_star)
                k_star = self._k_star[-1]
                return self.experts[k_star](ob)

    def logp(self, obs, acs):
        if self._ro_with_policy:
            self._ind_ro_pol.append(self._n_ro)
            self._ro_with_policy = False
            logp =  self.policy.logp(obs, acs)
        else: # roll-in policy and roll-out expert
            assert len(self._t_switch)==len(self._scale)
            if len(obs)<=self._t_switch[-1]:
                # not meaningful for updating value
                # use it to compute gradients instead
                assert len(self._t_switch)==len(self._k_star) or\
                       len(self._t_switch)==len(self._k_star)+1
                if len(self._t_switch)==len(self._k_star):
                    del self._k_star[-1]
                del self._t_switch[-1]
                del self._scale[-1]
                self._ind_ro_pol.append(self._n_ro)
                self._ro_with_policy = False
                logp = self.policy.logp(obs, acs)
            else:
                assert len(self._t_switch)==len(self._k_star)
                self._ind_ro_mix.append(self._n_ro)
                self._ro_with_policy = True
                t_switch = self._t_switch[-1]
                k_star = self._k_star[-1]
                logp0 = self.policy.logp(obs[:t_switch], acs[:t_switch])
                logp1 = self.experts[k_star].logp(obs[t_switch:], acs[t_switch:])
                logp = np.concatenate([logp0, logp1])
        self._locked =False
        self._n_ro+=1
        return logp

    def _sample_t_switch(self):
        # Define t_swich and p
        if self._sampling_rule=='cyclic':
            assert self._horizon < float('Inf')
            t_switch = (int(self._cyclic_rate*self._itr)%self._horizon)+1
            p = 1./self._horizon
        elif self._sampling_rule=='exponential':
            beta = self._avg_n_steps.val
            t_switch = int(np.ceil(np.random.exponential(beta)))
            p = 1./beta*np.exp(-t_switch/beta)
            print('exponential', t_switch, beta, p)
        else:
            if self._horizon < float('Inf'):
                p0 = self._gamma**np.arange(self._horizon)
                sump0 = np.sum(p0)
                p0 = p0/sump0
                ind = np.random.multinomial(1,p0)
                t_switch =  np.where(ind==1)[0][0]+1
                p = p0[t_switch-1]
            else:
                t_switch = np.random.geometric(p=1-self._gamma)[0]
                p = self._gamma**t_switch*(1-self._gamma)
        # correct for potential discount factor
        t_switch = min(t_switch, self._horizon)
        if self._horizon < float('Inf'):
            p0 = self._gamma**np.arange(self._horizon)
            sump0 = np.sum(p0)
            p0 = p0/sump0
            pp = p0[t_switch-1]
        else:
            sump0 = 1/(1-self._gamma)
            pp = self._gamma**t_switch*(1-self._gamma)
        scale = (pp/p)*sump0
        self._t_switch.append(t_switch)
        self._scale.append(scale)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                for k, expert in enumerate(self.experts):
                    pi_exp = lambda ob, t, done: expert(ob)
                    ro = gen_ro(pi_exp, logp=expert.logp)
                    self.aes[k].update(ro)
                    self.policy.update(ro['obs_short'])
        self._reset_pi_ro()

    def update(self, ro):
        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        # Mirror descent
        with timed('Update oracle'):
            # Split ro into two phases
            rollouts = ro.to_list()
            ro_mix = [rollouts[i] for i in self._ind_ro_mix]
            ro_pol = [rollouts[i] for i in self._ind_ro_pol]
            # if a rollout too short, it is treated as zero
            ro_exps = [ [] for _ in range(len(self.experts))]
            for r, t, s, k in zipsame(ro_mix, self._t_switch, self._scale, self._k_star):
                assert len(r) >= t
                if len(r)>=t:
                    r = r[t-1:]
                    r.scale = s
                    ro_exps[k].append(r)
                else:
                    ro_pol.append(r)
            if self._use_policy_as_expert:
                ro_pol += ro_exps[-1]
                del ro_exps[-1]
            ro_exps = [Dataset(ro_exp) for ro_exp in ro_exps]
            ro_pol = Dataset(ro_pol)
            self.oracle.update(ro_pol, update_vfn=False, policy=self.policy)
            # update value functions
            EV0, EV1 = [], []
            for k, ro_exp in enumerate(ro_exps):
                _, ev0, ev1 = self.aes[k].update(ro_exp)
                EV0.append(ev0)
                EV1.append(ev1)
            _, ev0, ev1 = self.aes[-1].update(ro_pol)

            # for adaptive sampling
            self._avg_n_steps.update(np.mean([len(r) for r in ro_pol]))

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

        logz.log_tabular('MeanExplainVarianceBefore(AE)', np.mean(EV0))
        logz.log_tabular('MeanExplainVarianceAfter(AE)', np.mean(EV1))
        logz.log_tabular('NumberOfExpertRollouts', np.sum([len(ro) for ro in ro_exps]))
        logz.log_tabular('NumberOfLearnerRollouts', len(ro_pol))

        # reset
        self._reset_pi_ro()
        self._itr+=1
